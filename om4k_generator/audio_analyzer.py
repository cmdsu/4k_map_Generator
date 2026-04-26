import librosa
import numpy as np
from typing import List, Tuple
import soundfile as sf
import io

class AudioAnalyzer:
    def __init__(self, file_bytes: bytes, manual_bpm: float = None, manual_offset: int = None):
        self.file_bytes = file_bytes
        self.manual_bpm = manual_bpm
        self.manual_offset = manual_offset
        self.y, self.sr = self._load_audio()
        
    def _load_audio(self):
        # Prefer soundfile for fast local decoding; fall back to librosa when needed.
        try:
            y, sr = sf.read(io.BytesIO(self.file_bytes), dtype="float32", always_2d=False)
            if getattr(y, "ndim", 1) > 1:
                y = np.mean(y, axis=1)
            return y, sr
        except Exception:
            y, sr = librosa.load(io.BytesIO(self.file_bytes), sr=None, mono=True)
            return y, sr

    def analyze(self) -> dict:
        duration_ms = int((len(self.y) / max(1, self.sr)) * 1000)
        feature_y, feature_sr = self._feature_signal()
        frame_length = 1024
        hop_length = 256

        rms, onset_env = self._frame_features(feature_y, frame_length, hop_length)
        frame_times_ms = self._frame_times_ms(len(rms), feature_sr, hop_length)
        onset_times_ms = self._detect_onsets(onset_env, frame_times_ms)

        if self.manual_bpm:
            bpm = float(self.manual_bpm)
        else:
            bpm = self._estimate_bpm(onset_env, feature_sr, hop_length)

        offset_ms = self._estimate_offset_ms(onset_times_ms)
        if self.manual_offset is not None:
            offset_ms = self.manual_offset

        beat_times_ms = self._build_beat_times(duration_ms, bpm, offset_ms)
        silent_regions = self._detect_silent_regions(rms, frame_times_ms, duration_ms)

        return {
            "bpm": bpm,
            "offset_ms": offset_ms,
            "duration_ms": duration_ms,
            "onset_times_ms": onset_times_ms,
            "beat_times_ms": beat_times_ms,
            "energy_curve": rms.tolist(),
            "silent_regions": silent_regions
        }

    def _feature_signal(self):
        if self.sr <= 11025:
            return self.y.astype(np.float32, copy=False), self.sr

        factor = max(1, int(round(self.sr / 11025)))
        return self.y[::factor].astype(np.float32, copy=False), self.sr // factor

    def _frame_features(self, signal, frame_length: int, hop_length: int):
        if len(signal) < frame_length:
            padded = np.pad(signal, (0, frame_length - len(signal)))
            frames = padded[np.newaxis, :]
        else:
            frame_count = 1 + (len(signal) - frame_length) // hop_length
            shape = (frame_count, frame_length)
            strides = (signal.strides[0] * hop_length, signal.strides[0])
            frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

        window = np.hanning(frame_length).astype(np.float32)
        windowed = frames * window
        rms = np.sqrt(np.mean(windowed * windowed, axis=1))
        spectrum = np.abs(np.fft.rfft(windowed, axis=1))
        flux = np.maximum(0.0, np.diff(spectrum, axis=0)).sum(axis=1)
        onset_env = np.concatenate(([0.0], flux)).astype(np.float32)

        if np.max(onset_env) > 0:
            onset_env /= np.max(onset_env)

        return rms.astype(np.float32), onset_env

    def _frame_times_ms(self, frame_count: int, feature_sr: int, hop_length: int):
        return [int((i * hop_length / max(1, feature_sr)) * 1000) for i in range(frame_count)]

    def _detect_onsets(self, onset_env, frame_times_ms):
        if len(onset_env) < 3:
            return []

        smooth = np.convolve(onset_env, np.ones(7, dtype=np.float32) / 7.0, mode="same")
        baseline = np.convolve(smooth, np.ones(31, dtype=np.float32) / 31.0, mode="same")
        peak_mask = (
            (smooth[1:-1] > smooth[:-2]) &
            (smooth[1:-1] >= smooth[2:]) &
            (smooth[1:-1] > baseline[1:-1] * 1.15)
        )
        peak_indices = np.where(peak_mask)[0] + 1
        return [frame_times_ms[idx] for idx in peak_indices]

    def _estimate_bpm(self, onset_env, feature_sr: int, hop_length: int) -> float:
        if len(onset_env) < 8:
            return 120.0

        centered = onset_env - np.mean(onset_env)
        autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
        frame_rate = feature_sr / max(1, hop_length)
        min_bpm = 60.0
        max_bpm = 220.0
        min_lag = max(1, int(frame_rate * 60.0 / max_bpm))
        max_lag = min(len(autocorr) - 1, int(frame_rate * 60.0 / min_bpm))
        if max_lag <= min_lag:
            return 120.0

        lag = min_lag + int(np.argmax(autocorr[min_lag:max_lag + 1]))
        bpm = 60.0 * frame_rate / max(1, lag)

        if bpm < 90.0:
            bpm *= 2.0
        elif bpm > 210.0:
            bpm /= 2.0

        return float(max(60.0, min(240.0, bpm)))

    def _estimate_offset_ms(self, onset_times_ms):
        return onset_times_ms[0] if onset_times_ms else 0

    def _build_beat_times(self, duration_ms: int, bpm: float, offset_ms: int):
        beat_times_ms = []
        step = 60000.0 / max(1.0, bpm)
        t = float(offset_ms)
        while t < duration_ms:
            beat_times_ms.append(int(round(t)))
            t += step
        return beat_times_ms

    def _detect_silent_regions(self, rms, frame_times_ms, duration_ms: int):
        threshold = float(np.mean(rms) * 0.2)
        min_silence_ms = 1500
        silent_regions = []
        in_silence = False
        start_sil = 0

        for t_ms, eng in zip(frame_times_ms, rms):
            if eng < threshold and not in_silence:
                in_silence = True
                start_sil = t_ms
            elif eng >= threshold and in_silence:
                in_silence = False
                if t_ms - start_sil >= min_silence_ms:
                    silent_regions.append((start_sil, t_ms))

        if in_silence:
            if duration_ms - start_sil >= min_silence_ms:
                silent_regions.append((start_sil, duration_ms))

        return silent_regions

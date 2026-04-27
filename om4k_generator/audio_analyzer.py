import librosa
import numpy as np
from typing import List, Tuple
import soundfile as sf
import io
import math

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

        (
            rms,
            onset_env,
            bass_onset_env,
            sustain_curve,
            vocal_sustain_curve,
            release_curve,
        ) = self._frame_features(feature_y, feature_sr, frame_length, hop_length)
        frame_times_ms = self._frame_times_ms(len(rms), feature_sr, hop_length)
        onset_peaks = self._detect_onset_peaks(onset_env, bass_onset_env, frame_times_ms)
        onset_times_ms = [time_ms for time_ms, _, _ in onset_peaks]

        if self.manual_bpm:
            bpm = float(self.manual_bpm)
            offset_ms = self._estimate_offset_ms(onset_peaks, bpm)
        else:
            bpm, offset_ms = self._estimate_timing(
                onset_env,
                onset_peaks,
                feature_sr,
                hop_length,
            )
        if self.manual_offset is not None:
            offset_ms = self.manual_offset

        beat_times_ms = self._build_beat_times(duration_ms, bpm, offset_ms)
        silent_regions = self._detect_silent_regions(rms, frame_times_ms, duration_ms)
        silent_regions = self._protect_leading_silence(silent_regions, onset_times_ms, duration_ms)
        release_points = self._detect_release_points(release_curve, frame_times_ms)
        sustain_segments = self._detect_sustain_segments(sustain_curve, vocal_sustain_curve, frame_times_ms, duration_ms)

        return {
            "bpm": bpm,
            "offset_ms": offset_ms,
            "duration_ms": duration_ms,
            "onset_times_ms": onset_times_ms,
            "onset_peaks": [
                {"time_ms": int(time_ms), "strength": float(strength), "bass": float(bass)}
                for time_ms, strength, bass in onset_peaks
            ],
            "beat_times_ms": beat_times_ms,
            "energy_curve": rms.tolist(),
            "sustain_curve": sustain_curve.tolist(),
            "vocal_sustain_curve": vocal_sustain_curve.tolist(),
            "release_curve": release_curve.tolist(),
            "release_points_ms": release_points,
            "sustain_segments": sustain_segments,
            "silent_regions": silent_regions
        }

    def _feature_signal(self):
        if self.sr <= 11025:
            return self.y.astype(np.float32, copy=False), self.sr

        factor = max(1, int(round(self.sr / 11025)))
        return self.y[::factor].astype(np.float32, copy=False), self.sr // factor

    def _frame_features(self, signal, feature_sr: int, frame_length: int, hop_length: int):
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

        freqs = np.fft.rfftfreq(frame_length, d=1.0 / max(1, feature_sr))
        bass_bins = freqs <= 220.0
        if np.any(bass_bins):
            bass_energy = spectrum[:, bass_bins].sum(axis=1)
        else:
            bass_energy = spectrum.sum(axis=1)
        bass_flux = np.maximum(0.0, np.diff(bass_energy))
        bass_onset_env = np.concatenate(([0.0], bass_flux)).astype(np.float32)

        if np.max(onset_env) > 0:
            onset_env /= np.max(onset_env)
        if np.max(bass_onset_env) > 0:
            bass_onset_env /= np.max(bass_onset_env)

        total_energy = spectrum.sum(axis=1) + 1e-9
        mid_bins = (freqs >= 260.0) & (freqs <= 3600.0)
        high_bins = (freqs >= 3600.0) & (freqs <= 9000.0)
        if np.any(mid_bins):
            mid_ratio = spectrum[:, mid_bins].sum(axis=1) / total_energy
        else:
            mid_ratio = np.zeros(len(rms), dtype=np.float32)
        if np.any(high_bins):
            high_ratio = spectrum[:, high_bins].sum(axis=1) / total_energy
        else:
            high_ratio = np.zeros(len(rms), dtype=np.float32)
        if np.any(bass_bins):
            bass_ratio = spectrum[:, bass_bins].sum(axis=1) / total_energy
        else:
            bass_ratio = np.zeros(len(rms), dtype=np.float32)

        rms_norm = self._normalize_curve(rms)
        onset_smooth = self._smooth_curve(onset_env, 5)
        rms_smooth = self._smooth_curve(rms_norm, 9)
        rms_delta = np.abs(np.diff(rms_smooth, prepend=rms_smooth[:1]))
        stability = 1.0 - self._normalize_curve(rms_delta)
        sustain_curve = rms_smooth * (1.0 - np.minimum(0.82, onset_smooth * 0.82)) * (0.58 + stability * 0.42)
        mid_focus = np.clip(mid_ratio * 1.35 - high_ratio * 0.28 - bass_ratio * 0.20, 0.0, 1.0)
        vocal_sustain_curve = sustain_curve * (0.35 + mid_focus * 0.65)

        sustain_curve = self._normalize_curve(self._smooth_curve(sustain_curve, 7))
        vocal_sustain_curve = self._normalize_curve(self._smooth_curve(vocal_sustain_curve, 7))
        sustain_drop = np.maximum(0.0, np.diff(sustain_curve, prepend=sustain_curve[:1]) * -1.0)
        vocal_drop = np.maximum(0.0, np.diff(vocal_sustain_curve, prepend=vocal_sustain_curve[:1]) * -1.0)
        release_curve = sustain_drop * 0.66 + vocal_drop * 0.34
        release_curve *= (0.45 + np.roll(sustain_curve, 1) * 0.55)
        release_curve[0] = 0.0
        release_curve = self._normalize_curve(self._smooth_curve(release_curve, 5))

        return (
            rms.astype(np.float32),
            onset_env.astype(np.float32),
            bass_onset_env.astype(np.float32),
            sustain_curve.astype(np.float32),
            vocal_sustain_curve.astype(np.float32),
            release_curve.astype(np.float32),
        )

    def _frame_times_ms(self, frame_count: int, feature_sr: int, hop_length: int):
        return [int((i * hop_length / max(1, feature_sr)) * 1000) for i in range(frame_count)]

    def _detect_onsets(self, onset_env, frame_times_ms):
        return [time_ms for time_ms, _, _ in self._detect_onset_peaks(onset_env, onset_env, frame_times_ms)]

    @staticmethod
    def _smooth_curve(values, width: int):
        values = np.asarray(values, dtype=np.float32)
        if len(values) == 0 or width <= 1:
            return values.astype(np.float32)
        kernel = np.ones(width, dtype=np.float32) / float(width)
        return np.convolve(values, kernel, mode="same").astype(np.float32)

    @staticmethod
    def _normalize_curve(values):
        values = np.asarray(values, dtype=np.float32)
        if len(values) == 0:
            return values.astype(np.float32)
        low = float(np.percentile(values, 10))
        high = float(np.percentile(values, 92))
        if high <= low + 1e-9:
            max_value = float(np.max(values))
            if max_value <= 1e-9:
                return np.zeros_like(values, dtype=np.float32)
            return np.clip(values / max_value, 0.0, 1.0).astype(np.float32)
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    def _detect_release_points(self, release_curve, frame_times_ms):
        if len(release_curve) < 3:
            return []

        smooth = self._smooth_curve(release_curve, 3)
        threshold = max(0.16, float(np.mean(smooth) + np.std(smooth) * 0.45))
        peak_mask = (
            (smooth[1:-1] >= smooth[:-2]) &
            (smooth[1:-1] > smooth[2:]) &
            (smooth[1:-1] >= threshold)
        )
        peak_indices = np.where(peak_mask)[0] + 1
        ranked = sorted(
            peak_indices,
            key=lambda idx: float(smooth[idx]),
            reverse=True,
        )

        selected = []
        for idx in ranked:
            time_ms = int(frame_times_ms[idx])
            if any(abs(time_ms - item["time_ms"]) < 70 for item in selected):
                continue
            selected.append({"time_ms": time_ms, "strength": float(max(0.0, min(1.0, smooth[idx])))})

        selected.sort(key=lambda item: item["time_ms"])
        return selected

    def _detect_sustain_segments(self, sustain_curve, vocal_sustain_curve, frame_times_ms, duration_ms: int):
        if len(sustain_curve) == 0 or not frame_times_ms:
            return []

        combined = self._smooth_curve(np.maximum(sustain_curve, vocal_sustain_curve * 1.08), 7)
        threshold = max(0.20, float(np.percentile(combined, 62)))
        release_threshold = max(0.12, threshold * 0.58)
        min_duration_ms = 180
        segments = []
        start_index = None

        for idx, value in enumerate(combined):
            if start_index is None and value >= threshold:
                start_index = idx
            elif start_index is not None and value < release_threshold:
                start_ms = int(frame_times_ms[start_index])
                end_ms = int(frame_times_ms[idx])
                if end_ms - start_ms >= min_duration_ms:
                    segment_values = combined[start_index:idx + 1]
                    vocal_values = vocal_sustain_curve[start_index:idx + 1]
                    segments.append({
                        "start_ms": start_ms,
                        "end_ms": min(end_ms, duration_ms),
                        "score": float(max(0.0, min(1.0, np.mean(segment_values)))),
                        "vocal": float(max(0.0, min(1.0, np.mean(vocal_values)))),
                    })
                start_index = None

        if start_index is not None:
            start_ms = int(frame_times_ms[start_index])
            end_ms = int(duration_ms)
            if end_ms - start_ms >= min_duration_ms:
                segment_values = combined[start_index:]
                vocal_values = vocal_sustain_curve[start_index:]
                segments.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "score": float(max(0.0, min(1.0, np.mean(segment_values)))),
                    "vocal": float(max(0.0, min(1.0, np.mean(vocal_values)))),
                })
        return segments

    def _detect_onset_peaks(self, onset_env, bass_onset_env, frame_times_ms):
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
        peaks = []
        for idx in peak_indices:
            onset_strength = float(smooth[idx])
            bass_strength = float(bass_onset_env[idx]) if idx < len(bass_onset_env) else 0.0
            peaks.append((frame_times_ms[idx], onset_strength, bass_strength))
        return peaks

    def _estimate_bpm(self, onset_env, feature_sr: int, hop_length: int) -> float:
        candidates = self._estimate_bpm_candidates(onset_env, feature_sr, hop_length)
        if not candidates:
            return 120.0
        return candidates[0][0]

    def _estimate_timing(self, onset_env, onset_peaks, feature_sr: int, hop_length: int) -> Tuple[float, int]:
        candidates = self._estimate_bpm_candidates(onset_env, feature_sr, hop_length)
        if not candidates:
            bpm = 120.0
            return bpm, self._estimate_offset_ms(onset_peaks, bpm)

        best_bpm = candidates[0][0]
        best_offset = self._estimate_offset_ms(onset_peaks, best_bpm)
        best_score = -1.0

        for bpm, tempo_strength in candidates:
            offset, phase_score = self._estimate_offset_for_bpm(onset_peaks, bpm)
            score = (phase_score * 0.82) + (tempo_strength * 0.18)
            if score > best_score:
                best_score = score
                best_bpm = bpm
                best_offset = offset

        return float(best_bpm), int(best_offset)

    def _estimate_bpm_candidates(self, onset_env, feature_sr: int, hop_length: int) -> List[Tuple[float, float]]:
        if len(onset_env) < 8:
            return [(120.0, 0.0)]

        centered = onset_env - np.mean(onset_env)
        autocorr = self._autocorrelate(centered)
        frame_rate = feature_sr / max(1, hop_length)
        min_bpm = 60.0
        max_bpm = 240.0
        min_lag = max(1, int(frame_rate * 60.0 / max_bpm))
        max_lag = min(len(autocorr) - 1, int(frame_rate * 60.0 / min_bpm))
        if max_lag <= min_lag:
            return [(120.0, 0.0)]

        region = autocorr[min_lag:max_lag + 1]
        if len(region) == 0:
            return [(120.0, 0.0)]

        peak_mask = np.ones(len(region), dtype=bool)
        if len(region) > 2:
            peak_mask[1:-1] = (region[1:-1] >= region[:-2]) & (region[1:-1] >= region[2:])
            peak_mask[0] = region[0] >= region[1]
            peak_mask[-1] = region[-1] >= region[-2]
        peak_indices = np.where(peak_mask)[0]
        if len(peak_indices) == 0:
            peak_indices = np.array([int(np.argmax(region))])

        top_count = min(12, len(peak_indices))
        ranked = peak_indices[np.argsort(region[peak_indices])[-top_count:]][::-1]
        normalizer = float(max(1e-9, autocorr[0]))
        candidates: dict[float, float] = {}

        for local_idx in ranked:
            lag = min_lag + int(local_idx)
            raw_bpm = 60.0 * frame_rate / max(1, lag)
            base_strength = max(0.0, float(region[local_idx]) / normalizer)
            for multiplier, penalty in [(1.0, 1.0), (2.0, 0.88), (0.5, 0.82), (1.5, 0.72), (2.0 / 3.0, 0.70)]:
                bpm = raw_bpm * multiplier
                if bpm < min_bpm or bpm > max_bpm:
                    continue
                bpm = self._normalize_bpm(bpm)
                key = round(bpm, 3)
                candidates[key] = max(candidates.get(key, 0.0), base_strength * penalty)

        if not candidates:
            return [(120.0, 0.0)]

        return sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:18]

    @staticmethod
    def _autocorrelate(values):
        values = np.asarray(values, dtype=np.float32)
        if len(values) == 0:
            return np.asarray([], dtype=np.float32)
        n_fft = 1 << int(math.ceil(math.log2(max(1, len(values) * 2 - 1))))
        spectrum = np.fft.rfft(values, n=n_fft)
        autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft)[: len(values)]
        return autocorr.astype(np.float32)

    @staticmethod
    def _normalize_bpm(bpm: float) -> float:
        if bpm < 90.0:
            bpm *= 2.0
        elif bpm > 210.0:
            bpm /= 2.0

        return float(max(60.0, min(240.0, bpm)))

    def _estimate_offset_ms(self, onset_peaks, bpm: float = 120.0):
        offset, _ = self._estimate_offset_for_bpm(onset_peaks, bpm)
        return offset

    def _estimate_offset_for_bpm(self, onset_peaks, bpm: float) -> Tuple[int, float]:
        if not onset_peaks:
            return 0, 0.0

        beat_length = 60000.0 / max(1.0, bpm)
        if beat_length <= 0:
            return int(onset_peaks[0][0]), 0.0

        peak_times, weights, bass_weights = self._weighted_timing_peaks(onset_peaks)
        if len(peak_times) == 0:
            return int(onset_peaks[0][0]), 0.0

        phases = self._candidate_phases(peak_times, weights, beat_length)
        if not phases:
            phases = [float(peak_times[0] % beat_length)]

        best_phase = phases[0]
        best_score = -1.0
        for phase in phases:
            refined_phase, score = self._refine_phase(peak_times, weights, bass_weights, beat_length, phase)
            if score > best_score:
                best_score = score
                best_phase = refined_phase

        offset = self._canonical_offset(best_phase, beat_length, peak_times, weights)
        return int(round(offset)), float(best_score)

    def _weighted_timing_peaks(self, onset_peaks):
        peaks = sorted(onset_peaks, key=lambda item: item[1] * (0.75 + item[2]), reverse=True)
        peaks = peaks[: min(500, len(peaks))]
        peaks.sort(key=lambda item: item[0])

        times = np.asarray([item[0] for item in peaks], dtype=np.float32)
        onset_strengths = np.asarray([item[1] for item in peaks], dtype=np.float32)
        bass_strengths = np.asarray([item[2] for item in peaks], dtype=np.float32)

        if len(times) == 0:
            return times, onset_strengths, bass_strengths

        if np.max(onset_strengths) > 0:
            onset_strengths = onset_strengths / np.max(onset_strengths)
        if np.max(bass_strengths) > 0:
            bass_strengths = bass_strengths / np.max(bass_strengths)

        weights = onset_strengths * (0.55 + 1.25 * bass_strengths)
        weights = np.maximum(weights, 1e-4)
        bass_weights = np.maximum(bass_strengths, 1e-4)
        return times, weights, bass_weights

    def _candidate_phases(self, peak_times, weights, beat_length: float) -> List[float]:
        bin_ms = 5.0
        bin_count = max(8, int(math.ceil(beat_length / bin_ms)))
        histogram = np.zeros(bin_count, dtype=np.float32)

        for time_ms, weight in zip(peak_times, weights):
            phase = float(time_ms % beat_length)
            center = int(round((phase / beat_length) * bin_count)) % bin_count
            for delta, scale in [(-1, 0.35), (0, 1.0), (1, 0.35)]:
                histogram[(center + delta) % bin_count] += float(weight) * scale

        top_count = min(16, bin_count)
        top_bins = np.argsort(histogram)[-top_count:][::-1]
        phases = [float((idx / bin_count) * beat_length) for idx in top_bins]

        strong_count = min(24, len(peak_times))
        strong_indices = np.argsort(weights)[-strong_count:]
        for idx in strong_indices:
            phases.append(float(peak_times[idx] % beat_length))

        unique_phases = []
        for phase in phases:
            phase = phase % beat_length
            if all(abs(self._phase_distance(phase, existing, beat_length)) > 3.0 for existing in unique_phases):
                unique_phases.append(phase)
        return unique_phases[:32]

    def _refine_phase(self, peak_times, weights, bass_weights, beat_length: float, phase: float) -> Tuple[float, float]:
        best_phase = phase % beat_length
        best_score = self._phase_score(peak_times, weights, bass_weights, beat_length, best_phase)

        for delta in range(-45, 46, 3):
            candidate = (phase + delta) % beat_length
            score = self._phase_score(peak_times, weights, bass_weights, beat_length, candidate)
            if score > best_score:
                best_score = score
                best_phase = candidate

        for delta in range(-4, 5):
            candidate = (best_phase + delta) % beat_length
            score = self._phase_score(peak_times, weights, bass_weights, beat_length, candidate)
            if score > best_score:
                best_score = score
                best_phase = candidate

        return best_phase, best_score

    def _phase_score(self, peak_times, weights, bass_weights, beat_length: float, phase: float) -> float:
        distances = np.abs(((peak_times - phase + beat_length / 2.0) % beat_length) - beat_length / 2.0)
        sigma = max(16.0, min(42.0, beat_length * 0.075))
        closeness = np.exp(-0.5 * (distances / sigma) ** 2)
        near = distances <= max(28.0, min(70.0, beat_length * 0.16))

        weight_sum = float(np.sum(weights)) or 1.0
        bass_sum = float(np.sum(bass_weights)) or 1.0
        onset_score = float(np.sum(weights * closeness) / weight_sum)
        bass_score = float(np.sum(bass_weights * closeness) / bass_sum)
        coverage = float(np.sum(weights * near) / weight_sum)
        return (onset_score * 0.45) + (bass_score * 0.35) + (coverage * 0.20)

    @staticmethod
    def _phase_distance(a: float, b: float, period: float) -> float:
        return ((a - b + period / 2.0) % period) - period / 2.0

    def _canonical_offset(self, phase: float, beat_length: float, peak_times, weights) -> float:
        if len(peak_times) == 0:
            return phase

        first_anchor = float(peak_times[0])

        offset = phase + round((first_anchor - phase) / beat_length) * beat_length
        while offset > first_anchor + beat_length * 0.25:
            offset -= beat_length
        while offset < first_anchor - beat_length * 0.75:
            offset += beat_length
        return offset

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

    def _protect_leading_silence(self, silent_regions, onset_times_ms, duration_ms: int):
        if not onset_times_ms:
            return silent_regions

        first_onset = max(0, int(onset_times_ms[0]) - 80)
        if first_onset < 400:
            return silent_regions

        protected = [(0, min(first_onset, duration_ms))]
        for start, end in silent_regions:
            if end <= protected[0][1]:
                continue
            if start <= protected[-1][1]:
                protected[-1] = (protected[-1][0], max(protected[-1][1], end))
            else:
                protected.append((start, end))
        return protected

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
        # Using soundfile for basic formats, or librosa wrapper
        y, sr = librosa.load(io.BytesIO(self.file_bytes), sr=None, mono=True)
        return y, sr

    def analyze(self) -> dict:
        duration_ms = int(librosa.get_duration(y=self.y, sr=self.sr) * 1000)
        
        # 1. Onsets
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
        onset_times_ms = [int(t * 1000) for t in librosa.frames_to_time(onset_frames, sr=self.sr)]
        
        # 2. BPM & Offset
        beat_times_ms = []
        if self.manual_bpm:
            bpm = self.manual_bpm
            if self.manual_offset is not None:
                offset_ms = self.manual_offset
            else:
                # find first beat offset
                offset_ms = min(onset_times_ms) if onset_times_ms else 0
            t = float(offset_ms)
            step = 60000.0 / bpm
            while t < duration_ms:
                beat_times_ms.append(int(t))
                t += step
        else:
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
            bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            # Find offset from beats
            beat_times_ms = [int(t * 1000) for t in librosa.frames_to_time(beats, sr=self.sr)]
            offset_ms = self.manual_offset if self.manual_offset is not None else (beat_times_ms[0] if beat_times_ms else 0)
            
            # fill beats to end
            if beat_times_ms:
                step = 60000.0 / bpm
                t = float(beat_times_ms[-1]) + step
                while t < duration_ms:
                    beat_times_ms.append(int(t))
                    t += step
        
        # 3. RMS Energy & Silent Regions
        rms = librosa.feature.rms(y=self.y)[0]
        energy_times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr)
        threshold = np.mean(rms) * 0.2 # 20% of mean as silent threshold
        
        silent_regions = []
        in_silence = False
        start_sil = 0
        
        for i, eng in enumerate(rms):
            t_ms = int(energy_times[i] * 1000)
            if eng < threshold and not in_silence:
                in_silence = True
                start_sil = t_ms
            elif eng >= threshold and in_silence:
                in_silence = False
                silent_regions.append((start_sil, t_ms))
        if in_silence:
            silent_regions.append((start_sil, duration_ms))
            
        return {
            "bpm": bpm,
            "offset_ms": offset_ms,
            "duration_ms": duration_ms,
            "onset_times_ms": onset_times_ms,
            "beat_times_ms": beat_times_ms,
            "energy_curve": rms.tolist(),
            "silent_regions": silent_regions
        }

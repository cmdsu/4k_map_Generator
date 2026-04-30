from dataclasses import dataclass
from typing import Literal, List, Optional

@dataclass
class DifficultyConfig:
    version: str
    target_star: Optional[float]
    target_msd: Optional[float]
    chart_type: Literal["rice", "ln"]
    key_style: Optional[Literal["jack", "tech", "speed", "stream"]]
    allowed_subdivisions: List[str]
    chord_enabled: bool
    max_chord_size: int
    chord_probability: float
    max_jack_length: int
    max_anchor_length: int
    hand_balance: float
    ln_ratio: float
    min_ln_ms: int
    max_ln_ms: int
    pattern_temperature: float = 0.35
    music_influence: float = 1.0

@dataclass
class NoteObject:
    time_ms: int
    lane: int
    end_time_ms: Optional[int] = None

    @property
    def is_ln(self) -> bool:
        return self.end_time_ms is not None

@dataclass
class AudioAnalysisResult:
    bpm: float
    offset_ms: int
    duration_ms: int
    onset_times_ms: List[int]
    energy_curve: List[float]
    silent_regions: List[tuple[int, int]]
    beat_grid_ms: List[int]
    snap_points_ms: List[int]


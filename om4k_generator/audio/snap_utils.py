from typing import Any, Dict, List, Set

from .grid_builder import GridBuilder
from ..core.models import DifficultyConfig
from ..core.style_rules import recommended_subdivisions
from ..core.calibration_utils import (
    _allowed_subdivision_denominators,
    _energy_score_at,
    _nearest_distance,
    _nearest_snap_point,
)


def build_snap_candidates(analysis: Dict[str, Any], config: DifficultyConfig) -> List[int]:
    allowed_subdivisions = list(config.allowed_subdivisions)
    if not allowed_subdivisions:
        allowed_subdivisions = recommended_subdivisions(
            analysis["bpm"],
            config.chart_type,
            config.key_style,
            config.target_star,
        )
    if config.key_style == "tech":
        allowed_subdivisions = [
            subdivision
            for subdivision in allowed_subdivisions
            if subdivision in ["1/4", "1/6", "1/8"]
        ] or ["1/4", "1/6", "1/8"]
    allowed_denominators = _allowed_subdivision_denominators(allowed_subdivisions)

    grid = GridBuilder.build(
        analysis["bpm"],
        analysis["offset_ms"],
        analysis["duration_ms"],
        allowed_subdivisions,
    )

    combined_candidates = analysis["onset_times_ms"] + analysis["beat_times_ms"]
    beat_length = 60000.0 / analysis["bpm"]
    for divisor in allowed_denominators:
        t_style = float(analysis["offset_ms"])
        style_step = beat_length / divisor
        while t_style < analysis["duration_ms"]:
            combined_candidates.append(int(round(t_style)))
            t_style += style_step

    combined_candidates = sorted(set(combined_candidates))
    return GridBuilder.snap(combined_candidates, grid)


def build_accent_snap_points(analysis: Dict[str, Any], snap_points: List[int]) -> Set[int]:
    if not snap_points:
        return set()

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    max_distance = max(35, min(70, int(beat_length * 0.12)))
    accents: Set[int] = set()
    beat_times = [int(t) for t in analysis.get("beat_times_ms", [])]

    for time_ms in beat_times:
        closest = _nearest_snap_point(snap_points, int(time_ms))
        if closest is not None and abs(closest - int(time_ms)) <= max_distance:
            accents.add(closest)

    for time_ms in analysis.get("onset_times_ms", []):
        time_ms = int(time_ms)
        near_beat = _nearest_distance(beat_times, time_ms)
        energy_score = _energy_score_at(analysis, time_ms)
        if near_beat is not None and near_beat <= max_distance:
            pass
        elif energy_score < 0.65:
            continue

        closest = _nearest_snap_point(snap_points, time_ms)
        if closest is not None and abs(closest - time_ms) <= max_distance:
            accents.add(closest)

    return accents

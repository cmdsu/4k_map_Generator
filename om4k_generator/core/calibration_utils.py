import bisect
from typing import Any, Dict, List, Tuple

from .models import DifficultyConfig, NoteObject


def _deterministic_index(modulo: int, *parts: int) -> int:
    if modulo <= 0:
        return 0
    value = 0x345678
    for part in parts:
        value = ((value ^ int(part)) * 1000003) & 0x7FFFFFFF
    return value % modulo


def _flatten_note_rows(rows: Dict[int, List[NoteObject]]) -> List[NoteObject]:
    return [
        note
        for time_ms in sorted(rows)
        for note in sorted(rows[time_ms], key=lambda item: (item.lane, item.end_time_ms or -1))
    ]


def _time_in_regions(time_ms: int, regions: List[Tuple[int, int]]) -> bool:
    return any(start <= time_ms <= end for start, end in regions)


def _nearest_snap_point(snap_points: List[int], time_ms: int) -> int | None:
    idx = bisect.bisect_left(snap_points, time_ms)
    candidates = []
    if idx < len(snap_points):
        candidates.append(snap_points[idx])
    if idx > 0:
        candidates.append(snap_points[idx - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: abs(candidate - time_ms))


def _nearest_distance(values: List[int], time_ms: int) -> int | None:
    if not values:
        return None
    idx = bisect.bisect_left(values, time_ms)
    candidates = []
    if idx < len(values):
        candidates.append(abs(values[idx] - time_ms))
    if idx > 0:
        candidates.append(abs(values[idx - 1] - time_ms))
    return min(candidates) if candidates else None


def _energy_score_at(analysis: Dict[str, Any], time_ms: int) -> float:
    cache = analysis.get("_energy_score_cache")
    curve_len = len(analysis.get("energy_curve", []))
    if not cache or cache.get("curve_len") != curve_len or cache.get("duration_ms") != analysis.get("duration_ms"):
        curve = [float(v) for v in analysis.get("energy_curve", [])]
        if curve and analysis["duration_ms"] > 0:
            values = sorted(curve)
            low = values[min(len(values) - 1, int(len(values) * 0.25))]
            high = values[min(len(values) - 1, int(len(values) * 0.90))]
        else:
            low = 0.0
            high = 1.0
        cache = {
            "curve_len": curve_len,
            "duration_ms": analysis.get("duration_ms"),
            "curve": curve,
            "low": low,
            "high": high,
        }
        analysis["_energy_score_cache"] = cache

    curve = cache.get("curve", [])
    if not curve or analysis["duration_ms"] <= 0:
        return 1.0

    low = float(cache.get("low", 0.0))
    high = float(cache.get("high", 1.0))
    if high <= low:
        return 0.5

    idx = int((time_ms / analysis["duration_ms"]) * (len(curve) - 1))
    idx = max(0, min(len(curve) - 1, idx))
    return max(0.0, min(1.0, (curve[idx] - low) / (high - low)))


def _analysis_curve_value(
    analysis: Dict[str, Any],
    curve_key: str,
    time_ms: int,
    default: float = 0.0,
) -> float:
    curve = analysis.get(curve_key, [])
    if not curve or analysis.get("duration_ms", 0) <= 0:
        return default
    idx = int((max(0, time_ms) / max(1, analysis["duration_ms"])) * (len(curve) - 1))
    idx = max(0, min(len(curve) - 1, idx))
    try:
        value = float(curve[idx])
    except (TypeError, ValueError, IndexError):
        return default
    return max(0.0, min(1.0, value))


def _analysis_curve_average(
    analysis: Dict[str, Any],
    curve_key: str,
    start_ms: int,
    end_ms: int,
    default: float = 0.0,
) -> float:
    curve = analysis.get(curve_key, [])
    duration_ms = int(analysis.get("duration_ms", 0))
    if not curve or duration_ms <= 0 or end_ms <= start_ms:
        return default
    start_idx = int((max(0, start_ms) / duration_ms) * (len(curve) - 1))
    end_idx = int((max(0, end_ms) / duration_ms) * (len(curve) - 1))
    start_idx = max(0, min(len(curve) - 1, start_idx))
    end_idx = max(start_idx, min(len(curve) - 1, end_idx))
    window = curve[start_idx:end_idx + 1]
    if not window:
        return default
    try:
        average = sum(float(value) for value in window) / len(window)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, average))


def _target_skeleton_divisor(config: DifficultyConfig) -> int:
    allowed_denominators = _allowed_subdivision_denominators(config.allowed_subdivisions)
    max_allowed = max(allowed_denominators) if allowed_denominators else 4
    target = config.target_star or 3.0

    if target < 2.0:
        desired = 1
    elif target < 3.0:
        desired = 2
    elif target < 5.0:
        desired = 4
    else:
        desired = 8

    return max(1, min(max_allowed, desired))


def _allowed_subdivision_denominators(subdivisions: List[str]) -> List[int]:
    denominators: List[int] = []
    for subdivision in subdivisions:
        try:
            numerator, denominator = subdivision.split("/")
            if int(numerator) == 1 and int(denominator) > 0:
                denominators.append(int(denominator))
        except ValueError:
            continue
    return sorted(set(denominators)) or [1, 2, 4]


def _nearest_allowed_divisor(preferred: int, subdivisions: List[str]) -> int:
    allowed = _allowed_subdivision_denominators(subdivisions)
    if preferred in allowed:
        return preferred
    return sorted(
        allowed,
        key=lambda divisor: (
            abs(divisor - preferred),
            divisor < preferred,
            divisor,
        ),
    )[0]

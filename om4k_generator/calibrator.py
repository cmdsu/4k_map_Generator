import random
import bisect
import math
from dataclasses import replace
from typing import Any, Dict, List, Set, Tuple

from .difficulty_estimator import DifficultyEstimator
from .grid_builder import GridBuilder
from .models import DifficultyConfig, NoteObject
from .pattern_generator import PatternGenerator
from .style_rules import recommended_subdivisions
from .validator import Validator


def build_snap_candidates(analysis: Dict[str, Any], config: DifficultyConfig) -> List[int]:
    allowed_subdivisions = config.allowed_subdivisions
    if config.key_style == "jack":
        style_subdivisions = recommended_subdivisions(
            analysis["bpm"],
            config.chart_type,
            config.key_style,
            None,
        )
        filtered = [subdivision for subdivision in allowed_subdivisions if subdivision in style_subdivisions]
        allowed_subdivisions = filtered or style_subdivisions
    elif config.target_star is None:
        allowed_subdivisions = recommended_subdivisions(
            analysis["bpm"],
            config.chart_type,
            config.key_style,
            config.target_star,
        )

    grid = GridBuilder.build(
        analysis["bpm"],
        analysis["offset_ms"],
        analysis["duration_ms"],
        allowed_subdivisions,
    )

    combined_candidates = analysis["onset_times_ms"] + analysis["beat_times_ms"]
    beat_length = 60000.0 / analysis["bpm"]
    divisor = _target_skeleton_divisor(config)
    step = beat_length / divisor

    t_extra = float(analysis["offset_ms"])
    while t_extra < analysis["duration_ms"]:
        combined_candidates.append(int(round(t_extra)))
        t_extra += step

    combined_candidates = sorted(set(combined_candidates))
    return GridBuilder.snap(combined_candidates, grid)


def generate_to_target_sr(
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    tolerance: float = 0.10,
    max_attempts: int = 24,
) -> Tuple[List[NoteObject], float, bool, int]:
    accent_snap_points = build_accent_snap_points(analysis, snap_points)

    if config.target_star is None:
        generator = PatternGenerator(config)
        raw_notes = generator.generate(
            snap_points,
            analysis["energy_curve"],
            analysis["silent_regions"],
            accent_times_ms=accent_snap_points,
        )
        clean_notes = Validator.validate_and_fix(raw_notes, config, analysis["silent_regions"], snap_points)
        clean_notes = _smooth_jack_gaps(clean_notes, analysis, snap_points, accent_snap_points, config)
        return clean_notes, DifficultyEstimator.estimate_sr(clean_notes, analysis["duration_ms"]), True, 1

    target = config.target_star
    best_notes: List[NoteObject] = []
    best_sr = 0.0
    best_diff = float("inf")

    if config.key_style == "jack":
        if target < 3.5:
            density = max(0.10, min(1.5, target / 7.5))
        else:
            density = max(0.12, min(1.5, target / 7.0))
        chord_probability = min(1.0, max(0.88, config.chord_probability + (target - 3.0) * 0.08))
    else:
        density = max(0.15, min(3.0, target / 4.0))
        chord_probability = min(1.0, max(0.0, config.chord_probability + (target - 3.0) * 0.08))

    for attempt in range(1, max_attempts + 1):
        trial_config = replace(config, chord_probability=chord_probability)

        # Keep calibration deterministic enough that identical inputs are debuggable.
        random.seed(20241007 + attempt)
        generator = PatternGenerator(trial_config)
        raw_notes = generator.generate(
            snap_points,
            analysis["energy_curve"],
            analysis["silent_regions"],
            density_multiplier=density,
            accent_times_ms=accent_snap_points,
        )
        clean_notes = Validator.validate_and_fix(raw_notes, trial_config, analysis["silent_regions"], snap_points)
        clean_notes = _smooth_jack_gaps(clean_notes, analysis, snap_points, accent_snap_points, trial_config)
        est_sr = DifficultyEstimator.estimate_sr(clean_notes, analysis["duration_ms"])
        diff = abs(est_sr - target)

        if diff < best_diff:
            best_diff = diff
            best_notes = clean_notes
            best_sr = est_sr

        if diff <= tolerance:
            config.chord_probability = chord_probability
            return clean_notes, est_sr, True, attempt

        if est_sr > target:
            ratio = target / max(est_sr, 0.01)
            density *= max(0.55, min(0.95, ratio))
            chord_probability = max(0.88 if config.key_style == "jack" else 0.0, chord_probability - 0.08)
        else:
            ratio = target / max(est_sr, 0.05)
            density *= min(1.35, max(1.05, ratio ** 0.35))
            chord_probability = min(1.0, chord_probability + 0.08)

    if best_sr < target:
        refined_notes, refined_sr, refined_met = _refine_upward(
            best_notes,
            target,
            tolerance,
            config,
            analysis,
            snap_points,
            accent_snap_points,
        )
        if abs(refined_sr - target) < best_diff:
            best_notes = refined_notes
            best_sr = refined_sr
            best_diff = abs(refined_sr - target)
        if refined_met:
            config.chord_probability = chord_probability
            return refined_notes, refined_sr, True, max_attempts

    config.chord_probability = chord_probability
    return best_notes, best_sr, best_diff <= tolerance, max_attempts


def _refine_upward(
    notes: List[NoteObject],
    target: float,
    tolerance: float,
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> Tuple[List[NoteObject], float, bool]:
    current_notes = list(notes)
    current_sr = DifficultyEstimator.estimate_sr(current_notes, analysis["duration_ms"])
    best_notes = current_notes
    best_sr = current_sr
    best_diff = abs(current_sr - target)

    existing = {}
    for note in current_notes:
        existing.setdefault(note.time_ms, set()).add(note.lane)

    pending: List[NoteObject] = []
    last_lane_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    for note in sorted(current_notes, key=lambda n: (n.time_ms, n.lane)):
        last_lane_time[note.lane] = note.time_ms

    jack_anchor_lanes: List[int] = []
    ordered_times = list(snap_points)
    if config.key_style == "jack":
        empty_accent_times = [time_ms for time_ms in snap_points if time_ms in accent_snap_points and not existing.get(time_ms)]
        occupied_times = [time_ms for time_ms in snap_points if time_ms not in empty_accent_times]
        ordered_times = empty_accent_times + occupied_times

    for time_ms in ordered_times:
        lanes_at_time = existing.setdefault(time_ms, set())
        if len(lanes_at_time) >= config.max_chord_size:
            if config.key_style == "jack":
                jack_anchor_lanes = sorted(lanes_at_time)
            continue

        additions_allowed = config.max_chord_size - len(lanes_at_time)
        if config.key_style == "jack":
            new_lanes = _jack_refine_lanes(
                lanes_at_time,
                jack_anchor_lanes,
                time_ms in accent_snap_points,
                config.max_chord_size,
                target,
            )
        else:
            new_lanes = []
            for lane in sorted(range(4), key=lambda l: last_lane_time[l]):
                if additions_allowed <= 0:
                    break
                if lane in lanes_at_time:
                    continue

                new_lanes.append(lane)
                additions_allowed -= 1

        for lane in new_lanes:
            pending.append(NoteObject(time_ms=time_ms, lane=lane))
            lanes_at_time.add(lane)
            last_lane_time[lane] = time_ms

        if config.key_style == "jack" and lanes_at_time:
            jack_anchor_lanes = sorted(lanes_at_time)

        if len(pending) % 16 != 0:
            continue

        candidate = Validator.validate_and_fix(current_notes + pending, config, analysis["silent_regions"], snap_points)
        candidate = _smooth_jack_gaps(candidate, analysis, snap_points, accent_snap_points, config)
        candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
        candidate_diff = abs(candidate_sr - target)

        if candidate_diff < best_diff:
            best_notes = candidate
            best_sr = candidate_sr
            best_diff = candidate_diff

        if candidate_diff <= tolerance:
            return candidate, candidate_sr, True

        if candidate_sr > target + tolerance:
            return best_notes, best_sr, False

    if pending:
        candidate = Validator.validate_and_fix(current_notes + pending, config, analysis["silent_regions"], snap_points)
        candidate = _smooth_jack_gaps(candidate, analysis, snap_points, accent_snap_points, config)
        candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
        candidate_diff = abs(candidate_sr - target)
        if candidate_diff < best_diff:
            best_notes = candidate
            best_sr = candidate_sr
            best_diff = candidate_diff

    return best_notes, best_sr, best_diff <= tolerance


def _jack_refine_lanes(
    lanes_at_time: Set[int],
    anchor_lanes: List[int],
    is_accent: bool,
    max_chord_size: int,
    target: float,
) -> List[int]:
    if len(lanes_at_time) >= max_chord_size:
        return []

    anchors = [lane for lane in anchor_lanes if 0 <= lane <= 3] or [0]
    ordered = anchors + [lane for lane in [0, 1, 2, 3] if lane not in anchors]

    if not is_accent:
        if lanes_at_time:
            if target < 4.5:
                return []
            desired_chord_size = 2 if target < 5.5 else min(3, max_chord_size)
        else:
            return []
    elif target < 4.5:
        desired_chord_size = 2
    elif target < 5.5:
        desired_chord_size = min(3, max_chord_size)
    elif target < 6.5:
        desired_chord_size = max_chord_size
    else:
        desired_chord_size = max_chord_size

    desired_chord_size = max(2, min(max_chord_size, desired_chord_size))
    additions_needed = max(0, desired_chord_size - len(lanes_at_time))
    if additions_needed <= 0:
        return []

    additions = []
    for lane in ordered:
        if lane in lanes_at_time or lane in additions:
            continue
        additions.append(lane)
        if len(additions) >= additions_needed:
            break
    return additions


def _smooth_jack_gaps(
    notes: List[NoteObject],
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    config: DifficultyConfig,
) -> List[NoteObject]:
    if config.key_style != "jack" or not notes:
        return notes

    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    gap_threshold = int(beat_length * 2.05)
    snap_set = set(snap_points)
    accent_times = sorted(t for t in accent_snap_points if t in snap_set)
    all_snap_times = sorted(snap_set)
    additions: List[NoteObject] = []

    sorted_times = sorted(rows)
    for prev_time, next_time in zip(sorted_times, sorted_times[1:]):
        gap = next_time - prev_time
        if gap < gap_threshold:
            continue

        between = [t for t in accent_times if prev_time < t < next_time]
        if not between:
            between = [t for t in all_snap_times if prev_time < t < next_time]
        if not between:
            continue

        shared = sorted(rows[prev_time] & rows[next_time])
        if shared:
            lane = shared[0]
        else:
            lane = sorted(rows[prev_time])[0]

        inserts_needed = max(1, math.ceil(gap / max(1, gap_threshold)) - 1)
        inserts_needed = min(inserts_needed, len(between))
        if inserts_needed <= 0:
            continue

        chosen_times = []
        for index in range(1, inserts_needed + 1):
            pick = int(round(index * (len(between) + 1) / (inserts_needed + 1))) - 1
            pick = max(0, min(len(between) - 1, pick))
            time_ms = between[pick]
            if time_ms not in chosen_times:
                chosen_times.append(time_ms)

        for time_ms in chosen_times:
            rows.setdefault(time_ms, set()).add(lane)
            additions.append(NoteObject(time_ms=time_ms, lane=lane))

    if not additions:
        return notes

    return Validator.validate_and_fix(notes + additions, config, analysis["silent_regions"], snap_points)


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
    curve = [float(v) for v in analysis.get("energy_curve", [])]
    if not curve or analysis["duration_ms"] <= 0:
        return 1.0

    values = sorted(curve)
    low = values[min(len(values) - 1, int(len(values) * 0.25))]
    high = values[min(len(values) - 1, int(len(values) * 0.90))]
    if high <= low:
        return 0.5

    idx = int((time_ms / analysis["duration_ms"]) * (len(curve) - 1))
    idx = max(0, min(len(curve) - 1, idx))
    return max(0.0, min(1.0, (curve[idx] - low) / (high - low)))


def _target_skeleton_divisor(config: DifficultyConfig) -> int:
    allowed_denominators = []
    for subdivision in config.allowed_subdivisions:
        try:
            numerator, denominator = subdivision.split("/")
            if int(numerator) == 1:
                allowed_denominators.append(int(denominator))
        except ValueError:
            continue

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

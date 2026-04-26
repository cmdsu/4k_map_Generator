import random
from dataclasses import replace
from typing import Any, Dict, List, Tuple

from .difficulty_estimator import DifficultyEstimator
from .grid_builder import GridBuilder
from .models import DifficultyConfig, NoteObject
from .pattern_generator import PatternGenerator
from .style_rules import recommended_subdivisions
from .validator import Validator


def build_snap_candidates(analysis: Dict[str, Any], config: DifficultyConfig) -> List[int]:
    allowed_subdivisions = config.allowed_subdivisions
    if config.target_star is None:
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
    if config.target_star is None:
        generator = PatternGenerator(config)
        raw_notes = generator.generate(snap_points, analysis["energy_curve"], analysis["silent_regions"])
        clean_notes = Validator.validate_and_fix(raw_notes, config, analysis["silent_regions"], snap_points)
        return clean_notes, DifficultyEstimator.estimate_sr(clean_notes, analysis["duration_ms"]), True, 1

    target = config.target_star
    best_notes: List[NoteObject] = []
    best_sr = 0.0
    best_diff = float("inf")

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
        )
        clean_notes = Validator.validate_and_fix(raw_notes, trial_config, analysis["silent_regions"], snap_points)
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
            chord_probability = max(0.0, chord_probability - 0.08)
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

    for time_ms in snap_points:
        lanes_at_time = existing.setdefault(time_ms, set())
        if len(lanes_at_time) >= config.max_chord_size:
            continue

        for lane in sorted(range(4), key=lambda l: last_lane_time[l]):
            if lane in lanes_at_time:
                continue

            pending.append(NoteObject(time_ms=time_ms, lane=lane))
            lanes_at_time.add(lane)
            last_lane_time[lane] = time_ms
            break

        if len(pending) % 16 != 0:
            continue

        candidate = Validator.validate_and_fix(current_notes + pending, config, analysis["silent_regions"], snap_points)
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
        candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
        candidate_diff = abs(candidate_sr - target)
        if candidate_diff < best_diff:
            best_notes = candidate
            best_sr = candidate_sr
            best_diff = candidate_diff

    return best_notes, best_sr, best_diff <= tolerance


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

import random
import bisect
import math
from collections import Counter
from dataclasses import replace
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.difficulty_estimator import DifficultyEstimator
from ..core.models import DifficultyConfig, NoteObject
from .pattern_generator import PatternGenerator
from ..core.style_rules import clamp_max_chord_size
from ..styles.speed_calibrator import generate_speed_to_target_sr
from ..styles.tech_calibrator import generate_tech_to_target_sr
from ..core.validator import Validator
from ..core.calibration_utils import (
    _deterministic_index,
    _flatten_note_rows,
    _nearest_allowed_divisor,
    _nearest_snap_point,
    _time_in_regions,
)
from ..styles.jack_anchor import (
    _choose_jack_pressure_replacement,
    _jack_stack_patterns,
    _rotate_jack_anchor_runs,
    _shape_jack_anchor_contrast,
    _shorten_jack_anchor_runs,
)
from .ln_tools import (
    _apply_safe_lns,
    _rebalance_lns_for_target,
    _refine_lns_to_sustain_and_hits,
    _trim_low_music_notes_for_target,
)
from .music_alignment import (
    _apply_music_influence,
    _chord_music_strength_at,
    _fill_reasonable_gaps,
    _jack_refine_lanes,
    _music_context,
    _music_entry,
    _music_influence,
    _music_receiver_lane_for_style,
    _nearest_music_row_time,
    _pick_chord_weight_donor_note,
)
from ..audio.snap_utils import build_accent_snap_points, build_snap_candidates


def generate_to_target_sr(
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    tolerance: float = 0.15,
    max_attempts: int = 24,
) -> Tuple[List[NoteObject], float, bool, int]:
    accent_snap_points = build_accent_snap_points(analysis, snap_points)
    _music_context(analysis, snap_points, accent_snap_points)

    if config.chart_type in ["rice", "ln"] and config.key_style == "tech":
        return generate_tech_to_target_sr(config, analysis, snap_points, accent_snap_points, tolerance, max_attempts)

    if config.target_star is None:
        generator = PatternGenerator(config)
        raw_notes = generator.generate(
            snap_points,
            analysis["energy_curve"],
            analysis["silent_regions"],
            accent_times_ms=accent_snap_points,
            bpm=analysis["bpm"],
        )
        clean_notes = _finalize_notes(raw_notes, config, analysis, snap_points, accent_snap_points)
        return clean_notes, DifficultyEstimator.estimate_sr(clean_notes, analysis["duration_ms"]), True, 1

    target = config.target_star
    best_notes: List[NoteObject] = []
    best_sr = 0.0
    best_diff = float("inf")


    if config.chart_type in ["rice", "ln"] and config.key_style == "jack":
        jack_notes = _shape_jack_stack_profile([], config, analysis, snap_points, accent_snap_points)
        jack_notes = _refine_lns_to_sustain_and_hits(jack_notes, config, analysis, snap_points, accent_snap_points)
        jack_notes = Validator.validate_and_fix(jack_notes, config, analysis["silent_regions"], snap_points)
        jack_notes, jack_sr = _rebalance_lns_for_target(jack_notes, target, tolerance, config, analysis, snap_points, accent_snap_points)
        jack_notes = _shorten_jack_anchor_runs(jack_notes, config.max_anchor_length, _pattern_temperature(config))
        jack_notes = Validator.validate_and_fix(jack_notes, config, analysis["silent_regions"], snap_points)
        jack_notes = _shorten_jack_anchor_runs(jack_notes, config.max_anchor_length, _pattern_temperature(config))
        jack_notes = Validator.validate_and_fix(jack_notes, config, analysis["silent_regions"], snap_points)
        jack_sr = DifficultyEstimator.estimate_sr(jack_notes, analysis["duration_ms"])
        return jack_notes, jack_sr, abs(jack_sr - target) <= tolerance, 1

    if config.chart_type in ["rice", "ln"] and config.key_style == "stream":
        stream_notes = _shape_stream_profile([], config, analysis, snap_points, accent_snap_points)
        stream_notes = _refine_lns_to_sustain_and_hits(stream_notes, config, analysis, snap_points, accent_snap_points)
        stream_notes = Validator.validate_and_fix(stream_notes, config, analysis["silent_regions"], snap_points)
        stream_notes, stream_sr = _rebalance_lns_for_target(stream_notes, target, tolerance, config, analysis, snap_points, accent_snap_points)
        stream_notes = _enforce_stream_fast_cut_integrity(stream_notes, config, analysis, snap_points, accent_snap_points)
        stream_notes = Validator.validate_and_fix(stream_notes, config, analysis["silent_regions"], snap_points)
        stream_sr = DifficultyEstimator.estimate_sr(stream_notes, analysis["duration_ms"])
        if stream_sr > target + tolerance:
            stream_notes, stream_sr = _trim_low_music_notes_for_target(
                stream_notes,
                target,
                tolerance,
                config,
                analysis,
                snap_points,
                accent_snap_points,
                stream_sr,
            )
            stream_notes = _enforce_stream_fast_cut_integrity(stream_notes, config, analysis, snap_points, accent_snap_points)
            stream_notes = Validator.validate_and_fix(stream_notes, config, analysis["silent_regions"], snap_points)
            stream_sr = DifficultyEstimator.estimate_sr(stream_notes, analysis["duration_ms"])
        return stream_notes, stream_sr, abs(stream_sr - target) <= tolerance, 1

    if config.chart_type in ["rice", "ln"] and config.key_style == "speed":
        return generate_speed_to_target_sr(config, analysis, snap_points, accent_snap_points, tolerance, max_attempts)

    if config.key_style == "jack":
        chord_floor = _jack_chord_probability_floor(target)
        if target < 3.5:
            density = max(0.10, min(1.5, target / 7.5))
        else:
            density = max(0.12, min(1.5, target / 7.0))
        chord_probability = min(1.0, max(chord_floor, config.chord_probability + (target - 3.0) * 0.08))
    else:
        chord_floor = 0.0
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
            bpm=analysis["bpm"],
        )
        clean_notes = _finalize_notes(raw_notes, trial_config, analysis, snap_points, accent_snap_points)
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
            chord_probability = max(chord_floor, chord_probability - 0.08)
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


def _jack_chord_probability_floor(target: float) -> float:
    if target >= 5.5:
        return 0.88
    if target >= 4.5:
        return 0.62
    if target >= 3.5:
        return 0.28
    return 0.08


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
    if _music_influence(config) > 0.05:
        ordered_times = sorted(
            ordered_times,
            key=lambda time_ms: (
                -_music_entry(analysis, snap_points, accent_snap_points, time_ms)["accent"],
                time_ms not in accent_snap_points,
                time_ms,
            ),
        )

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

        candidate = _finalize_notes(current_notes + pending, config, analysis, snap_points, accent_snap_points)
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
        candidate = _finalize_notes(current_notes + pending, config, analysis, snap_points, accent_snap_points)
        candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
        candidate_diff = abs(candidate_sr - target)
        if candidate_diff < best_diff:
            best_notes = candidate
            best_sr = candidate_sr
            best_diff = candidate_diff

    return best_notes, best_sr, best_diff <= tolerance


def _finalize_notes(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    fixed = Validator.validate_and_fix(notes, config, analysis["silent_regions"], snap_points)
    fixed = _fill_reasonable_gaps(fixed, analysis, snap_points, accent_snap_points, config)
    fixed = Validator.validate_and_fix(fixed, config, analysis["silent_regions"], snap_points)
    fixed = _shape_jack_stack_profile(fixed, config, analysis, snap_points, accent_snap_points)
    fixed = _shape_stream_profile(fixed, config, analysis, snap_points, accent_snap_points)
    fixed = _shape_speed_profile(fixed, config, analysis, snap_points, accent_snap_points)
    fixed = _apply_music_influence(fixed, config, analysis, snap_points, accent_snap_points, config.key_style)
    fixed = _refine_lns_to_sustain_and_hits(fixed, config, analysis, snap_points, accent_snap_points)
    fixed = _repair_stream_cut_collisions(fixed, config, analysis)
    fixed = Validator.validate_and_fix(fixed, config, analysis["silent_regions"], snap_points)
    fixed = _repair_stream_cut_collisions(fixed, config, analysis)
    fixed = Validator.validate_and_fix(fixed, config, analysis["silent_regions"], snap_points)
    fixed = _repair_stream_cut_collisions(fixed, config, analysis)
    fixed = _collapse_stream_micro_rows(fixed, config, analysis, snap_points, accent_snap_points)
    fixed = Validator.validate_and_fix(fixed, config, analysis["silent_regions"], snap_points)
    fixed = _stabilize_stream_cut_repairs(fixed, config, analysis)
    fixed = _reinforce_stream_music_chords(fixed, config, analysis, snap_points, accent_snap_points)
    fixed = Validator.validate_and_fix(fixed, config, analysis["silent_regions"], snap_points)
    fixed = _stabilize_stream_cut_repairs(fixed, config, analysis)
    fixed = _enforce_stream_fast_cut_integrity(fixed, config, analysis, snap_points, accent_snap_points)
    return Validator.validate_and_fix(fixed, config, analysis["silent_regions"], snap_points)


def _shape_jack_stack_profile(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    if config.chart_type not in ["rice", "ln"] or config.key_style != "jack" or config.target_star is None:
        return notes

    rows = [
        time_ms
        for time_ms in snap_points
        if not _time_in_regions(time_ms, analysis.get("silent_regions", []))
    ]
    if not rows:
        return notes

    target = config.target_star
    temperature = _pattern_temperature(config)
    max_chord_size = max(1, min(4, config.max_chord_size))
    min_chord_size = _jack_profile_min_chord_size(target, temperature)
    if config.chart_type == "ln":
        min_chord_size = max(1, min_chord_size - 1)
    max_profile_chord_size = min(max_chord_size, _jack_profile_max_chord_size(target))
    if min_chord_size > max_profile_chord_size:
        min_chord_size = max_profile_chord_size

    best_notes = notes
    best_sr = DifficultyEstimator.estimate_sr(notes, analysis["duration_ms"]) if notes else 0.0
    best_diff = abs(best_sr - target)

    lower = float(min_chord_size)
    upper = float(max_profile_chord_size)
    tested_averages: Set[float] = set()

    def evaluate(average_chord_size: float) -> float:
        nonlocal best_notes, best_sr, best_diff
        average_chord_size = round(max(lower, min(upper, average_chord_size)), 3)
        if average_chord_size in tested_averages:
            return best_sr
        tested_averages.add(average_chord_size)

        sizes = _build_jack_chord_sizes(
            row_count=len(rows),
            average_chord_size=average_chord_size,
            min_chord_size=min_chord_size,
            max_chord_size=max_profile_chord_size,
            rows=rows,
            accent_snap_points=accent_snap_points,
            analysis=analysis,
            target=target,
            temperature=temperature,
        )
        candidate = _build_jack_stack_notes(rows, sizes, target, temperature)
        candidate = _apply_safe_lns(candidate, config, analysis, snap_points, "jack")
        candidate = _apply_music_influence(candidate, config, analysis, snap_points, accent_snap_points, "jack")
        candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
        original_candidate = candidate
        original_sr = DifficultyEstimator.estimate_sr(original_candidate, analysis["duration_ms"])
        balanced_candidate = _rebalance_jack_lane_pressure(candidate, target, temperature)
        balanced_candidate = Validator.validate_and_fix(balanced_candidate, config, analysis["silent_regions"], snap_points)
        balanced_sr = DifficultyEstimator.estimate_sr(balanced_candidate, analysis["duration_ms"])
        if target >= 5.75 and balanced_sr < target - 0.12 and original_sr > balanced_sr + 0.08:
            candidate = original_candidate
        else:
            candidate = balanced_candidate
        candidate = _rotate_jack_anchor_runs(candidate, target, temperature)
        candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
        candidate = _shape_jack_anchor_contrast(candidate, target, temperature)
        candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
        candidate = _shorten_jack_anchor_runs(candidate, config.max_anchor_length, temperature)
        candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
        candidate = _shorten_jack_anchor_runs(candidate, config.max_anchor_length, temperature)
        candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
        candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
        diff = abs(candidate_sr - target)
        if diff < best_diff:
            best_notes = candidate
            best_sr = candidate_sr
            best_diff = diff
        return candidate_sr

    low = lower
    high = upper
    evaluate(low)
    evaluate(high)
    for _ in range(12):
        mid = (low + high) / 2.0
        candidate_sr = evaluate(mid)
        if best_diff <= 0.03:
            break
        if candidate_sr < target:
            low = mid
        else:
            high = mid

    center = (low + high) / 2.0
    for offset in [step * 0.01 for step in range(-8, 9)]:
        evaluate(center + offset)
        if best_diff <= 0.03:
            break

    return best_notes


def _shape_stream_profile(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    tolerance: float = 0.15,
) -> List[NoteObject]:
    if config.chart_type not in ["rice", "ln"] or config.key_style != "stream" or config.target_star is None:
        return notes

    target = config.target_star
    temperature = _pattern_temperature(config)
    max_chord_size = min(3, max(1, config.max_chord_size))
    min_chord_size = _stream_profile_min_chord_size(target)
    max_profile_chord_size = min(max_chord_size, _stream_profile_max_chord_size(target))
    if min_chord_size > max_profile_chord_size:
        min_chord_size = max_profile_chord_size

    best_notes = notes
    best_sr = DifficultyEstimator.estimate_sr(notes, analysis["duration_ms"]) if notes else 0.0
    best_diff = abs(best_sr - target)

    divisors = _stream_divisor_candidates(target, analysis["bpm"])
    preferred_divisor = divisors[0] if divisors else 0
    for divisor in divisors:
        rows = _regular_style_rows(analysis, snap_points, divisor)
        if not rows:
            continue

        lower = float(min_chord_size)
        upper = float(max_profile_chord_size)
        tested_averages: Set[float] = set()

        def evaluate(average_chord_size: float) -> float:
            nonlocal best_notes, best_sr, best_diff
            average_chord_size = round(max(lower, min(upper, average_chord_size)), 3)
            key = average_chord_size
            if key in tested_averages:
                return best_sr
            tested_averages.add(key)

            sizes = _build_stream_chord_sizes(
                row_count=len(rows),
                average_chord_size=average_chord_size,
                min_chord_size=min_chord_size,
                max_chord_size=max_profile_chord_size,
                rows=rows,
                accent_snap_points=accent_snap_points,
                analysis=analysis,
                temperature=temperature,
                target=target,
            )
            candidate = _build_stream_notes(rows, sizes, target, temperature)
            candidate = _apply_safe_lns(candidate, config, analysis, snap_points, "stream")
            candidate = _enforce_stream_fast_cut_integrity(candidate, config, analysis, snap_points, accent_snap_points)
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
            diff = abs(candidate_sr - target)
            if diff < best_diff:
                best_notes = candidate
                best_sr = candidate_sr
                best_diff = diff
            return candidate_sr

        low = lower
        high = upper
        evaluate(low)
        evaluate(high)
        for _ in range(12):
            mid = (low + high) / 2.0
            candidate_sr = evaluate(mid)
            if best_diff <= 0.03:
                break
            if candidate_sr < target:
                low = mid
            else:
                high = mid

        center = (low + high) / 2.0
        for offset in [step * 0.01 for step in range(-8, 9)]:
            evaluate(center + offset)
            if best_diff <= 0.03:
                break

        if divisor == preferred_divisor and best_diff <= tolerance:
            return best_notes

    return best_notes


def _stream_profile_min_chord_size(target: float) -> int:
    return 1


def _stream_profile_max_chord_size(target: float) -> int:
    if target < 3.75:
        return 2
    return 3


def _stream_divisor_candidates(target: float, bpm: float = 0.0) -> List[int]:
    beat_length = 60000.0 / bpm if bpm > 0 else 0.0
    if target < 3.75:
        desired_interval = 112.0
        divisors = [4, 5]
    elif target < 5.25:
        desired_interval = 86.0
        divisors = [4, 5, 6]
    elif target < 6.25:
        desired_interval = 68.0
        divisors = [6, 7, 8, 5]
    else:
        desired_interval = 66.0
        divisors = [6, 7, 5, 8]
    if beat_length <= 0:
        return divisors

    return sorted(
        divisors,
        key=lambda divisor: (
            abs((beat_length / divisor) - desired_interval),
            divisor > 7,
            divisor,
        ),
    )


def _regular_style_rows(analysis: Dict[str, Any], snap_points: List[int], divisor: int) -> List[int]:
    if divisor <= 0 or not snap_points:
        return []

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    step = beat_length / divisor
    snap_set = set(snap_points)
    rows: List[int] = []
    time_ms = float(analysis["offset_ms"])
    while time_ms < analysis["duration_ms"]:
        rounded = int(round(time_ms))
        if not _time_in_regions(rounded, analysis.get("silent_regions", [])):
            if rounded not in snap_set:
                return []
            rows.append(rounded)
        time_ms += step
    return rows


def _build_stream_chord_sizes(
    row_count: int,
    average_chord_size: float,
    min_chord_size: int,
    max_chord_size: int,
    rows: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
    temperature: float,
    target: float,
) -> List[int]:
    if row_count <= 0:
        return []

    sizes = [1 for _ in range(row_count)]
    desired_total = int(round(row_count * average_chord_size))
    desired_total = max(row_count, min(row_count * max_chord_size, desired_total))
    extra_units = desired_total - row_count

    if max_chord_size >= 3 and extra_units >= 2 and target >= 4.75:
        triple_budget = min(
            extra_units // 2,
            int(round(row_count * _stream_triple_ratio(target, temperature))),
        )
        triple_candidates = _stream_spaced_candidates(row_count, rows, accent_snap_points, analysis, target, for_triples=True)
        triples = _pick_non_adjacent(
            triple_candidates,
            triple_budget,
            temperature,
            salt=int(target * 401) + row_count,
            radius=2,
        )
        for index in triples:
            sizes[index] = 3
        extra_units -= len(triples) * 2

    if max_chord_size >= 2 and extra_units > 0:
        double_candidates = _stream_spaced_candidates(row_count, rows, accent_snap_points, analysis, target, for_triples=False)
        double_candidates = [
            index
            for index in double_candidates
            if sizes[index] == 1 and not _is_next_to_stream_triple(sizes, index)
        ]
        selected = _spread_pick(
            double_candidates,
            min(extra_units, len(double_candidates)),
            temperature,
            salt=int(target * 503) + desired_total,
        )
        for index in selected:
            sizes[index] = 2
        extra_units -= len(selected)

    if max_chord_size >= 2 and extra_units > 0:
        fallback = [
            index
            for index, size in enumerate(sizes)
            if size == 1 and not _is_next_to_stream_triple(sizes, index)
        ]
        selected = _spread_pick(fallback, min(extra_units, len(fallback)), temperature, salt=int(target * 607))
        for index in selected:
            sizes[index] = 2
        extra_units -= len(selected)

    if max_chord_size >= 3 and extra_units > 0 and target >= 7.25:
        promote_candidates = [
            index
            for index, size in enumerate(sizes)
            if size == 2 and not _is_next_to_stream_triple(sizes, index)
        ]
        selected = _pick_non_adjacent(
            promote_candidates,
            min(extra_units, len(promote_candidates)),
            temperature,
            salt=int(target * 709),
            radius=2,
        )
        for index in selected:
            sizes[index] = 3

    _break_stream_size_loops(sizes, target, max_chord_size)
    _shape_stream_chord_phrases(sizes, rows, accent_snap_points, analysis, target, max_chord_size, temperature)
    _enforce_stream_fast_cut_size_limits(sizes, rows, accent_snap_points, analysis)

    return sizes


def _break_stream_size_loops(sizes: List[int], target: float, max_chord_size: int) -> None:
    if target < 4.5 or not sizes:
        return

    index = 0
    while index < len(sizes):
        if sizes[index] != 2:
            index += 1
            continue

        start = index
        while index < len(sizes) and sizes[index] == 2:
            index += 1
        run_length = index - start

        if run_length < 5:
            continue

        step = 6 if target < 5.5 else 5
        for pos in range(start + 2, index, step):
            if pos >= index:
                break
            if max_chord_size >= 3 and target >= 5.0 and not _is_next_to_stream_triple(sizes, pos):
                sizes[pos - 1] = 1
                sizes[pos] = 3
                if pos + 1 < index:
                    sizes[pos + 1] = 1
            else:
                sizes[pos] = 1


def _shape_stream_chord_phrases(
    sizes: List[int],
    rows: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
    target: float,
    max_chord_size: int,
    temperature: float,
) -> None:
    if not sizes or max_chord_size <= 1:
        return

    scores = [
        _row_chord_upgrade_score(index, rows, accent_snap_points, analysis)
        for index in range(len(sizes))
    ]
    if not scores:
        return

    strong_threshold = 0.46 if target < 6.0 else 0.40
    medium_threshold = 0.24 if target < 6.0 else 0.20
    phrase_threshold = 0.18 if target < 6.0 else 0.16
    strong_budget = max(1, int(round(len(sizes) * (0.030 + min(0.045, target * 0.006)))))
    phrase_budget = max(1, int(round(len(sizes) * (0.025 + temperature * 0.020))))
    strong_used = 0
    phrase_used = 0

    strong_indexes = sorted(
        [index for index, score in enumerate(scores) if score >= strong_threshold],
        key=lambda index: (-scores[index], index),
    )
    for index in strong_indexes:
        if strong_used >= strong_budget:
            break
        desired = 3 if max_chord_size >= 3 and scores[index] >= strong_threshold + 0.08 else 2
        if sizes[index] < desired:
            sizes[index] = desired
            strong_used += desired - 1

    medium_indexes = sorted(
        [index for index, score in enumerate(scores) if score >= medium_threshold and sizes[index] == 1],
        key=lambda index: (-scores[index], index),
    )
    for index in medium_indexes:
        if strong_used >= strong_budget:
            break
        sizes[index] = 2
        strong_used += 1

    # Strong stream accents should feel like a short handstream phrase, not a lone chord
    # surrounded by single-note speed. Only promote neighbors when the music has enough
    # local onset/stack evidence, so timing correctness stays anchored to the song.
    for index in strong_indexes:
        if phrase_used >= phrase_budget:
            break
        if sizes[index] < 3:
            continue
        neighbors = [
            candidate
            for candidate in [index - 1, index + 1]
            if 0 <= candidate < len(sizes)
            and sizes[candidate] == 1
            and scores[candidate] >= phrase_threshold
        ]
        neighbors.sort(key=lambda candidate: (-scores[candidate], abs(candidate - index), candidate))
        for neighbor in neighbors[:1 if target < 6.25 else 2]:
            if phrase_used >= phrase_budget:
                break
            sizes[neighbor] = 2
            phrase_used += 1

    _soften_stream_size_flicker(sizes, scores, target, max_chord_size)


def _soften_stream_size_flicker(
    sizes: List[int],
    scores: List[float],
    target: float,
    max_chord_size: int,
) -> None:
    if len(sizes) < 3:
        return

    for index in range(1, len(sizes) - 1):
        if sizes[index] < 3:
            continue
        if sizes[index - 1] != 1 or sizes[index + 1] != 1:
            continue
        left_score = scores[index - 1]
        right_score = scores[index + 1]
        if max(left_score, right_score) >= (0.16 if target < 6.0 else 0.14):
            if left_score >= right_score:
                sizes[index - 1] = 2
            else:
                sizes[index + 1] = 2
        elif scores[index] < (0.36 if target < 6.0 else 0.32):
            sizes[index] = 2

    run_start = 0
    while run_start < len(sizes):
        run_end = run_start + 1
        while run_end < len(sizes) and sizes[run_end] == sizes[run_start]:
            run_end += 1
        if sizes[run_start] == 1 and run_end - run_start >= 9:
            best = max(range(run_start, run_end), key=lambda idx: (scores[idx], -idx))
            if scores[best] >= (0.20 if target < 6.0 else 0.17):
                sizes[best] = 2
        run_start = run_end


def _enforce_stream_fast_cut_size_limits(
    sizes: List[int],
    rows: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
) -> None:
    if len(sizes) < 2 or len(rows) != len(sizes):
        return

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    fast_gap_ms = max(86, min(112, int(round(beat_length / 4.0))))
    scores = [
        _row_chord_upgrade_score(index, rows, accent_snap_points, analysis)
        for index in range(len(sizes))
    ]

    changed = True
    while changed:
        changed = False
        for index in range(1, len(sizes)):
            if rows[index] - rows[index - 1] > fast_gap_ms:
                continue
            while sizes[index] + sizes[index - 1] > 4:
                if sizes[index] <= 1 and sizes[index - 1] <= 1:
                    break
                if sizes[index] <= 1:
                    demote_index = index - 1
                elif sizes[index - 1] <= 1:
                    demote_index = index
                elif scores[index] + 0.04 < scores[index - 1]:
                    demote_index = index
                else:
                    demote_index = index - 1
                if sizes[demote_index] <= 1:
                    break
                sizes[demote_index] -= 1
                changed = True


def _stream_triple_ratio(target: float, temperature: float) -> float:
    if target < 4.75:
        base = 0.0
    elif target < 5.5:
        base = 0.11 + (target - 4.75) * 0.10
    elif target < 6.5:
        base = 0.20 + (target - 5.5) * 0.075
    else:
        base = 0.27 + min(0.08, (target - 6.5) * 0.05)
    return max(0.0, min(0.38, base * (0.88 + temperature * 0.26)))


def _stream_spaced_candidates(
    row_count: int,
    rows: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
    target: float,
    for_triples: bool,
) -> List[int]:
    if row_count <= 0:
        return []

    if for_triples:
        music_ranked = _music_ranked_row_indexes(
            [index for index in range(1, row_count - 1)],
            rows,
            accent_snap_points,
            analysis,
            min_score=0.22 if target < 6.0 else 0.18,
        )
        groups = [
            [index for index in music_ranked if index % 2 == 0],
            music_ranked,
            [index for index in range(1, row_count - 1) if rows[index] in accent_snap_points and index % 2 == 0],
            [index for index in range(1, row_count - 1) if rows[index] in accent_snap_points],
            [index for index in range(1, row_count - 1) if index % 4 == 0],
            [index for index in range(1, row_count - 1) if index % 2 == 0],
        ]
    else:
        music_ranked = _music_ranked_row_indexes(
            [index for index in range(row_count)],
            rows,
            accent_snap_points,
            analysis,
            min_score=0.14,
        )
        groups = [
            music_ranked,
            [index for index in range(row_count) if rows[index] in accent_snap_points],
            [index for index in range(row_count) if index % 2 == 0],
            [index for index in range(row_count) if index % 2 == 1],
        ]

    ordered: List[int] = []
    seen: Set[int] = set()
    for group in groups:
        for index in group:
            if index in seen:
                continue
            ordered.append(index)
            seen.add(index)
    return ordered


def _music_ranked_row_indexes(
    indexes: List[int],
    rows: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
    min_score: float,
) -> List[int]:
    ranked = [
        index
        for index in indexes
        if _row_chord_upgrade_score(index, rows, accent_snap_points, analysis) >= min_score
    ]
    return sorted(
        ranked,
        key=lambda index: (
            -_row_chord_upgrade_score(index, rows, accent_snap_points, analysis),
            index,
        ),
    )


def _row_chord_upgrade_score(
    index: int,
    rows: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
) -> float:
    if index < 0 or index >= len(rows):
        return 0.0
    return _chord_music_strength_at(analysis, rows, accent_snap_points, rows[index])


def _pick_non_adjacent(values: List[int], count: int, temperature: float, salt: int, radius: int = 1) -> List[int]:
    if count <= 0 or not values:
        return []

    picked: List[int] = []
    blocked: Set[int] = set()
    for index in _spread_pick(values, min(count * 3, len(values)), temperature, salt=salt):
        if index in blocked:
            continue
        picked.append(index)
        blocked.update(range(index - radius, index + radius + 1))
        if len(picked) >= count:
            break

    if len(picked) >= count:
        return picked

    for index in values:
        if index in blocked:
            continue
        picked.append(index)
        blocked.update(range(index - radius, index + radius + 1))
        if len(picked) >= count:
            break

    return picked


def _is_next_to_stream_triple(sizes: List[int], index: int) -> bool:
    return (
        (index > 0 and sizes[index - 1] >= 3)
        or sizes[index] >= 3
        or (index + 1 < len(sizes) and sizes[index + 1] >= 3)
    )


def _rebalance_stream_handstream_sizes(
    sizes: List[int],
    rows: List[int],
    accent_snap_points: Set[int],
    target: float,
    temperature: float,
    max_chord_size: int,
) -> None:
    if target < 6.35 or max_chord_size < 3 or not sizes:
        return

    row_count = len(sizes)
    desired_ratio = 0.025 + max(0.0, min(1.0, (target - 6.35) / 0.90)) * 0.095
    desired_ratio *= 0.85 + max(0.0, min(1.0, temperature)) * 0.30
    desired_triples = int(round(row_count * desired_ratio))
    current_triples = sum(1 for size in sizes if size >= 3)
    needed = max(0, desired_triples - current_triples)
    if needed <= 0:
        return

    promote_candidates = [
        index
        for index, size in enumerate(sizes)
        if size == 2
        and (rows[index] in accent_snap_points or index % 2 == 0)
        and (index == 0 or sizes[index - 1] < 3)
        and (index + 1 >= row_count or sizes[index + 1] < 3)
    ]
    if not promote_candidates:
        return

    accent_promotes = [index for index in promote_candidates if rows[index] in accent_snap_points]
    offbeat_promotes = [index for index in promote_candidates if rows[index] not in accent_snap_points]
    ordered_promotes = accent_promotes + offbeat_promotes

    demote_candidates = [
        index
        for index, size in enumerate(sizes)
        if size == 2 and rows[index] not in accent_snap_points and index % 2 == 1
    ]
    if not demote_candidates:
        demote_candidates = [
            index
            for index, size in enumerate(sizes)
            if size == 2 and rows[index] not in accent_snap_points
        ]
    if not demote_candidates:
        demote_candidates = [index for index, size in enumerate(sizes) if size == 2 and index % 2 == 1]
    if not demote_candidates:
        demote_candidates = [index for index, size in enumerate(sizes) if size == 2]

    pair_count = min(needed, len(ordered_promotes), len(demote_candidates))
    if pair_count <= 0:
        return

    promote_pick = _spread_pick(
        ordered_promotes,
        pair_count,
        temperature,
        salt=int(target * 311) + row_count,
    )
    promote_set = set(promote_pick)
    demote_pool = [index for index in demote_candidates if index not in promote_set]
    demote_pick = _spread_pick(
        demote_pool,
        len(promote_pick),
        temperature,
        salt=int(target * 347) + len(promote_pick),
    )
    pair_count = min(len(promote_pick), len(demote_pick))
    for index in promote_pick[:pair_count]:
        sizes[index] = 3
    for index in demote_pick[:pair_count]:
        sizes[index] = 1


def _select_stream_upgrade_rows(
    eligible: List[int],
    rows: List[int],
    accent_snap_points: Set[int],
    promote_count: int,
    temperature: float,
    target: float,
    next_size: int,
) -> List[int]:
    if promote_count <= 0:
        return []

    if temperature >= 0.70:
        return _spread_pick(
            eligible,
            promote_count,
            temperature,
            salt=int(target * 151) + next_size * 211,
        )

    groups = [
        [index for index in eligible if rows[index] in accent_snap_points],
        [index for index in eligible if rows[index] not in accent_snap_points and index % 4 == 0],
        [index for index in eligible if rows[index] not in accent_snap_points and index % 2 == 0],
        [index for index in eligible if index % 2 == 1],
    ]
    selected: List[int] = []
    selected_set: Set[int] = set()
    remaining = promote_count

    for group in groups:
        group = [index for index in group if index not in selected_set]
        if not group or remaining <= 0:
            continue
        group.sort()
        picks = group if remaining >= len(group) else _spread_pick(
            group,
            remaining,
            temperature,
            salt=int(target * 97) + next_size * 53 + len(selected),
        )
        for index in picks:
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            remaining -= 1
            if remaining <= 0:
                break

    if remaining > 0:
        fallback = [index for index in eligible if index not in selected_set]
        for index in _spread_pick(fallback, remaining, temperature, salt=next_size * 271):
            selected.append(index)

    return selected


def _build_stream_notes(
    rows: List[int],
    sizes: List[int],
    target: float,
    temperature: float,
) -> List[NoteObject]:
    notes: List[NoteObject] = []
    previous_lanes: List[int] = []
    recent_lanes: List[List[int]] = []
    stream_index = 0

    for row_index, (time_ms, size) in enumerate(zip(rows, sizes)):
        lanes = _next_stream_lanes(size, previous_lanes, recent_lanes, stream_index, row_index, target, temperature)
        stream_index += 1
        previous_lanes = lanes
        recent_lanes.append(lanes)
        if len(recent_lanes) > 6:
            recent_lanes.pop(0)
        for lane in lanes:
            notes.append(NoteObject(time_ms=time_ms, lane=lane))

    return notes


def _next_stream_lanes(
    size: int,
    previous_lanes: List[int],
    recent_lanes: List[List[int]],
    stream_index: int,
    row_index: int,
    target: float,
    temperature: float,
) -> List[int]:
    patterns = _stream_patterns(size)
    if not patterns:
        return []

    previous_set = set(previous_lanes)
    complement = {0, 1, 2, 3} - previous_set
    stack_penalties = {
        tuple(lanes): _stream_stack_penalty(lanes, previous_set, complement, recent_lanes, target)
        for lanes in patterns
    }
    ranked = sorted(
        patterns,
        key=lambda lanes: (
            stack_penalties[tuple(lanes)],
            _stream_hand_bias(lanes, row_index, target),
            _deterministic_index(97, stream_index, row_index, size, int(target * 100), int(temperature * 100)),
            patterns.index(lanes),
        ),
    )
    best_stack_penalty = stack_penalties[tuple(ranked[0])]
    safe_ranked = [
        lanes
        for lanes in ranked
        if stack_penalties[tuple(lanes)] <= best_stack_penalty + 0.01
    ]
    if safe_ranked:
        ranked = safe_ranked

    pool_size = 1 + int(round(max(0.0, min(1.0, temperature)) * (len(ranked) - 1)))
    pool = ranked[: max(1, pool_size)]
    choice = _deterministic_index(
        len(pool),
        stream_index,
        row_index,
        size,
        int(target * 100),
        int(temperature * 1000),
        sum(previous_lanes) if previous_lanes else 0,
    )
    return list(pool[choice])


def _stream_stack_penalty(
    lanes: List[int],
    previous_set: Set[int],
    complement: Set[int],
    recent_lanes: List[List[int]],
    target: float,
) -> float:
    lane_set = set(lanes)
    previous_overlap = len(lane_set & previous_set)
    penalty = previous_overlap * 1000.0

    if previous_overlap == 0:
        penalty -= 8.0
    if lane_set == complement:
        penalty -= 1.0

    if len(recent_lanes) >= 2:
        two_back = set(recent_lanes[-2])
        two_back_overlap = len(lane_set & two_back)
        penalty += two_back_overlap * 1.0
        if lane_set == two_back:
            penalty += 8.0 if target < 6.5 else 6.0

    if len(recent_lanes) >= 4:
        repeated = sum(1 for old in recent_lanes[-4:] if set(old) == lane_set)
        penalty += repeated * (1.5 if target < 6.5 else 1.0)

    return penalty


def _stream_hand_bias(lanes: List[int], row_index: int, target: float) -> int:
    lane_set = set(lanes)
    left = len(lane_set & {0, 1})
    right = len(lane_set & {2, 3})
    hand_balance_penalty = abs(left - right)
    edge_penalty = 0 if lane_set in [{0, 3}, {0, 1}, {2, 3}] else 1
    if target >= 5.0 and len(lanes) >= 2:
        edge_penalty = 0 if lane_set in [{0, 1}, {2, 3}, {0, 3}, {0, 2}, {1, 3}] else 1
    return hand_balance_penalty + edge_penalty + ((row_index + sum(lanes)) % 2)


def _stream_patterns(size: int) -> List[List[int]]:
    if size <= 1:
        return [[0], [2], [1], [3]]
    if size == 2:
        return [[0, 2], [1, 3], [0, 3], [1, 2], [0, 1], [2, 3]]
    return [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]


def _repair_stream_cut_collisions(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
) -> List[NoteObject]:
    if config.chart_type not in ["rice", "ln"] or config.key_style != "stream" or not notes:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)

    row_times = sorted(rows)
    if len(row_times) < 2:
        return notes

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    fast_cut_gap_ms = max(140, min(170, int(round(beat_length / 2.55))))
    target = config.target_star or 4.0
    active_until = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    last_lane_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    last_ln_tail_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    repaired_rows: Dict[int, Set[int]] = {}
    result: List[NoteObject] = []

    for index, time_ms in enumerate(row_times):
        for lane in range(4):
            if active_until[lane] <= time_ms:
                active_until[lane] = -999999

        row_notes = sorted(rows[time_ms], key=lambda note: (note.lane, note.end_time_ms or -1))
        row_size = len(row_notes)
        current_lanes = {note.lane for note in row_notes}
        chosen_lanes = sorted(current_lanes)

        prev_time = row_times[index - 1] if index > 0 else None
        next_time = row_times[index + 1] if index + 1 < len(row_times) else None
        prev_lanes = repaired_rows.get(prev_time, set()) if prev_time is not None else set()
        next_lanes = {note.lane for note in rows[next_time]} if next_time is not None else set()
        prev_fast = prev_time is not None and time_ms - prev_time <= fast_cut_gap_ms
        next_fast = next_time is not None and next_time - time_ms <= fast_cut_gap_ms
        prev_overlap = len(current_lanes & prev_lanes) if prev_fast else 0
        prev_unavoidable = max(0, row_size + len(prev_lanes) - 4) if prev_fast else 0
        prev_avoidable = max(0, prev_overlap - prev_unavoidable)
        current_blocked = any(
            active_until[lane] > time_ms
            or time_ms - last_lane_time[lane] < 40
            or time_ms - last_ln_tail_time[lane] <= 30
            for lane in current_lanes
        )

        needs_repair = False
        if current_blocked and row_size <= 3:
            needs_repair = True
        if prev_avoidable > 0 and row_size <= 3 and len(prev_lanes) <= 3:
            needs_repair = True
        if row_size <= 2 and prev_fast and len(prev_lanes) <= 2 and (current_lanes & prev_lanes):
            needs_repair = True
        if (
            prev_fast
            and next_fast
            and len(prev_lanes) <= 2
            and len(next_lanes) <= 2
            and row_size >= 2
            and ((prev_lanes & next_lanes) & current_lanes)
        ):
            needs_repair = True

        patterns = _stream_patterns(row_size) if row_size <= 3 else []
        pattern_options: List[Tuple[List[int], int, int]] = [
            (pattern, 0, option_index)
            for option_index, pattern in enumerate(patterns)
        ]
        if needs_repair and row_size > 1:
            option_index = len(pattern_options)
            for smaller_size in range(row_size - 1, 0, -1):
                for pattern in _stream_patterns(smaller_size):
                    pattern_options.append((pattern, row_size - smaller_size, option_index))
                    option_index += 1
        if needs_repair and config.chart_type == "ln":
            pattern_options.append(([], row_size, len(pattern_options)))
        if needs_repair and patterns:
            original_set = set(chosen_lanes)
            bridge_lanes = (prev_lanes & next_lanes) if next_fast else set()
            classic_pair_cut = (
                row_size == 3
                and not any(note.is_ln for note in row_notes)
                and prev_fast
                and next_fast
                and len(prev_lanes) == 2
                and prev_lanes == next_lanes
            )
            complement_pair = sorted({0, 1, 2, 3} - prev_lanes)
            if (
                classic_pair_cut
                and len(complement_pair) == 2
                and all(active_until[lane] <= time_ms for lane in complement_pair)
                and all(time_ms - last_lane_time[lane] >= 40 for lane in complement_pair)
                and all(time_ms - last_ln_tail_time[lane] > 30 for lane in complement_pair)
            ):
                chosen_lanes = complement_pair
            else:

                def score(option: Tuple[List[int], int, int]) -> Tuple[float, int, int]:
                    pattern, reduction, option_index = option
                    lane_set = set(pattern)
                    prev_overlap = len(lane_set & prev_lanes) if prev_fast and len(prev_lanes) <= 3 else 0
                    next_overlap = len(lane_set & next_lanes) if next_fast and len(next_lanes) <= 3 else 0
                    prev_unavoidable = max(0, len(lane_set) + len(prev_lanes) - 4) if prev_fast else 0
                    next_unavoidable = max(0, len(lane_set) + len(next_lanes) - 4) if next_fast else 0
                    prev_avoidable = max(0, prev_overlap - prev_unavoidable)
                    next_avoidable = max(0, next_overlap - next_unavoidable)
                    bridge_overlap = len(lane_set & bridge_lanes)
                    held_conflict = sum(1 for lane in lane_set if active_until[lane] > time_ms)
                    too_close_last = sum(1 for lane in lane_set if time_ms - last_lane_time[lane] < 40)
                    tail_close = sum(1 for lane in lane_set if time_ms - last_ln_tail_time[lane] <= 30)
                    exact_prev = prev_fast and lane_set == prev_lanes
                    change_count = len(lane_set ^ original_set)
                    return (
                        held_conflict * 100000.0
                        + too_close_last * 100000.0
                        + tail_close * 100000.0
                        + prev_avoidable * 24000.0
                        + bridge_overlap * 12000.0
                        + next_avoidable * 220.0
                        + prev_overlap * 18.0
                        + next_overlap * 8.0
                        + (4500.0 if exact_prev else 0.0)
                        + change_count * 2.0
                        + reduction * 7000.0
                        + _stream_hand_bias(pattern, index, target),
                        _deterministic_index(211, time_ms, row_size, sum(pattern), int(target * 100)),
                        option_index,
                    )

                chosen_lanes = sorted(min(pattern_options, key=score)[0])

        for note, lane in zip(row_notes, chosen_lanes):
            remapped = replace(note, lane=lane)
            result.append(remapped)
            last_lane_time[lane] = time_ms
            if remapped.end_time_ms is not None:
                active_until[lane] = max(active_until[lane], remapped.end_time_ms)
                last_ln_tail_time[lane] = max(last_ln_tail_time[lane], remapped.end_time_ms)
        repaired_rows[time_ms] = set(chosen_lanes)

    return result


def _collapse_stream_micro_rows(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    if config.chart_type not in ["rice", "ln"] or config.key_style != "stream" or not notes:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    row_times = sorted(rows)
    if len(row_times) < 2:
        return notes

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    min_gap = max(32, min(44, int(round(beat_length / 12.5))))
    max_chord_size = min(3, clamp_max_chord_size(config, "stream"))
    collapsed: List[NoteObject] = []
    changed = False

    def row_strength(time_ms: int) -> Tuple[float, int, int, int]:
        entry = _music_entry(analysis, snap_points, accent_snap_points, time_ms)
        row_notes = rows.get(time_ms, [])
        return (
            entry["score"] + entry["accent"] * 0.55 + entry["kick"] * 0.35 + entry["protected"] * 0.35,
            len(row_notes),
            int(any(note.is_ln for note in row_notes)),
            -time_ms,
        )

    def flush(cluster: List[int]) -> None:
        nonlocal changed
        if len(cluster) == 1:
            collapsed.extend(rows[cluster[0]])
            return

        changed = True
        anchor_time = max(cluster, key=row_strength)
        candidates: List[NoteObject] = []
        for time_ms in sorted(cluster, key=lambda value: (value != anchor_time, -row_strength(value)[0], value)):
            candidates.extend(sorted(rows[time_ms], key=lambda note: (not note.is_ln, note.lane, note.end_time_ms or -1)))

        used_lanes: Set[int] = set()
        merged: List[NoteObject] = []
        for note in candidates:
            if note.lane in used_lanes or len(merged) >= max_chord_size:
                continue
            delta = anchor_time - note.time_ms
            end_time_ms = note.end_time_ms + delta if note.end_time_ms is not None else None
            if end_time_ms is not None and end_time_ms <= anchor_time:
                continue
            merged.append(NoteObject(time_ms=anchor_time, lane=note.lane, end_time_ms=end_time_ms))
            used_lanes.add(note.lane)

        collapsed.extend(merged or rows[anchor_time])

    cluster = [row_times[0]]
    for time_ms in row_times[1:]:
        if time_ms - cluster[-1] < min_gap:
            cluster.append(time_ms)
        else:
            flush(cluster)
            cluster = [time_ms]
    flush(cluster)

    if not changed:
        return notes
    return sorted(collapsed, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _stabilize_stream_cut_repairs(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    passes: int = 4,
) -> List[NoteObject]:
    current = notes
    for _ in range(max(1, passes)):
        repaired = _repair_stream_cut_collisions(current, config, analysis)
        if _note_signature(repaired) == _note_signature(current):
            return repaired
        current = repaired
    return current


def _note_signature(notes: List[NoteObject]) -> Tuple[Tuple[int, int, Optional[int]], ...]:
    return tuple((note.time_ms, note.lane, note.end_time_ms) for note in notes)


def _reinforce_stream_music_chords(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    if config.chart_type not in ["rice", "ln"] or config.key_style != "stream" or not notes:
        return notes

    max_chord_size = min(3, clamp_max_chord_size(config, "stream"))
    if max_chord_size <= 1:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    row_times = sorted(rows)
    if len(row_times) < 3:
        return notes

    target = config.target_star or 4.0
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    fast_gap = max(44, min(82, int(round(beat_length / 5.2))))
    row_lanes: Dict[int, Set[int]] = {time_ms: {note.lane for note in row_notes} for time_ms, row_notes in rows.items()}
    strengths = {
        time_ms: _chord_music_strength_at(analysis, snap_points, accent_snap_points, time_ms)
        for time_ms in row_times
    }
    strong_threshold = 0.46 if target < 6.0 else 0.42
    medium_threshold = 0.24 if target < 6.0 else 0.22
    phrase_threshold = 0.22 if target < 6.0 else 0.18
    max_additions = max(1, int(round(len(row_times) * (0.110 + min(0.055, target * 0.006)))))
    additions: List[NoteObject] = []
    reinforced: Set[int] = set()

    def desired_size(time_ms: int) -> int:
        strength = strengths.get(time_ms, 0.0)
        if strength >= strong_threshold + 0.10 and max_chord_size >= 3:
            return 3
        if strength >= medium_threshold:
            return 2
        return 1

    def pick_pattern(time_ms: int, desired: int) -> Optional[List[int]]:
        current = row_lanes.get(time_ms, set())
        if len(current) >= desired:
            return sorted(current)
        index = bisect.bisect_left(row_times, time_ms)
        prev_time = row_times[index - 1] if index > 0 else None
        next_time = row_times[index + 1] if index + 1 < len(row_times) else None
        prev_lanes = row_lanes.get(prev_time, set()) if prev_time is not None and time_ms - prev_time <= fast_gap else set()
        next_lanes = row_lanes.get(next_time, set()) if next_time is not None and next_time - time_ms <= fast_gap else set()
        candidates = [pattern for pattern in _stream_patterns(desired) if current.issubset(set(pattern))]
        if not candidates:
            return None

        def score(pattern: List[int]) -> Tuple[float, int, int]:
            lane_set = set(pattern)
            prev_overlap = len(lane_set & prev_lanes)
            next_overlap = len(lane_set & next_lanes)
            unavoidable_prev = max(0, len(lane_set) + len(prev_lanes) - 4)
            unavoidable_next = max(0, len(lane_set) + len(next_lanes) - 4)
            avoidable = max(0, prev_overlap - unavoidable_prev) * 2 + max(0, next_overlap - unavoidable_next)
            return (
                avoidable * 100.0
                + prev_overlap * 4.0
                + next_overlap * 2.0
                + _stream_hand_bias(pattern, index, target),
                _deterministic_index(353, time_ms, desired, sum(pattern), int(target * 100)),
                sum(pattern),
            )

        return list(min(candidates, key=score))

    def add_to_row(time_ms: int, desired: int) -> int:
        pattern = pick_pattern(time_ms, desired)
        if not pattern:
            return 0
        added = 0
        for lane in pattern:
            if lane in row_lanes[time_ms] or len(row_lanes[time_ms]) >= desired:
                continue
            row_lanes[time_ms].add(lane)
            additions.append(NoteObject(time_ms=time_ms, lane=lane))
            added += 1
        if added:
            reinforced.add(time_ms)
        return added

    added_count = 0
    for time_ms in sorted(row_times, key=lambda value: (-strengths.get(value, 0.0), value)):
        if added_count >= max_additions:
            break
        desired = desired_size(time_ms)
        if desired <= len(row_lanes.get(time_ms, set())):
            continue
        added_count += add_to_row(time_ms, desired)

    for time_ms in sorted(reinforced, key=lambda value: (-strengths.get(value, 0.0), value)):
        if added_count >= max_additions:
            break
        index = bisect.bisect_left(row_times, time_ms)
        neighbors = [
            candidate
            for candidate in [
                row_times[index - 1] if index > 0 else None,
                row_times[index + 1] if index + 1 < len(row_times) else None,
            ]
            if candidate is not None
            and abs(candidate - time_ms) <= fast_gap
            and strengths.get(candidate, 0.0) >= phrase_threshold
            and len(row_lanes.get(candidate, set())) == 1
        ]
        neighbors.sort(key=lambda value: (-strengths.get(value, 0.0), value))
        for neighbor in neighbors[:1 if target < 6.25 else 2]:
            if added_count >= max_additions:
                break
            added_count += add_to_row(neighbor, 2)

    if not additions:
        return notes
    return sorted(notes + additions, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _enforce_stream_fast_cut_integrity(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    if config.chart_type not in ["rice", "ln"] or config.key_style != "stream" or not notes:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    row_times = sorted(rows)
    if len(row_times) < 2:
        return notes

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    fast_gap_ms = max(86, min(112, int(round(beat_length / 4.0))))
    changed = False
    repaired_rows: Dict[int, List[NoteObject]] = {}
    previous_time: Optional[int] = None
    previous_lanes: Set[int] = set()

    for time_ms in row_times:
        row_notes = sorted(rows[time_ms], key=lambda note: (note.lane, note.end_time_ms or -1))
        current_lanes = {note.lane for note in row_notes}

        if previous_time is not None and time_ms - previous_time <= fast_gap_ms:
            current_strength = _chord_music_strength_at(analysis, snap_points, accent_snap_points, time_ms)
            previous_strength = _chord_music_strength_at(analysis, snap_points, accent_snap_points, previous_time)
            if current_strength > previous_strength + 0.04 and current_lanes & previous_lanes:
                previous_row = repaired_rows.get(previous_time, [])
                for lane in sorted(current_lanes & previous_lanes):
                    if len(previous_row) <= 1:
                        break
                    removable = [
                        note
                        for note in previous_row
                        if note.lane == lane and not note.is_ln
                    ]
                    if not removable:
                        continue
                    previous_row.remove(removable[0])
                    previous_lanes.discard(lane)
                    changed = True
                repaired_rows[previous_time] = previous_row

            repaired_notes: List[NoteObject] = []
            for note in row_notes:
                if note.lane not in previous_lanes:
                    repaired_notes.append(note)
                    continue

                available = [
                    lane
                    for lane in range(4)
                    if lane not in previous_lanes and lane not in current_lanes
                ]
                if available:
                    entry = _music_entry(analysis, snap_points, accent_snap_points, time_ms)
                    preferred_lane = _music_receiver_lane_for_style(
                        time_ms,
                        {previous_time: previous_lanes, time_ms: current_lanes},
                        [previous_time, time_ms],
                        entry,
                        clamp_max_chord_size(config, "stream"),
                        "stream",
                        config.target_star or 4.0,
                        beat_length,
                    )
                    lane = preferred_lane if preferred_lane in available else available[0]
                    current_lanes.discard(note.lane)
                    current_lanes.add(lane)
                    repaired_notes.append(replace(note, lane=lane))
                    changed = True
                elif len(current_lanes) > 1 and not note.is_ln:
                    current_lanes.discard(note.lane)
                    changed = True
                else:
                    repaired_notes.append(note)

            # If a 3-note row follows a 2-note row (or similar), the previous
            # rewrite may still be impossible on four columns. Trim the weakest
            # non-LN tail until the fast cut can be played without same-lane reuse.
            while previous_lanes & {note.lane for note in repaired_notes}:
                removable = [note for note in repaired_notes if not note.is_ln]
                if len(repaired_notes) <= 1 or not removable:
                    break
                weakest = sorted(
                    removable,
                    key=lambda note: (
                        _chord_music_strength_at(analysis, snap_points, accent_snap_points, time_ms),
                        -abs(note.lane - 1.5),
                        note.lane,
                    ),
                )[0]
                repaired_notes.remove(weakest)
                changed = True

            row_notes = repaired_notes

        repaired_rows[time_ms] = row_notes
        previous_time = time_ms
        previous_lanes = {note.lane for note in row_notes}

    if not changed:
        return notes

    repaired = [
        note
        for time_ms in row_times
        for note in repaired_rows.get(time_ms, [])
    ]
    return sorted(repaired, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _shape_speed_profile(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    tolerance: float = 0.15,
) -> List[NoteObject]:
    if config.chart_type not in ["rice", "ln"] or config.key_style != "speed" or config.target_star is None:
        return notes

    target = config.target_star
    temperature = _pattern_temperature(config)
    max_chord_size = min(3, max(1, config.max_chord_size))
    max_profile_chord_size = min(max_chord_size, _speed_profile_max_chord_size(target))

    best_notes = notes
    best_sr = DifficultyEstimator.estimate_sr(notes, analysis["duration_ms"]) if notes else 0.0
    best_diff = abs(best_sr - target)
    best_within_tolerance = best_diff <= tolerance
    best_style_score = float("inf")

    for divisor in _speed_divisor_candidates(target, analysis["bpm"]):
        rows = _regular_style_rows(analysis, snap_points, divisor)
        rows = _shape_speed_rows(rows, target, temperature, analysis, snap_points, accent_snap_points)
        if not rows:
            continue

        lower = 1.0
        upper = _speed_profile_average_upper(target, max_profile_chord_size, config.chart_type)
        upper = max(lower, min(float(max_profile_chord_size), upper))
        tested_averages: Set[float] = set()

        def evaluate(average_chord_size: float) -> float:
            nonlocal best_notes, best_sr, best_diff, best_within_tolerance, best_style_score
            average_chord_size = round(max(lower, min(upper, average_chord_size)), 3)
            if average_chord_size in tested_averages:
                return best_sr
            tested_averages.add(average_chord_size)

            sizes = _build_speed_chord_sizes(
                row_count=len(rows),
                average_chord_size=average_chord_size,
                max_chord_size=max_profile_chord_size,
                rows=rows,
                accent_snap_points=accent_snap_points,
                target=target,
                temperature=temperature,
            )
            candidate = _build_speed_notes(rows, sizes, target, temperature)
            candidate = _apply_safe_lns(candidate, config, analysis, snap_points, "speed")
            candidate = _apply_music_influence(candidate, config, analysis, snap_points, accent_snap_points, "speed")
            candidate = _fortify_speed_accent_chords(candidate, config, analysis, snap_points, accent_snap_points)
            candidate = _cap_speed_chord_density(candidate, config, analysis, snap_points, accent_snap_points)
            candidate = _repair_speed_single_lane_flow(candidate, target)
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate = _repair_speed_single_lane_flow(candidate, target)
            candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
            diff = abs(candidate_sr - target)
            style_score = _speed_style_score(divisor, average_chord_size, target, analysis["bpm"])
            if diff <= tolerance:
                if (
                    not best_within_tolerance
                    or style_score < best_style_score
                    or (abs(style_score - best_style_score) <= 0.001 and diff < best_diff)
                ):
                    best_notes = candidate
                    best_sr = candidate_sr
                    best_diff = diff
                    best_within_tolerance = True
                    best_style_score = style_score
            elif not best_within_tolerance and diff < best_diff:
                best_notes = candidate
                best_sr = candidate_sr
                best_diff = diff
            return candidate_sr

        low = lower
        high = upper
        evaluate(low)
        evaluate(high)
        for _ in range(12):
            mid = (low + high) / 2.0
            candidate_sr = evaluate(mid)
            if best_diff <= 0.03:
                break
            if candidate_sr < target:
                low = mid
            else:
                high = mid

        center = (low + high) / 2.0
        for offset in [step * 0.01 for step in range(-10, 11)]:
            evaluate(center + offset)
            if best_diff <= 0.03:
                break

        if best_diff <= 0.03:
            break

    return best_notes


def _speed_profile_max_chord_size(target: float) -> int:
    if target < 5.75:
        return 2
    return 3


def _speed_profile_average_upper(target: float, max_chord_size: int, chart_type: str) -> float:
    if max_chord_size <= 1:
        return 1.0
    if target < 3.5:
        upper = 1.16
    elif target < 4.5:
        upper = 1.22
    elif target < 5.5:
        upper = 1.42
    elif target < 6.5:
        upper = 1.62
    else:
        upper = 1.82

    return max(1.0, min(float(max_chord_size), upper))


def _speed_style_score(divisor: int, average_chord_size: float, target: float, bpm: float) -> float:
    beat_length = 60000.0 / bpm if bpm > 0 else 0.0
    interval = beat_length / divisor if divisor > 0 and beat_length > 0 else 999.0
    if target < 3.25:
        desired_interval = 86.0
    elif target < 3.75:
        desired_interval = 72.0
    elif target < 4.75:
        desired_interval = 54.0
    elif target < 5.75:
        desired_interval = 52.0
    else:
        desired_interval = 43.0

    desired_average = _speed_desired_average_chord_size(target)
    if target < 5.75:
        chord_weight = 10.0
        interval_weight = 0.45
    else:
        chord_weight = 8.0
        interval_weight = 0.05

    chord_penalty = abs(average_chord_size - desired_average) * chord_weight
    interval_penalty = abs(interval - desired_interval) * interval_weight
    return chord_penalty + interval_penalty


def _speed_desired_average_chord_size(target: float) -> float:
    if target < 3.5:
        return 1.04
    if target < 4.5:
        return 1.07
    if target < 5.5:
        return 1.14
    if target < 6.0:
        return 1.18
    if target < 6.5:
        return 1.20
    return 1.32


def _speed_divisor_candidates(target: float, bpm: float = 0.0) -> List[int]:
    beat_length = 60000.0 / bpm if bpm > 0 else 0.0
    if target < 3.25:
        desired_interval = 86.0
    elif target < 3.75:
        desired_interval = 62.0
    elif target < 4.50:
        desired_interval = 54.0
    elif target < 5.75:
        desired_interval = 52.0
    elif target < 6.75:
        desired_interval = 43.0
    else:
        desired_interval = 43.0

    divisors = [4, 5, 6, 7, 8, 10, 12]
    if beat_length <= 0:
        return divisors

    return sorted(
        divisors,
        key=lambda divisor: (
            abs((beat_length / divisor) - desired_interval),
            divisor < 6,
            divisor,
        ),
    )


def _shape_speed_rows(
    rows: List[int],
    target: float,
    temperature: float,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[int]:
    if len(rows) < 32:
        return rows

    # Speed reference charts are not an unbroken wall of the smallest snap.
    # They keep fast cells, but puncture them with 2x/4x gaps and preserve
    # accents so the result reads as "乱" instead of a metronomic cut stream.
    if target < 4.5:
        masks = [
            [1, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 1, 0],
        ]
    elif target < 5.75:
        masks = [
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        ]
    else:
        masks = [
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        ]

    context = _music_context(analysis, snap_points, accent_snap_points)
    shaped: List[int] = []
    phrase_index = 0
    row_index = 0
    while row_index < len(rows):
        mask = masks[_deterministic_index(len(masks), phrase_index, int(target * 100), int(temperature * 1000))]
        phrase_len = len(mask)
        phrase = rows[row_index : row_index + phrase_len]
        for local_index, time_ms in enumerate(phrase):
            entry = context.get(time_ms, {})
            forced = (
                time_ms in accent_snap_points
                or entry.get("accent", 0.0) >= 0.62
                or entry.get("kick", 0.0) >= 0.56
                or entry.get("onset", 0.0) >= 0.58
            )
            if forced or mask[local_index % len(mask)]:
                shaped.append(time_ms)
        row_index += phrase_len
        phrase_index += 1

    # Avoid deleting too much on sparse music; speed still needs long fast lanes.
    minimum_ratio = 0.58 if target >= 5.75 else 0.62
    if len(shaped) < int(len(rows) * minimum_ratio):
        shaped_set = set(shaped)
        for index, time_ms in enumerate(rows):
            if len(shaped) >= int(len(rows) * minimum_ratio):
                break
            if time_ms not in shaped_set and index % 4 in [1, 3]:
                shaped.append(time_ms)
                shaped_set.add(time_ms)
        shaped.sort()
    return shaped


def _build_speed_chord_sizes(
    row_count: int,
    average_chord_size: float,
    max_chord_size: int,
    rows: List[int],
    accent_snap_points: Set[int],
    target: float,
    temperature: float,
) -> List[int]:
    if row_count <= 0:
        return []

    sizes = [1 for _ in range(row_count)]
    desired_total = int(round(row_count * average_chord_size))
    desired_total = max(row_count, min(row_count * max_chord_size, desired_total))
    extra_units = desired_total - row_count
    if extra_units <= 0 or max_chord_size <= 1:
        return sizes

    if max_chord_size >= 3 and target >= 5.75 and extra_units >= 2:
        triple_budget = min(
            extra_units // 2,
            int(round(row_count * _speed_triple_ratio(target, temperature))),
        )
        triple_candidates = _speed_chord_candidates(row_count, rows, accent_snap_points, target, for_triples=True)
        triples = _pick_non_adjacent(
            triple_candidates,
            triple_budget,
            temperature,
            salt=int(target * 811) + row_count,
            radius=3,
        )
        for index in triples:
            sizes[index] = 3
        extra_units -= len(triples) * 2

    double_candidates = _speed_chord_candidates(row_count, rows, accent_snap_points, target, for_triples=False)
    double_candidates = [
        index
        for index in double_candidates
        if sizes[index] == 1 and not _is_next_to_stream_triple(sizes, index)
    ]
    selected = _spread_pick(
        double_candidates,
        min(extra_units, len(double_candidates)),
        temperature,
        salt=int(target * 853) + desired_total,
    )
    for index in selected:
        sizes[index] = 2
    extra_units -= len(selected)

    if extra_units > 0:
        fallback = [
            index
            for index, size in enumerate(sizes)
            if size == 1 and not _is_next_to_stream_triple(sizes, index)
        ]
        selected = _spread_pick(fallback, min(extra_units, len(fallback)), temperature, salt=int(target * 907))
        for index in selected:
            sizes[index] = 2

    _thin_speed_chord_runs(sizes, target)
    _shape_speed_chord_bursts(sizes, rows, accent_snap_points, target, max_chord_size, temperature)
    return sizes


def _speed_triple_ratio(target: float, temperature: float) -> float:
    if target < 5.75:
        return 0.0
    if target < 6.75:
        base = 0.015
    else:
        base = 0.035
    return max(0.0, min(0.06, base * (0.75 + temperature * 0.50)))


def _speed_chord_candidates(
    row_count: int,
    rows: List[int],
    accent_snap_points: Set[int],
    target: float,
    for_triples: bool,
) -> List[int]:
    if row_count <= 0:
        return []

    groups: List[List[int]] = [
        [index for index, time_ms in enumerate(rows) if time_ms in accent_snap_points],
        [index for index in range(row_count) if index % 8 == 0],
        [index for index in range(row_count) if index % 16 in [4, 12]],
    ]
    if target >= 4.75 and not for_triples:
        groups.append([index for index in range(row_count) if index % 4 == 2])
    if target >= 5.75 and not for_triples:
        groups.append([index for index in range(row_count) if index % 2 == 1])
    if target >= 6.50 and not for_triples:
        groups.append(list(range(row_count)))

    ordered: List[int] = []
    seen: Set[int] = set()
    for group in groups:
        for index in group:
            if index in seen:
                continue
            ordered.append(index)
            seen.add(index)
    return ordered


def _thin_speed_chord_runs(sizes: List[int], target: float) -> None:
    if target >= 6.75:
        max_run = 5
    elif target >= 5.75:
        max_run = 3
    else:
        max_run = 2

    run = 0
    demoted = 0
    for index, size in enumerate(sizes):
        if size >= 2:
            run += 1
            if run > max_run:
                sizes[index] = 1
                demoted += size - 1
                run = 0
        else:
            run = 0

    if demoted <= 0:
        return

    for index, size in enumerate(sizes):
        if demoted <= 0:
            break
        if size != 1:
            continue
        prev_chord = index > 0 and sizes[index - 1] >= 2
        next_chord = index + 1 < len(sizes) and sizes[index + 1] >= 2
        if prev_chord and next_chord:
            continue
        sizes[index] = 2
        demoted -= 1


def _shape_speed_chord_bursts(
    sizes: List[int],
    rows: List[int],
    accent_snap_points: Set[int],
    target: float,
    max_chord_size: int,
    temperature: float,
) -> None:
    if target < 5.5 or max_chord_size < 2 or len(sizes) < 48:
        return

    block = 64 if target >= 6.0 else 48
    burst_span = 18 if target >= 6.0 else 14
    burst_slots = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    if temperature >= 0.55:
        burst_slots = [0, 1, 3, 5, 8, 10, 13, 15, 17]

    for block_start in range(0, len(sizes), block):
        block_end = min(len(sizes), block_start + block)
        if block_end - block_start < 24:
            continue
        start_shift = _deterministic_index(3, block_start, int(target * 100), int(temperature * 1000)) * 2
        promoted: List[int] = []
        for slot in burst_slots:
            index = block_start + start_shift + slot
            if index >= min(block_end, block_start + burst_span + start_shift):
                continue
            if index >= len(sizes) or sizes[index] > 1:
                continue
            sizes[index] = 2
            promoted.append(index)

        if max_chord_size >= 3 and target >= 6.25:
            triple_candidates = [
                index
                for index in promoted
                if rows[index] in accent_snap_points or index % 8 in [0, 4]
            ]
            for index in triple_candidates[:2]:
                sizes[index] = 3

        if not promoted:
            continue

        demote_needed = sum(max(0, sizes[index] - 1) for index in promoted)
        demote_candidates = [
            index
            for index in range(block_start + burst_span + start_shift, block_end)
            if index < len(sizes)
            and sizes[index] > 1
            and rows[index] not in accent_snap_points
        ]
        demote_candidates.sort(
            key=lambda index: (
                sizes[index],
                _deterministic_index(311, index, block_start, int(target * 100)),
            )
        )
        for index in demote_candidates:
            if demote_needed <= 0:
                break
            while sizes[index] > 1 and demote_needed > 0:
                sizes[index] -= 1
                demote_needed -= 1


def _build_speed_notes(
    rows: List[int],
    sizes: List[int],
    target: float,
    temperature: float,
) -> List[NoteObject]:
    notes: List[NoteObject] = []
    recent_lanes: List[List[int]] = []

    for row_index, (time_ms, size) in enumerate(zip(rows, sizes)):
        lanes = _next_speed_lanes_chaotic(
            size,
            recent_lanes,
            row_index,
            target,
            temperature,
        )
        recent_lanes.append(lanes)
        if len(recent_lanes) > 24:
            recent_lanes.pop(0)

        for lane in lanes:
            notes.append(NoteObject(time_ms=time_ms, lane=lane))

    return notes


def _speed_phrase_length(target: float, temperature: float, phrase_index: int) -> int:
    base = 16 if target < 4.5 else 12 if target < 6.0 else 8
    if temperature < 0.30:
        return base
    jitter_span = 2 if temperature < 0.75 else 4
    return max(4, base + _deterministic_jitter(phrase_index, int(target * 271), jitter_span))


def _speed_phrase_order(phrase_index: int, temperature: float) -> List[int]:
    orders = [
        [0, 3, 2, 1],
        [2, 1, 0, 3],
        [0, 2, 1, 3],
        [3, 1, 2, 0],
        [1, 3, 0, 2],
        [2, 0, 3, 1],
    ]
    if temperature < 0.25:
        pool = orders[:2]
    elif temperature < 0.65:
        pool = orders[:4]
    else:
        pool = orders
    return list(pool[_deterministic_index(len(pool), phrase_index, int(temperature * 1000))])


def _next_speed_lanes(
    size: int,
    base_lane: int,
    previous_lanes: List[int],
    recent_lanes: List[List[int]],
    row_index: int,
    phrase_index: int,
    target: float,
    temperature: float,
) -> List[int]:
    patterns = _speed_patterns(size)
    if not patterns:
        return []

    previous_set = set(previous_lanes)
    ranked = sorted(
        patterns,
        key=lambda lanes: (
            len(set(lanes) & previous_set) * 100,
            0 if base_lane in lanes else 2,
            _speed_recent_pattern_penalty(lanes, recent_lanes, target),
            abs(sum(lanes) - 3),
            _deterministic_index(113, row_index, phrase_index, size, int(target * 100), int(temperature * 1000)),
            patterns.index(lanes),
        ),
    )

    safest_overlap = len(set(ranked[0]) & previous_set)
    safe_ranked = [lanes for lanes in ranked if len(set(lanes) & previous_set) == safest_overlap]
    if safe_ranked:
        ranked = safe_ranked

    pool_size = 1 + int(round(max(0.0, min(1.0, temperature)) * (len(ranked) - 1)))
    pool = ranked[: max(1, pool_size)]
    choice = _deterministic_index(
        len(pool),
        row_index,
        phrase_index,
        base_lane,
        size,
        sum(previous_lanes) if previous_lanes else 0,
    )
    return list(pool[choice])


def _next_speed_lanes_chaotic(
    size: int,
    recent_lanes: List[List[int]],
    row_index: int,
    target: float,
    temperature: float,
) -> List[int]:
    patterns = _speed_patterns(size)
    if not patterns:
        return []

    prev = set(recent_lanes[-1]) if recent_lanes else set()
    prev2 = set(recent_lanes[-2]) if len(recent_lanes) >= 2 else set()
    prev3 = set(recent_lanes[-3]) if len(recent_lanes) >= 3 else set()
    recent_window = recent_lanes[-16:]
    lane_counts = Counter(lane for lanes in recent_window for lane in lanes)
    pattern_counts = Counter(tuple(lanes) for lanes in recent_lanes[-12:])
    if size <= 1 and len(prev) == 1:
        filtered = [lanes for lanes in patterns if not (len(lanes) == 1 and lanes[0] in prev)]
        if filtered:
            patterns = filtered
        single_tail: List[int] = []
        for lanes in reversed(recent_lanes):
            if len(lanes) != 1:
                break
            single_tail.append(lanes[0])
            if len(single_tail) >= 6:
                break
        if len(single_tail) >= 3 and single_tail[0] == single_tail[2]:
            filtered = [lanes for lanes in patterns if not (len(lanes) == 1 and lanes[0] == single_tail[1])]
            if filtered:
                patterns = filtered
        if len(single_tail) >= 4 and len(set(single_tail[:4])) <= 2:
            stale_lanes = set(single_tail[:4])
            filtered = [lanes for lanes in patterns if not (len(lanes) == 1 and lanes[0] in stale_lanes)]
            if filtered:
                patterns = filtered

    def single_cycle_penalty(lanes: List[int]) -> int:
        if len(lanes) != 1:
            return 0
        lane = lanes[0]
        penalty = 0
        if lane in prev:
            penalty += 10
        if len(prev) == 1 and len(prev2) == 1 and lane in prev2:
            penalty += 4
        if len(prev) == 1 and len(prev2) == 1 and len(prev3) == 1:
            # Avoid ABAB-style cutting as the default speed texture.
            prev_lane = next(iter(prev))
            prev2_lane = next(iter(prev2))
            prev3_lane = next(iter(prev3))
            if lane == prev2_lane and prev_lane == prev3_lane:
                penalty += 8
        return penalty

    def score(lanes: List[int]) -> Tuple[int, int, int, int, int, int]:
        lane_set = set(lanes)
        overlap_prev = len(lane_set & prev)
        overlap_prev2 = len(lane_set & prev2)
        exact_reuse = pattern_counts.get(tuple(lanes), 0)
        pressure = sum(lane_counts.get(lane, 0) for lane in lanes)
        balance = abs(sum(lanes) - (1.5 * len(lanes)))
        if len(lanes) == 1:
            overlap_penalty = overlap_prev * 7 + overlap_prev2
        else:
            # Chords may share a lane with nearby rows in reference speed maps,
            # but exact repeated chord cells should not become jack/stream blocks.
            overlap_penalty = overlap_prev + max(0, overlap_prev2 - 1)
        return (
            single_cycle_penalty(lanes),
            exact_reuse * 3,
            overlap_penalty,
            pressure,
            int(balance * 2),
            _deterministic_index(997, row_index, size, int(target * 100), int(temperature * 1000), sum(lanes)),
        )

    ranked = sorted(patterns, key=score)
    pool_size = 1 + int(round(max(0.0, min(1.0, temperature)) * min(3, len(ranked) - 1)))
    pool = ranked[: max(1, pool_size)]
    choice = _deterministic_index(
        len(pool),
        row_index,
        size,
        sum(sum(lanes) for lanes in recent_lanes[-4:]) if recent_lanes else 0,
        int(target * 100),
    )
    return list(pool[choice])


def _speed_recent_pattern_penalty(lanes: List[int], recent_lanes: List[List[int]], target: float) -> int:
    lane_set = set(lanes)
    penalty = 0
    if len(recent_lanes) >= 2 and lane_set == set(recent_lanes[-2]):
        penalty += 4
    if len(recent_lanes) >= 4:
        penalty += sum(1 for old in recent_lanes[-4:] if set(old) == lane_set)
    if target < 5.5 and len(lanes) >= 2:
        penalty += sum(1 for old in recent_lanes[-3:] if len(old) >= 2) * 2
    return penalty


def _speed_patterns(size: int) -> List[List[int]]:
    if size <= 1:
        return [[0], [1], [2], [3]]
    if size == 2:
        return [[0, 3], [1, 2], [0, 2], [1, 3], [0, 1], [2, 3]]
    return [[0, 1, 3], [0, 2, 3], [0, 1, 2], [1, 2, 3]]


def _repair_speed_single_lane_flow(notes: List[NoteObject], target: float) -> List[NoteObject]:
    if not notes:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)

    row_times = sorted(rows)
    row_gaps = [b - a for a, b in zip(row_times, row_times[1:])]
    median_gap = sorted(row_gaps)[len(row_gaps) // 2] if row_gaps else 999
    flow_gap_ms = max(150, min(260, int(round(median_gap * 2.25))))
    active_ln_until = {lane: -1 for lane in range(4)}
    previous_lanes: Set[int] = set()
    previous_time: Optional[int] = None
    lane_close_run = {lane: 0 for lane in range(4)}
    recent_single_lanes: List[int] = []
    repaired: List[NoteObject] = []

    for row_index, time_ms in enumerate(row_times):
        row_notes = [
            NoteObject(time_ms=note.time_ms, lane=note.lane, end_time_ms=note.end_time_ms)
            for note in sorted(rows[time_ms], key=lambda item: (item.lane, item.end_time_ms or -1))
        ]
        next_lanes = _next_speed_row_lanes(rows, row_times, row_index)
        held_lanes = {
            lane
            for lane, end_time in active_ln_until.items()
            if end_time > time_ms
        }

        close_to_previous = previous_time is not None and time_ms - previous_time <= flow_gap_ms
        if close_to_previous and previous_lanes:
            current_lanes = {note.lane for note in row_notes}
            for note in row_notes:
                if note.lane not in previous_lanes:
                    continue
                direct_single_repeat = len(row_notes) == 1 and len(previous_lanes) == 1
                if not direct_single_repeat and lane_close_run.get(note.lane, 0) < 2:
                    continue
                replacement = _speed_replacement_lane_for_row(
                    time_ms=time_ms,
                    current_lane=note.lane,
                    previous_lanes=previous_lanes,
                    current_lanes=current_lanes,
                    next_lanes=next_lanes,
                    held_lanes=held_lanes,
                    recent_single_lanes=recent_single_lanes,
                    target=target,
                )
                if replacement is not None:
                    current_lanes.discard(note.lane)
                    note.lane = replacement
                    current_lanes.add(replacement)

        if len(row_notes) == 1:
            note = row_notes[0]
            recent_single_lanes.append(note.lane)
            if len(recent_single_lanes) > 8:
                recent_single_lanes.pop(0)

        for note in row_notes:
            repaired.append(note)
            if note.end_time_ms is not None:
                active_ln_until[note.lane] = max(active_ln_until[note.lane], note.end_time_ms)
        current_lanes = {note.lane for note in row_notes}
        updated_runs = {lane: 0 for lane in range(4)}
        for lane in current_lanes:
            if close_to_previous and lane in previous_lanes:
                updated_runs[lane] = lane_close_run.get(lane, 0) + 1
            else:
                updated_runs[lane] = 1
        lane_close_run = updated_runs
        previous_lanes = current_lanes
        previous_time = time_ms

    return sorted(repaired, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _next_speed_row_lanes(
    rows: Dict[int, List[NoteObject]],
    row_times: List[int],
    row_index: int,
) -> Set[int]:
    if row_index + 1 >= len(row_times):
        return set()
    return {note.lane for note in rows.get(row_times[row_index + 1], [])}


def _fortify_speed_accent_chords(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    if not notes or config.key_style != "speed":
        return notes

    max_chord_size = min(3, clamp_max_chord_size(config, "speed"))
    if max_chord_size <= 1:
        return notes

    target = config.target_star or 4.0
    rows: Dict[int, List[NoteObject]] = {}
    row_lanes: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
        row_lanes.setdefault(note.time_ms, set()).add(note.lane)

    row_times = sorted(row_lanes)
    if not row_times:
        return notes

    row_count = len(row_times)
    current_chords = sum(1 for lanes in row_lanes.values() if len(lanes) > 1)
    target_chords = int(round(row_count * _speed_target_accent_chord_ratio(target)))
    if current_chords >= target_chords:
        return notes

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    context = _music_context(analysis, snap_points, accent_snap_points)
    if not context:
        return notes

    needed = target_chords - current_chords
    max_additions = min(
        needed,
        max(1, int(round(row_count * _speed_accent_chord_upgrade_cap(target)))),
    )
    threshold = _speed_accent_chord_threshold(target)
    additions: List[NoteObject] = []

    candidates = sorted(
        (
            (time_ms, entry, _chord_music_strength_at(analysis, snap_points, accent_snap_points, time_ms))
            for time_ms, entry in context.items()
            if entry.get("silent", 0.0) <= 0.0
        ),
        key=lambda item: (-item[2], -item[1].get("accent", 0.0), item[0]),
    )

    for anchor_time, entry, strength in candidates:
        if len(additions) >= max_additions:
            break
        if strength < threshold and entry.get("accent", 0.0) < threshold + 0.04:
            break

        row_time = _nearest_music_row_time(row_times, anchor_time, "speed", beat_length)
        if row_time is None:
            continue
        lanes_at_time = row_lanes.get(row_time, set())
        desired_size = _speed_accent_desired_chord_size(entry, strength, max_chord_size, target)
        if len(lanes_at_time) >= desired_size:
            continue
        if _speed_chord_run_if_upgraded(row_times, row_lanes, row_time) > _speed_accent_chord_run_limit(target):
            continue

        while len(lanes_at_time) < desired_size and len(additions) < max_additions:
            lane = _music_receiver_lane_for_style(
                row_time,
                row_lanes,
                row_times,
                entry,
                max_chord_size,
                "speed",
                target,
                beat_length,
            )
            if lane is None:
                break
            lanes_at_time.add(lane)
            additions.append(NoteObject(time_ms=row_time, lane=lane))

    if not additions:
        return notes
    return sorted(notes + additions, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _cap_speed_chord_density(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    if not notes or config.key_style != "speed":
        return notes

    target = config.target_star or 4.0
    max_ratio = _speed_max_chord_ratio(target)

    rows: Dict[int, List[NoteObject]] = {}
    row_lanes: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
        row_lanes.setdefault(note.time_ms, set()).add(note.lane)

    row_times = sorted(row_lanes)
    if not row_times:
        return notes

    removed: Set[int] = set()
    row_time_lookup = sorted(row_lanes)
    max_row_size = _speed_max_row_chord_size(target)
    for time_ms in row_times:
        while len(row_lanes.get(time_ms, set())) > max_row_size:
            row_notes = rows.get(time_ms, [])
            note_to_remove = _pick_chord_weight_donor_note(time_ms, row_notes, row_lanes, row_time_lookup, removed, "speed")
            if note_to_remove is None:
                break
            removed.add(id(note_to_remove))
            row_lanes[time_ms].discard(note_to_remove.lane)

    chord_times = [time_ms for time_ms in row_times if len(row_lanes.get(time_ms, set())) > 1]
    max_chords = int(round(len(row_times) * max_ratio))
    if len(chord_times) <= max_chords:
        if removed:
            return sorted(
                [note for note in notes if id(note) not in removed],
                key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1),
            )
        return notes

    strengths = {
        time_ms: _chord_music_strength_at(analysis, snap_points, accent_snap_points, time_ms)
        for time_ms in chord_times
    }
    removable = sorted(
        chord_times,
        key=lambda time_ms: (strengths.get(time_ms, 0.0), -len(row_lanes.get(time_ms, set())), time_ms),
    )

    excess = len(chord_times) - max_chords
    for time_ms in removable:
        if excess <= 0:
            break
        row_notes = rows.get(time_ms, [])
        note_to_remove = _pick_chord_weight_donor_note(time_ms, row_notes, row_lanes, row_time_lookup, removed, "speed")
        if note_to_remove is None:
            continue
        removed.add(id(note_to_remove))
        row_lanes[time_ms].discard(note_to_remove.lane)
        excess -= 1

    if not removed:
        return notes
    return sorted(
        [note for note in notes if id(note) not in removed],
        key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1),
    )


def _speed_max_chord_ratio(target: float) -> float:
    if target < 4.5:
        return 0.135
    if target < 5.5:
        return 0.155
    if target < 6.5:
        return 0.175
    return 0.280


def _thin_speed_chord_clusters(
    rows: Dict[int, List[NoteObject]],
    row_lanes: Dict[int, Set[int]],
    row_times: List[int],
    removed: Set[int],
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    target: float,
) -> None:
    chord_times = [time_ms for time_ms in row_times if len(row_lanes.get(time_ms, set())) > 1]
    if len(chord_times) < 2:
        return

    min_spacing = _speed_min_chord_spacing_ms(target, analysis.get("bpm", 120.0))
    strengths = {
        time_ms: _chord_music_strength_at(analysis, snap_points, accent_snap_points, time_ms)
        for time_ms in chord_times
    }
    kept: List[int] = []
    for time_ms in sorted(chord_times, key=lambda item: (-strengths.get(item, 0.0), item)):
        if any(abs(time_ms - kept_time) < min_spacing for kept_time in kept):
            row_notes = rows.get(time_ms, [])
            while len(row_lanes.get(time_ms, set())) > 1:
                note_to_remove = _pick_chord_weight_donor_note(time_ms, row_notes, row_lanes, row_times, removed, "speed")
                if note_to_remove is None:
                    break
                removed.add(id(note_to_remove))
                row_lanes[time_ms].discard(note_to_remove.lane)
            continue
        kept.append(time_ms)


def _speed_min_chord_spacing_ms(target: float, bpm: float) -> int:
    beat_length = 60000.0 / max(1.0, bpm)
    if target < 4.5:
        return max(300, int(round(beat_length * 0.78)))
    if target < 5.5:
        return max(320, int(round(beat_length * 0.78)))
    if target < 6.5:
        return max(300, int(round(beat_length * 0.72)))
    return max(320, int(round(beat_length * 0.74)))


def _speed_max_row_chord_size(target: float) -> int:
    if target < 5.75:
        return 2
    return 3


def _speed_target_accent_chord_ratio(target: float) -> float:
    if target < 3.5:
        return 0.060
    if target < 4.5:
        return 0.085
    if target < 5.5:
        return 0.130
    if target < 6.0:
        return 0.120
    if target < 6.5:
        return 0.145
    return 0.250


def _speed_accent_chord_upgrade_cap(target: float) -> float:
    if target < 4.5:
        return 0.045
    if target < 5.5:
        return 0.075
    if target < 6.0:
        return 0.055
    if target < 6.5:
        return 0.075
    return 0.140


def _speed_accent_chord_threshold(target: float) -> float:
    if target < 4.5:
        return 0.48
    if target < 5.5:
        return 0.36
    if target < 6.0:
        return 0.40
    if target < 6.5:
        return 0.34
    return 0.34


def _speed_accent_desired_chord_size(
    entry: Dict[str, float],
    strength: float,
    max_chord_size: int,
    target: float,
) -> int:
    desired = 2
    if (
        max_chord_size >= 3
        and target >= 5.9
        and (
            strength >= 0.72
            or entry.get("kick", 0.0) >= 0.68
            or entry.get("stack", 0.0) >= 0.78
        )
    ):
        desired = 3
    return min(max_chord_size, desired)


def _speed_accent_chord_run_limit(target: float) -> int:
    if target < 5.5:
        return 2
    if target < 6.5:
        return 3
    return 4


def _speed_chord_run_if_upgraded(
    row_times: List[int],
    row_lanes: Dict[int, Set[int]],
    row_time: int,
) -> int:
    idx = bisect.bisect_left(row_times, row_time)
    if idx >= len(row_times) or row_times[idx] != row_time:
        return 0
    run = 1
    left = idx - 1
    while left >= 0 and len(row_lanes.get(row_times[left], set())) > 1:
        run += 1
        left -= 1
    right = idx + 1
    while right < len(row_times) and len(row_lanes.get(row_times[right], set())) > 1:
        run += 1
        right += 1
    return run


def _speed_replacement_lane_for_row(
    time_ms: int,
    current_lane: int,
    previous_lanes: Set[int],
    current_lanes: Set[int],
    next_lanes: Set[int],
    held_lanes: Set[int],
    recent_single_lanes: List[int],
    target: float,
) -> Optional[int]:
    candidates = [
        lane
        for lane in range(4)
        if lane not in previous_lanes
        and lane not in current_lanes
        and lane not in held_lanes
    ]
    if not candidates:
        candidates = [
            lane
            for lane in range(4)
            if lane not in previous_lanes and lane not in current_lanes
        ]
    if not candidates:
        return None

    recent_counts = Counter(recent_single_lanes[-6:])
    last_direction = 0
    if len(recent_single_lanes) >= 2:
        last_direction = recent_single_lanes[-1] - recent_single_lanes[-2]
    previous_center = sum(previous_lanes) / max(1, len(previous_lanes))

    def score(lane: int) -> Tuple[int, int, int, int, int]:
        same_as_next = 1 if lane in next_lanes else 0
        edge_bias = 0 if target >= 5.75 else abs(lane - 1)
        direction_penalty = 0
        if last_direction != 0:
            projected = int(round(previous_center + (1 if last_direction > 0 else -1)))
            if 0 <= projected <= 3 and lane != projected:
                direction_penalty = 1
        return (
            same_as_next * 4,
            recent_counts.get(lane, 0),
            direction_penalty,
            edge_bias,
            _deterministic_index(97, time_ms, current_lane, sum(previous_lanes), lane),
        )

    return sorted(candidates, key=score)[0]


def _pattern_temperature(config: DifficultyConfig) -> float:
    value = getattr(config, "pattern_temperature", 0.35)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.35


def _jack_profile_min_chord_size(target: float, temperature: float = 0.35) -> int:
    if target >= 6.35:
        return 2 if temperature >= 0.70 else 3
    if target >= 4.90:
        return 2
    return 1


def _jack_profile_max_chord_size(target: float) -> int:
    if target < 3.75:
        return 2
    if target < 4.75:
        return 3
    return 4


def _build_jack_chord_sizes(
    row_count: int,
    average_chord_size: float,
    min_chord_size: int,
    max_chord_size: int,
    rows: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
    target: float,
    temperature: float,
) -> List[int]:
    if row_count <= 0:
        return []

    sizes = [min_chord_size for _ in range(row_count)]
    desired_total = int(round(row_count * average_chord_size))
    desired_total = max(row_count * min_chord_size, min(row_count * max_chord_size, desired_total))
    extra_units = desired_total - (row_count * min_chord_size)

    for next_size in range(min_chord_size + 1, max_chord_size + 1):
        if extra_units <= 0:
            break
        eligible = [index for index, size in enumerate(sizes) if size == next_size - 1]
        if not eligible:
            continue

        promote_count = min(extra_units, len(eligible))
        selected = _select_jack_upgrade_rows(
            eligible,
            rows,
            accent_snap_points,
            analysis,
            promote_count,
            temperature,
            next_size,
        )
        for index in selected:
            sizes[index] = next_size
        extra_units -= len(selected)

    _add_high_temperature_chord_variance(sizes, min_chord_size, max_chord_size, temperature, target)

    if max_chord_size >= 4:
        _limit_jack_quad_runs(sizes, max_streak=1 if target < 7.25 else 2)

    return sizes


def _add_high_temperature_chord_variance(
    sizes: List[int],
    min_chord_size: int,
    max_chord_size: int,
    temperature: float,
    target: float,
) -> None:
    if temperature < 0.75 or min_chord_size > 2 or max_chord_size < 4:
        return

    row_count = len(sizes)
    if row_count <= 0:
        return

    variance_ratio = 0.05 + ((temperature - 0.75) / 0.25) * 0.11
    if target < 6.0:
        variance_ratio *= 0.65
    swap_count = max(0, int(row_count * variance_ratio))
    if swap_count <= 0:
        return

    existing_quads = {index for index, size in enumerate(sizes) if size == 4}
    peak_candidates = [
        index
        for index, size in enumerate(sizes)
        if size == 3
        and (index - 1) not in existing_quads
        and (index + 1) not in existing_quads
    ]
    selected_peaks: List[int] = []
    blocked = set(existing_quads)
    for index in _spread_pick(peak_candidates, swap_count, temperature, salt=int(target * 177)):
        if index in blocked or (index - 1) in blocked or (index + 1) in blocked:
            continue
        selected_peaks.append(index)
        blocked.add(index)

    if not selected_peaks:
        return

    selected_peak_set = set(selected_peaks)
    valley_candidates = [
        index
        for index, size in enumerate(sizes)
        if size == 3 and index not in selected_peak_set
    ]
    selected_valleys = _spread_pick(
        valley_candidates,
        len(selected_peaks),
        temperature,
        salt=int(target * 223) + len(selected_peaks),
    )
    pair_count = min(len(selected_peaks), len(selected_valleys))
    if pair_count <= 0:
        return

    for index in selected_peaks[:pair_count]:
        sizes[index] = 4
    for index in selected_valleys[:pair_count]:
        sizes[index] = 2


def _select_jack_upgrade_rows(
    eligible: List[int],
    rows: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
    promote_count: int,
    temperature: float,
    next_size: int,
) -> List[int]:
    if promote_count <= 0:
        return []

    if temperature >= 0.75 and next_size < 3:
        return _spread_pick(
            eligible,
            promote_count,
            temperature,
            salt=next_size * 193 + promote_count,
        )

    eligible_set = set(eligible)
    if next_size >= 3:
        music_ranked = _music_ranked_row_indexes(
            eligible,
            rows,
            accent_snap_points,
            analysis,
            min_score=0.20 if next_size == 3 else 0.34,
        )
        groups = [
            music_ranked,
            [index for index in eligible if rows[index] in accent_snap_points and _row_chord_upgrade_score(index, rows, accent_snap_points, analysis) >= 0.16],
            [index for index in eligible if rows[index] not in accent_snap_points and index % 4 == 0],
            [index for index in eligible if rows[index] not in accent_snap_points and index % 2 == 0],
            [index for index in eligible if index % 2 == 1],
        ]
    else:
        groups = [
            [index for index in eligible if rows[index] in accent_snap_points],
            _music_ranked_row_indexes(eligible, rows, accent_snap_points, analysis, min_score=0.12),
            [index for index in eligible if rows[index] not in accent_snap_points and index % 4 == 0],
            [index for index in eligible if rows[index] not in accent_snap_points and index % 2 == 0],
            [index for index in eligible if index % 2 == 1],
        ]

    selected: List[int] = []
    selected_set: Set[int] = set()
    remaining = promote_count

    for group in groups:
        group = [index for index in group if index in eligible_set and index not in selected_set]
        if not group or remaining <= 0:
            continue

        if next_size >= 3:
            group.sort(key=lambda index: (-_row_chord_upgrade_score(index, rows, accent_snap_points, analysis), index))
        else:
            group.sort()
        if remaining >= len(group):
            picks = group
        else:
            if next_size >= 3:
                pool_size = min(len(group), max(remaining, int(round(remaining * (1.8 + temperature)))))
                pool = sorted(group[:pool_size])
                picks = _spread_pick(pool, remaining, temperature, salt=next_size * 97 + len(selected))
            else:
                picks = _spread_pick(group, remaining, temperature, salt=next_size * 97 + len(selected))

        for index in picks:
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            remaining -= 1
            if remaining <= 0:
                break

    if remaining > 0:
        fallback = [index for index in eligible if index not in selected_set]
        for index in _spread_pick(fallback, remaining, temperature, salt=next_size * 131):
            selected.append(index)
            selected_set.add(index)

    return selected


def _spread_pick(values: List[int], count: int, temperature: float = 0.0, salt: int = 0) -> List[int]:
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return list(values)

    picked: List[int] = []
    used: Set[int] = set()
    average_spacing = len(values) / max(1, count)
    jitter_span = int(round(average_spacing * max(0.0, min(1.0, temperature)) * 1.75))
    for pick_index in range(count):
        raw = int(round(pick_index * (len(values) - 1) / max(1, count - 1)))
        raw += _deterministic_jitter(pick_index, salt, jitter_span)
        raw = max(0, min(len(values) - 1, raw))
        while raw in used and raw + 1 < len(values):
            raw += 1
        while raw in used and raw - 1 >= 0:
            raw -= 1
        if raw in used:
            continue
        used.add(raw)
        picked.append(values[raw])
    return picked


def _deterministic_jitter(index: int, salt: int, span: int) -> int:
    if span <= 0:
        return 0
    value = (index * 1103515245 + salt * 12345 + 0x45D9F3B) & 0x7FFFFFFF
    return (value % (span * 2 + 1)) - span


def _limit_jack_quad_runs(sizes: List[int], max_streak: int) -> None:
    streak = 0
    downgraded = 0
    for index, size in enumerate(sizes):
        if size == 4:
            streak += 1
            if streak > max_streak:
                sizes[index] = 3
                downgraded += 1
                streak = 0
        else:
            streak = 0

    if downgraded <= 0:
        return

    for index, size in enumerate(sizes):
        if downgraded <= 0:
            break
        if size != 3:
            continue
        prev_is_quad = index > 0 and sizes[index - 1] == 4
        next_is_quad = index + 1 < len(sizes) and sizes[index + 1] == 4
        if prev_is_quad or next_is_quad:
            continue
        sizes[index] = 4
        downgraded -= 1


def _build_jack_stack_notes(
    rows: List[int],
    sizes: List[int],
    target: float,
    temperature: float = 0.35,
) -> List[NoteObject]:
    notes: List[NoteObject] = []
    current_lanes: List[int] = []
    lane_pressure: Optional[List[int]] = [0, 0, 0, 0] if target < 5.75 else None
    phrase_index = -1
    next_phrase_start = 0

    for index, (time_ms, size) in enumerate(zip(rows, sizes)):
        phrase_boundary = index >= next_phrase_start
        if phrase_boundary:
            phrase_index += 1
            phrase_len = _jack_stack_phrase_length(target, temperature, phrase_index)
            next_phrase_start = index + phrase_len
            current_lanes = _next_jack_stack_pattern(
                size,
                current_lanes,
                phrase_index,
                temperature,
                lane_pressure,
            )
        else:
            current_lanes = _resize_jack_stack(
                current_lanes,
                size,
                phrase_index,
                temperature,
                lane_pressure,
            )
            current_lanes = _vary_jack_stack_inside_phrase(
                current_lanes,
                size,
                phrase_index,
                index,
                target,
                temperature,
                lane_pressure,
            )

        for lane in current_lanes[:size]:
            notes.append(NoteObject(time_ms=time_ms, lane=lane))
            if lane_pressure is not None and 0 <= lane <= 3:
                lane_pressure[lane] += 1

    return notes


def _jack_stack_phrase_length(target: float, temperature: float, phrase_index: int) -> int:
    if target < 3.75:
        base = 8
        minimum = 6
    elif target < 5.25:
        base = 6
        minimum = 4
    else:
        base = 4
        minimum = 3

    if temperature <= 0.20:
        return base + 2
    if temperature < 0.55:
        return base

    jitter_span = 1 if temperature < 0.80 else 2
    return max(minimum, base + _deterministic_jitter(phrase_index, int(target * 100), jitter_span))


def _next_jack_stack_pattern(
    size: int,
    current_lanes: List[int],
    phrase_index: int,
    temperature: float = 0.35,
    lane_pressure: Optional[List[int]] = None,
) -> List[int]:
    patterns = _jack_stack_patterns(size)
    if not patterns:
        return []

    current_set = set(current_lanes)
    candidates = [lanes for lanes in patterns if set(lanes) != current_set]
    if not candidates:
        candidates = patterns

    if current_set and size > 1:
        overlapping = [lanes for lanes in candidates if set(lanes) & current_set]
        if overlapping:
            candidates = overlapping

    ranked = sorted(
        candidates,
        key=lambda lanes: (
            -(len(set(lanes) & current_set) if current_set else 0),
            _jack_stack_lane_pressure_score(lanes, lane_pressure),
            (patterns.index(lanes) - phrase_index) % len(patterns),
        ),
    )

    pool_size = 1 + int(round(max(0.0, min(1.0, temperature)) * (len(ranked) - 1)))
    if lane_pressure and max(lane_pressure) - min(lane_pressure) >= 6:
        pool_size = min(pool_size, 2)
    pool = ranked[: max(1, pool_size)]
    choice_index = _deterministic_index(
        len(pool),
        phrase_index,
        size,
        int(temperature * 100),
        sum(current_lanes) if current_lanes else 0,
        int(sum(lane_pressure or []) * 3),
    )
    return list(pool[choice_index])


def _vary_jack_stack_inside_phrase(
    current_lanes: List[int],
    size: int,
    phrase_index: int,
    row_index: int,
    target: float,
    temperature: float,
    lane_pressure: Optional[List[int]] = None,
) -> List[int]:
    if temperature < 0.55 or size >= 4:
        return current_lanes
    if size <= 1 and target < 3.75:
        return current_lanes

    interval = 4
    if temperature >= 0.90:
        interval = 2
    elif temperature >= 0.75:
        interval = 3

    trigger = _deterministic_index(interval, phrase_index, int(target * 100), size)
    if row_index % interval != trigger:
        return current_lanes

    current_set = set(current_lanes)
    patterns = _jack_stack_patterns(size)
    if size <= 1:
        min_overlap = 0
    elif size == 2:
        min_overlap = 1
    else:
        min_overlap = 2

    candidates = [
        lanes
        for lanes in patterns
        if set(lanes) != current_set and len(set(lanes) & current_set) >= min_overlap
    ]
    if not candidates:
        return current_lanes

    patterns_by_key = _jack_stack_patterns(size)
    ranked = sorted(
        candidates,
        key=lambda lanes: (
            _jack_stack_lane_pressure_score(lanes, lane_pressure),
            (patterns_by_key.index(lanes) - phrase_index - row_index) % len(patterns_by_key),
        ),
    )
    pool_size = 1 + int(round(max(0.0, min(1.0, temperature)) * (len(ranked) - 1)))
    if lane_pressure and max(lane_pressure) - min(lane_pressure) >= 6:
        pool_size = min(pool_size, 2)
    pool = ranked[: max(1, pool_size)]
    choice = _deterministic_index(
        len(pool),
        phrase_index,
        row_index,
        size,
        int(temperature * 1000),
        sum(current_lanes),
        int(sum(lane_pressure or []) * 5),
    )
    return list(pool[choice])


def _resize_jack_stack(
    current_lanes: List[int],
    size: int,
    phrase_index: int,
    temperature: float = 0.35,
    lane_pressure: Optional[List[int]] = None,
) -> List[int]:
    current = [lane for lane in current_lanes if lane in [0, 1, 2, 3]]
    if len(current) == size:
        return current
    if not current:
        return _next_jack_stack_pattern(size, [], phrase_index, temperature, lane_pressure)
    if size == 4:
        return [0, 1, 2, 3]
    if len(current) > size:
        if len(current) == 4:
            return _next_jack_stack_pattern(size, current, phrase_index, temperature, lane_pressure)
        subsets = [list(combo) for combo in combinations(current, size)]
        if not subsets:
            return current[:size]
        ranked = sorted(
            subsets,
            key=lambda lanes: (
                _jack_stack_lane_pressure_score(lanes, lane_pressure),
                _deterministic_index(97, phrase_index, size, sum(lanes), int(temperature * 100)),
            ),
        )
        return list(ranked[0])

    target_pattern = _next_jack_stack_pattern(size, current, phrase_index, temperature, lane_pressure)
    expanded = list(current)
    for lane in target_pattern:
        if lane not in expanded:
            expanded.append(lane)
        if len(expanded) >= size:
            break
    fallback_lanes = sorted(
        [lane for lane in [0, 1, 2, 3] if lane not in expanded],
        key=lambda lane: (
            (lane_pressure[lane] if lane_pressure and 0 <= lane < len(lane_pressure) else 0),
            0 if lane in [0, 3] else 1,
            lane,
        ),
    )
    for lane in fallback_lanes:
        if lane not in expanded:
            expanded.append(lane)
        if len(expanded) >= size:
            break
    return expanded[:size]


def _jack_stack_lane_pressure_score(lanes: List[int], lane_pressure: Optional[List[int]]) -> float:
    if not lane_pressure or not lanes:
        return 0.0

    valid_lanes = [lane for lane in lanes if 0 <= lane < len(lane_pressure)]
    if not valid_lanes:
        return 0.0

    average_pressure = sum(lane_pressure[:4]) / 4.0
    selected_pressure = sum(lane_pressure[lane] for lane in valid_lanes) / max(1, len(valid_lanes))
    center_excess = max(0.0, lane_pressure[1] - average_pressure) + max(0.0, lane_pressure[2] - average_pressure)
    outer_shortage = max(0.0, average_pressure - lane_pressure[0]) + max(0.0, average_pressure - lane_pressure[3])
    center_count = sum(1 for lane in valid_lanes if lane in [1, 2])
    outer_count = len(valid_lanes) - center_count

    score = selected_pressure
    if center_excess > 0.0 or outer_shortage > 0.0:
        score += (center_count - outer_count * 0.55) * 0.28 * (center_excess + outer_shortage)
    if set(valid_lanes) == {1, 2}:
        # The 2+3 stack is valid jack texture, but if it becomes the default
        # receiver it overloads the middle fingers compared with reference maps.
        score += max(0.0, center_excess) * 0.55 + 1.5
    return score


def _rebalance_jack_lane_pressure(
    notes: List[NoteObject],
    target: float,
    temperature: float,
) -> List[NoteObject]:
    if not notes:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    row_times = sorted(rows)
    if len(row_times) < 8:
        return notes

    lane_counts = Counter(note.lane for note in notes if 0 <= note.lane <= 3)
    total_notes = sum(lane_counts.values())
    if total_notes <= 0:
        return notes

    center_cap = 0.295 if target < 6.25 else 0.315
    outer_floor = 0.165 if target < 6.25 else 0.135
    change_ratio = 0.18 + max(0.0, min(1.0, temperature)) * 0.10
    if target >= 6.25:
        change_ratio = 0.055 + max(0.0, min(1.0, temperature)) * 0.035
    max_changes = max(1, int(round(len(row_times) * change_ratio)))
    changes = 0

    for pass_index in range(2):
        if changes >= max_changes:
            break
        for row_pos, time_ms in enumerate(row_times):
            if changes >= max_changes:
                break

            overloaded = [
                lane
                for lane in [1, 2]
                if lane_counts[lane] / max(1, total_notes) > center_cap
            ]
            underloaded = [
                lane
                for lane in [0, 3]
                if lane_counts[lane] / max(1, total_notes) < outer_floor
            ]
            if not overloaded and not underloaded:
                return _flatten_note_rows(rows)

            row_notes = sorted(rows.get(time_ms, []), key=lambda item: (item.lane, item.end_time_ms or -1))
            current_lanes = sorted({note.lane for note in row_notes if 0 <= note.lane <= 3})
            size = len(current_lanes)
            if size <= 0 or size >= 4:
                continue
            if not any(lane in current_lanes for lane in overloaded or [1, 2]):
                continue

            prev_lanes = (
                sorted({note.lane for note in rows[row_times[row_pos - 1]] if 0 <= note.lane <= 3})
                if row_pos > 0
                else []
            )
            next_lanes = (
                sorted({note.lane for note in rows[row_times[row_pos + 1]] if 0 <= note.lane <= 3})
                if row_pos + 1 < len(row_times)
                else []
            )
            replacement = _choose_jack_pressure_replacement(
                current_lanes,
                prev_lanes,
                next_lanes,
                lane_counts,
                total_notes,
                center_cap,
                outer_floor,
                row_pos,
                pass_index,
                target,
            )
            if replacement == current_lanes:
                continue

            for lane in current_lanes:
                lane_counts[lane] -= 1
            for lane in replacement:
                lane_counts[lane] += 1
            rows[time_ms] = [
                NoteObject(
                    time_ms=note.time_ms,
                    lane=lane,
                    end_time_ms=note.end_time_ms,
                )
                for note, lane in zip(row_notes, replacement)
            ]
            changes += 1

    return _flatten_note_rows(rows)



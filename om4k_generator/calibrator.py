import random
import bisect
import math
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from .difficulty_estimator import DifficultyEstimator
from .grid_builder import GridBuilder
from .models import DifficultyConfig, NoteObject
from .pattern_generator import PatternGenerator
from .style_rules import clamp_max_chord_size, normalize_hybrid_weights, recommended_subdivisions
from .validator import Validator


def build_snap_candidates(analysis: Dict[str, Any], config: DifficultyConfig) -> List[int]:
    allowed_subdivisions = list(config.allowed_subdivisions)
    if not allowed_subdivisions:
        allowed_subdivisions = recommended_subdivisions(
            analysis["bpm"],
            config.chart_type,
            config.key_style,
            config.target_star,
        )
    if config.chart_type == "hybrid":
        allowed_subdivisions = [
            subdivision
            for subdivision in allowed_subdivisions
            if subdivision in ["1/2", "1/4", "1/8"]
        ] or ["1/2", "1/4", "1/8"]
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


def generate_to_target_sr(
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    tolerance: float = 0.15,
    max_attempts: int = 24,
) -> Tuple[List[NoteObject], float, bool, int]:
    accent_snap_points = build_accent_snap_points(analysis, snap_points)
    _music_context(analysis, snap_points, accent_snap_points)

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
        jack_sr = DifficultyEstimator.estimate_sr(jack_notes, analysis["duration_ms"])
        return jack_notes, jack_sr, abs(jack_sr - target) <= tolerance, 1

    if config.chart_type in ["rice", "ln"] and config.key_style == "stream":
        stream_notes = _shape_stream_profile([], config, analysis, snap_points, accent_snap_points)
        stream_notes = _refine_lns_to_sustain_and_hits(stream_notes, config, analysis, snap_points, accent_snap_points)
        stream_notes = Validator.validate_and_fix(stream_notes, config, analysis["silent_regions"], snap_points)
        stream_notes, stream_sr = _rebalance_lns_for_target(stream_notes, target, tolerance, config, analysis, snap_points, accent_snap_points)
        stream_sr = DifficultyEstimator.estimate_sr(stream_notes, analysis["duration_ms"])
        return stream_notes, stream_sr, abs(stream_sr - target) <= tolerance, 1

    if config.chart_type in ["rice", "ln"] and config.key_style == "speed":
        speed_notes = _shape_speed_profile([], config, analysis, snap_points, accent_snap_points, tolerance)
        speed_notes = _refine_lns_to_sustain_and_hits(speed_notes, config, analysis, snap_points, accent_snap_points)
        speed_notes = Validator.validate_and_fix(speed_notes, config, analysis["silent_regions"], snap_points)
        speed_notes, speed_sr = _rebalance_lns_for_target(speed_notes, target, tolerance, config, analysis, snap_points, accent_snap_points)
        speed_sr = DifficultyEstimator.estimate_sr(speed_notes, analysis["duration_ms"])
        return speed_notes, speed_sr, abs(speed_sr - target) <= tolerance, 1

    if config.chart_type in ["rice", "ln"] and config.key_style == "tech":
        tech_notes = _shape_tech_profile([], config, analysis, snap_points, accent_snap_points, tolerance)
        tech_notes = _refine_lns_to_sustain_and_hits(tech_notes, config, analysis, snap_points, accent_snap_points)
        tech_notes = Validator.validate_and_fix(tech_notes, config, analysis["silent_regions"], snap_points)
        tech_notes, tech_sr = _rebalance_lns_for_target(tech_notes, target, tolerance, config, analysis, snap_points, accent_snap_points)
        tech_sr = DifficultyEstimator.estimate_sr(tech_notes, analysis["duration_ms"])
        return tech_notes, tech_sr, abs(tech_sr - target) <= tolerance, 1

    if config.chart_type == "hybrid":
        hybrid_notes = _shape_hybrid_profile([], config, analysis, snap_points, accent_snap_points, tolerance)
        hybrid_notes = _refine_lns_to_sustain_and_hits(hybrid_notes, config, analysis, snap_points, accent_snap_points)
        hybrid_notes = Validator.validate_and_fix(hybrid_notes, config, analysis["silent_regions"], snap_points)
        hybrid_notes, hybrid_sr = _rebalance_lns_for_target(hybrid_notes, target, tolerance, config, analysis, snap_points, accent_snap_points)
        hybrid_sr = DifficultyEstimator.estimate_sr(hybrid_notes, analysis["duration_ms"])
        return hybrid_notes, hybrid_sr, abs(hybrid_sr - target) <= tolerance, 1

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
    fixed = _shape_tech_profile(fixed, config, analysis, snap_points, accent_snap_points)
    fixed = _shape_hybrid_profile(fixed, config, analysis, snap_points, accent_snap_points)
    fixed = _apply_music_influence(fixed, config, analysis, snap_points, accent_snap_points, config.key_style)
    fixed = _refine_lns_to_sustain_and_hits(fixed, config, analysis, snap_points, accent_snap_points)
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

    for divisor in _stream_divisor_candidates(target, analysis["bpm"]):
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
                temperature=temperature,
                target=target,
            )
            candidate = _build_stream_notes(rows, sizes, target, temperature)
            candidate = _apply_safe_lns(candidate, config, analysis, snap_points, "stream")
            candidate = _apply_music_influence(candidate, config, analysis, snap_points, accent_snap_points, "stream")
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

        if divisor == 4 and best_diff <= 0.10:
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
    elif target < 5.25:
        desired_interval = 86.0
    elif target < 6.25:
        desired_interval = 74.0
    else:
        desired_interval = 66.0

    divisors = [4, 5, 6, 7, 8]
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
        triple_candidates = _stream_spaced_candidates(row_count, rows, accent_snap_points, target, for_triples=True)
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
        double_candidates = _stream_spaced_candidates(row_count, rows, accent_snap_points, target, for_triples=False)
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


def _stream_triple_ratio(target: float, temperature: float) -> float:
    if target < 4.75:
        base = 0.0
    elif target < 5.5:
        base = 0.08 + (target - 4.75) * 0.08
    elif target < 6.5:
        base = 0.14 + (target - 5.5) * 0.07
    else:
        base = 0.22 + min(0.08, (target - 6.5) * 0.05)
    return max(0.0, min(0.32, base * (0.85 + temperature * 0.30)))


def _stream_spaced_candidates(
    row_count: int,
    rows: List[int],
    accent_snap_points: Set[int],
    target: float,
    for_triples: bool,
) -> List[int]:
    if row_count <= 0:
        return []

    if for_triples:
        groups = [
            [index for index in range(1, row_count - 1) if rows[index] in accent_snap_points and index % 2 == 0],
            [index for index in range(1, row_count - 1) if rows[index] in accent_snap_points],
            [index for index in range(1, row_count - 1) if index % 4 == 0],
            [index for index in range(1, row_count - 1) if index % 2 == 0],
        ]
    else:
        groups = [
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
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
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
    if target < 5.5:
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

    if target < 5.75:
        chord_weight = 8.0
        interval_weight = 0.45
    else:
        chord_weight = 12.0
        interval_weight = 0.05

    chord_penalty = max(0.0, average_chord_size - 1.0) * chord_weight
    interval_penalty = abs(interval - desired_interval) * interval_weight
    return chord_penalty + interval_penalty


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


def _build_speed_notes(
    rows: List[int],
    sizes: List[int],
    target: float,
    temperature: float,
) -> List[NoteObject]:
    notes: List[NoteObject] = []
    previous_lanes: List[int] = []
    recent_lanes: List[List[int]] = []
    phrase_order = _speed_phrase_order(0, temperature)
    phrase_len = _speed_phrase_length(target, temperature, 0)
    phrase_start = 0
    phrase_index = 0

    for row_index, (time_ms, size) in enumerate(zip(rows, sizes)):
        if row_index >= phrase_start + phrase_len:
            phrase_index += 1
            phrase_start = row_index
            phrase_len = _speed_phrase_length(target, temperature, phrase_index)
            phrase_order = _speed_phrase_order(phrase_index, temperature)

        local_index = row_index - phrase_start
        base_lane = phrase_order[local_index % len(phrase_order)]
        lanes = _next_speed_lanes(
            size,
            base_lane,
            previous_lanes,
            recent_lanes,
            row_index,
            phrase_index,
            target,
            temperature,
        )
        previous_lanes = lanes
        recent_lanes.append(lanes)
        if len(recent_lanes) > 8:
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


def _shape_tech_profile(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    tolerance: float = 0.15,
) -> List[NoteObject]:
    if config.chart_type not in ["rice", "ln"] or config.key_style != "tech" or config.target_star is None:
        return notes

    target = config.target_star
    temperature = _pattern_temperature(config)
    max_chord_size = min(4, max(1, config.max_chord_size))
    max_profile_chord_size = min(max_chord_size, _tech_profile_max_chord_size(target))

    best_notes = notes
    best_sr = DifficultyEstimator.estimate_sr(notes, analysis["duration_ms"]) if notes else 0.0
    best_diff = abs(best_sr - target)
    best_within_tolerance = best_diff <= tolerance
    best_style_score = float("inf")

    density_center = _tech_burst_density_center(target, config.chart_type)
    density_lower = 0.18 if target < 4.75 else 0.34 if target < 5.5 else 0.58
    density_upper = 1.70 if target < 6.5 else 4.50
    density_values = _tech_search_values(density_center, density_lower, density_upper, 0.10)
    chord_center = _tech_burst_chord_center(target, config.chart_type)
    chord_upper = _tech_profile_average_upper(target, max_profile_chord_size, config.chart_type)
    chord_values = _tech_search_values(chord_center, 1.0, chord_upper, 0.08)

    for density_bias in density_values:
        for chord_bias in chord_values:
            candidate = _build_tech_burst_cell_notes(
                analysis,
                snap_points,
                accent_snap_points,
                target,
                temperature,
                density_bias,
                chord_bias,
                max_profile_chord_size,
                config.chart_type,
            )
            candidate = _soften_tech_fast_jacks(candidate, target, temperature)
            candidate = _break_tech_speed_runs(candidate, target, max_profile_chord_size)
            candidate = _apply_safe_lns(candidate, config, analysis, snap_points, "tech")
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate = _ensure_tech_musical_continuity(candidate, config, analysis, snap_points, accent_snap_points)
            candidate = _apply_music_influence(candidate, config, analysis, snap_points, accent_snap_points, "tech")
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate = _reduce_tech_short_jacks(candidate, target, max_profile_chord_size)
            candidate = _reinforce_tech_safe_chords(candidate, target, max_profile_chord_size)
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate = _reduce_tech_short_jacks(candidate, target, max_profile_chord_size)
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate = _reduce_tech_short_jacks(candidate, target, max_profile_chord_size)
            candidate = _reduce_tech_short_jacks(candidate, target, max_profile_chord_size)
            candidate = _reinforce_tech_safe_chords(candidate, target, max_profile_chord_size)
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate = _reduce_tech_short_jacks(candidate, target, max_profile_chord_size)
            candidate = _apply_music_influence(candidate, config, analysis, snap_points, accent_snap_points, "tech")
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate = _apply_safe_lns(candidate, config, analysis, snap_points, "tech")
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate = _reduce_tech_short_jacks(candidate, target, max_profile_chord_size)
            candidate = _prioritize_tech_music_anchors(candidate, config, analysis, snap_points, accent_snap_points, max_profile_chord_size)
            candidate = _reduce_tech_short_jacks(candidate, target, max_profile_chord_size)
            if config.chart_type == "ln":
                candidate = _apply_safe_lns(candidate, config, analysis, snap_points, "tech")
                candidate = _align_rows_to_music_anchors(candidate, config, analysis, snap_points, accent_snap_points, "tech", _music_influence(config))
            candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
            candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
            diff = abs(candidate_sr - target)
            style_score = _tech_burst_style_score(candidate, target)
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
            elif not best_within_tolerance and (diff < best_diff or (abs(diff - best_diff) <= 0.03 and style_score < best_style_score)):
                best_notes = candidate
                best_sr = candidate_sr
                best_diff = diff
                best_style_score = style_score

            if best_within_tolerance and best_style_score <= _tech_accept_style_score(target, config.chart_type):
                break

        if best_within_tolerance and best_style_score <= _tech_accept_style_score(target, config.chart_type):
            break

    if best_notes:
        repaired = _ensure_tech_musical_continuity(best_notes, config, analysis, snap_points, accent_snap_points)
        repaired = _reduce_tech_short_jacks(repaired, target, max_profile_chord_size)
        repaired = _prioritize_tech_music_anchors(repaired, config, analysis, snap_points, accent_snap_points, max_profile_chord_size)
        repaired = _reduce_tech_short_jacks(repaired, target, max_profile_chord_size)
        if config.chart_type == "ln":
            repaired = _apply_safe_lns(repaired, config, analysis, snap_points, "tech")
            repaired = _align_rows_to_music_anchors(repaired, config, analysis, snap_points, accent_snap_points, "tech", _music_influence(config))
        repaired = Validator.validate_and_fix(repaired, config, analysis["silent_regions"], snap_points)
        repaired_sr = DifficultyEstimator.estimate_sr(repaired, analysis["duration_ms"])
        repaired_diff = abs(repaired_sr - target)
        if repaired_diff <= tolerance or repaired_diff <= best_diff + 0.08:
            best_notes = repaired

    return best_notes


def _tech_accept_style_score(target: float, chart_type: str) -> float:
    base = 1.85 if target < 6.25 else 2.15
    if chart_type == "ln":
        base += 0.35
    return base


def _shape_hybrid_profile(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    tolerance: float = 0.15,
) -> List[NoteObject]:
    if config.chart_type != "hybrid" or config.target_star is None:
        return notes

    target = config.target_star
    if target < 3.25:
        config = replace(
            config,
            max_chord_size=1,
            ln_ratio=min(config.ln_ratio, 0.18),
            hybrid_weights={"jack": 0.04, "stream": 0.46, "tech": 0.24, "speed": 0.26},
            music_influence=min(config.music_influence, 0.55),
            pattern_temperature=min(config.pattern_temperature, 0.10),
        )
    temperature = _pattern_temperature(config)
    max_chord_size = clamp_max_chord_size(config)
    effective_ln = _hybrid_effective_ln_ratio(config, target)
    density_center = _hybrid_density_center(target)
    chord_center = _hybrid_chord_center(target)

    best_notes = notes
    best_sr = DifficultyEstimator.estimate_sr(notes, analysis["duration_ms"]) if notes else 0.0
    best_diff = abs(best_sr - target)
    best_within_tolerance = best_diff <= tolerance
    best_style_score = float("inf")

    # Hybrid candidates are expensive because each one runs section materializing,
    # music-fit repair, legality validation, and SR estimation. Use a feedback walk
    # instead of a full density/chord/LN grid so CLI testing stays practical.
    pending: List[Tuple[float, float, float]] = []
    seen: Set[Tuple[float, float, float]] = set()

    def queue_candidate(density_bias: float, chord_bias: float, ln_bias: float) -> None:
        density_bias = round(max(0.20, min(1.95, density_bias)), 3)
        chord_bias = round(max(0.88, min(float(max_chord_size), chord_bias)), 3)
        ln_bias = round(max(0.12, min(0.82, ln_bias)), 3)
        key = (density_bias, chord_bias, ln_bias)
        if key not in seen:
            seen.add(key)
            pending.append(key)

    low_ln_offset = 0.14 if target < 4.5 else 0.06
    if target < 3.25:
        initial_candidates = [
            (0.42, 0.92, 0.12),
            (0.56, 0.96, 0.12),
            (0.70, 1.00, 0.12),
            (0.84, 1.04, 0.12),
            (1.00, 1.08, 0.12),
        ]
    elif target < 3.75:
        initial_candidates = [
            (0.36, 1.08, 0.12),
            (0.42, 1.14, 0.14),
            (0.48, 1.22, 0.16),
            (0.54, 1.32, 0.18),
            (density_center - 0.40, chord_center + 0.10, 0.18),
            (density_center - 0.34, chord_center + 0.20, 0.20),
        ]
    elif target < 4.5:
        initial_candidates = [
            # Low-star hybrid should gain weight from accent chords and LN phrasing,
            # not from same-lane mini-jacks.
            (0.50, float(max_chord_size), 0.25),
            (0.56, float(max_chord_size) - 0.15, 0.34),
            (density_center - 0.38, chord_center + 0.44, 0.16),
            (density_center - 0.34, chord_center + 0.34, 0.22),
            (density_center - 0.48, chord_center + 0.20, effective_ln - low_ln_offset),
            (density_center - 0.58, chord_center + 0.34, effective_ln - low_ln_offset - 0.02),
            (density_center - 0.60, chord_center - 0.48, effective_ln - 0.06),
            (density_center - 0.36, chord_center - 0.20, effective_ln - 0.03),
            (density_center, chord_center, effective_ln),
        ]
    elif target < 6.0:
        initial_candidates = [
            (density_center - 0.60, chord_center - 0.48, effective_ln - 0.06),
            (density_center - 0.46, chord_center - 0.34, 0.45),
            (density_center - 0.46, chord_center - 0.34, 0.55),
            (density_center - 0.48, chord_center + 0.20, effective_ln - low_ln_offset),
            (density_center - 0.36, chord_center - 0.20, effective_ln - 0.03),
            (density_center - 0.18, chord_center - 0.08, effective_ln),
            (density_center, chord_center, effective_ln),
            (density_center + 0.18, chord_center + 0.08, effective_ln),
        ]
    else:
        initial_candidates = [
            (density_center - 0.70, chord_center + 0.38, 0.45),
            (density_center - 0.66, chord_center + 0.50, 0.45),
            (density_center - 0.18, chord_center - 0.08, effective_ln),
            (density_center, chord_center, effective_ln),
            (density_center + 0.18, chord_center + 0.08, effective_ln),
            (density_center - 0.36, chord_center - 0.20, effective_ln - 0.03),
            (density_center - 0.60, chord_center - 0.48, effective_ln - 0.06),
        ]
        if target >= 6.8:
            initial_candidates = [
                (density_center + 0.34, min(float(max_chord_size), chord_center + 0.52), effective_ln),
                (density_center + 0.24, min(float(max_chord_size), chord_center + 0.70), effective_ln - 0.04),
                (density_center + 0.12, min(float(max_chord_size), chord_center + 0.42), effective_ln + 0.04),
            ] + initial_candidates

    for density_bias, chord_bias, ln_bias in initial_candidates:
        queue_candidate(density_bias, chord_bias, ln_bias)

    current_density = density_center
    current_chord = chord_center
    current_ln = effective_ln
    max_candidates = 10
    evaluated = 0

    while pending and evaluated < max_candidates:
        density_bias, chord_bias, ln_bias = pending.pop(0)
        candidate, sections = _build_hybrid_section_notes(
            config,
            analysis,
            snap_points,
            accent_snap_points,
            density_bias,
            chord_bias,
            ln_bias,
        )
        candidate = _repair_hybrid_continuity(candidate, sections, config, analysis, snap_points, accent_snap_points)
        candidate = _apply_music_influence(candidate, config, analysis, snap_points, accent_snap_points, None)
        candidate = _reinforce_hybrid_safe_chords(candidate, config, analysis, snap_points, accent_snap_points, target)
        candidate = _reduce_hybrid_short_jacks(candidate, target, clamp_max_chord_size(config))
        candidate = _lift_hybrid_ln_texture(candidate, config, analysis, snap_points, accent_snap_points, target)
        candidate = _refine_lns_to_sustain_and_hits(candidate, config, analysis, snap_points, accent_snap_points)
        candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
        candidate = _repair_hybrid_continuity(candidate, sections, config, analysis, snap_points, accent_snap_points)
        candidate = _reinforce_hybrid_safe_chords(candidate, config, analysis, snap_points, accent_snap_points, target)
        candidate = _reduce_hybrid_short_jacks(candidate, target, clamp_max_chord_size(config))
        candidate = _lift_hybrid_ln_texture(candidate, config, analysis, snap_points, accent_snap_points, target)
        candidate = _refine_lns_to_sustain_and_hits(candidate, config, analysis, snap_points, accent_snap_points)
        candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
        candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
        diff = abs(candidate_sr - target)
        style_score = _hybrid_style_score(candidate, sections, target)
        evaluated += 1

        if diff <= tolerance:
            if (
                not best_within_tolerance
                or style_score < best_style_score
                or (abs(style_score - best_style_score) <= 0.001 and diff < best_diff)
            ):
                best_notes = candidate
                best_sr = candidate_sr
                best_diff = diff
                best_style_score = style_score
                best_within_tolerance = True
            if style_score <= 1.35:
                break
        elif not best_within_tolerance and (diff < best_diff or (abs(diff - best_diff) <= 0.04 and style_score < best_style_score)):
            best_notes = candidate
            best_sr = candidate_sr
            best_diff = diff
            best_style_score = style_score

        direction = 1.0 if candidate_sr < target else -1.0
        sr_gap = abs(candidate_sr - target) / max(1.0, target)
        step = max(0.08, min(0.24, 0.06 + sr_gap * 0.72))
        current_density = max(0.20, min(1.95, density_bias + direction * step))
        current_chord = max(0.88, min(float(max_chord_size), chord_bias + direction * step * 0.46))
        if config.ln_ratio > 0.12:
            current_ln = max(0.18, min(0.82, ln_bias + direction * 0.025))

        queue_candidate(current_density, current_chord, current_ln)
        if evaluated in [2, 5]:
            queue_candidate(density_center + direction * step * 1.7, chord_center, effective_ln)
            queue_candidate(density_center, chord_center + direction * step * 0.9, current_ln)

    return best_notes


def _hybrid_density_center(target: float) -> float:
    if target < 3.5:
        return 0.72
    if target < 4.5:
        return 0.88
    if target < 5.5:
        return 1.04
    if target < 6.5:
        return 1.22
    return 1.42


def _hybrid_chord_center(target: float) -> float:
    if target < 3.5:
        return 1.12
    if target < 4.5:
        return 1.22
    if target < 5.5:
        return 1.36
    if target < 6.5:
        return 1.54
    return 1.78


def _hybrid_effective_ln_ratio(config: DifficultyConfig, target: float) -> float:
    floor = 0.38 + max(0.0, min(1.0, (target - 3.0) / 4.0)) * 0.18
    requested = max(0.0, min(1.0, config.ln_ratio))
    if requested <= 0.12:
        requested = floor
    else:
        requested = max(requested, 0.38 + requested * 0.62)
    return max(0.18, min(0.76, max(requested, floor)))


def _hybrid_search_values(center: float, lower: float, upper: float, step: float) -> List[float]:
    values: List[float] = []
    for offset in [0.0, -0.14, 0.14, -0.28, 0.28]:
        value = round(max(lower, min(upper, center + offset)), 3)
        if value not in values:
            values.append(value)
    value = round(lower, 3)
    while value <= upper + 0.0001:
        rounded = round(value, 3)
        if abs(rounded - center) <= step * 1.1 and rounded not in values:
            values.append(rounded)
        value += step
    return values


def _build_hybrid_section_notes(
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    density_bias: float,
    chord_bias: float,
    ln_bias: float,
) -> Tuple[List[NoteObject], List[Dict[str, Any]]]:
    if not snap_points:
        return [], []

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    target = config.target_star or 4.0
    temperature = _pattern_temperature(config)
    context = _music_context(analysis, snap_points, accent_snap_points)
    weights = normalize_hybrid_weights(config.hybrid_weights)
    sections = _hybrid_sections(analysis, snap_points, context, beat_length, target, temperature)
    notes: List[NoteObject] = []
    previous_module = ""
    previous_run = 0

    for index, section in enumerate(sections):
        module = _choose_hybrid_module(
            section,
            weights,
            ln_bias,
            target,
            temperature,
            previous_module,
            previous_run,
            index,
        )
        section["module"] = module
        previous_run = previous_run + 1 if module == previous_module else 1
        previous_module = module

        rows = _hybrid_section_rows(
            analysis,
            config,
            snap_points,
            accent_snap_points,
            section,
            module,
            target,
            density_bias,
        )
        section["rows"] = rows
        section_notes = _materialize_hybrid_section(
            rows,
            module,
            config,
            analysis,
            snap_points,
            accent_snap_points,
            section,
            density_bias,
            chord_bias,
            ln_bias,
        )
        notes.extend(section_notes)

    return sorted(notes, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1)), sections


def _hybrid_sections(
    analysis: Dict[str, Any],
    snap_points: List[int],
    context: Dict[int, Dict[str, float]],
    beat_length: float,
    target: float,
    temperature: float,
) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    duration_ms = int(analysis["duration_ms"])
    start = float(max(analysis["offset_ms"], snap_points[0]))
    index = 0
    while start < duration_ms:
        local_profile = _hybrid_section_profile(context, int(round(start)), int(round(start + beat_length * 8.0)))
        beats = _hybrid_section_beats(local_profile, target, temperature, index)
        end = min(float(duration_ms), start + beat_length * beats)
        profile = _hybrid_section_profile(context, int(round(start)), int(round(end)))
        sections.append(
            {
                "index": index,
                "start": int(round(start)),
                "end": int(round(end)),
                "beats": beats,
                "profile": profile,
            }
        )
        start = end
        index += 1
    return sections


def _hybrid_section_profile(
    context: Dict[int, Dict[str, float]],
    start: int,
    end: int,
) -> Dict[str, float]:
    entries = [entry for time_ms, entry in context.items() if start <= time_ms < end]
    if not entries:
        return {
            "score": 0.0,
            "energy": 0.0,
            "accent": 0.0,
            "kick": 0.0,
            "onset": 0.0,
            "stack": 0.0,
            "silent": 1.0,
            "onset_density": 0.0,
            "accent_density": 0.0,
        }

    count = len(entries)
    return {
        "score": sum(entry["score"] for entry in entries) / count,
        "energy": sum(entry["energy"] for entry in entries) / count,
        "accent": max(entry["accent"] for entry in entries),
        "kick": max(entry["kick"] for entry in entries),
        "onset": max(entry["onset"] for entry in entries),
        "stack": max(entry["stack"] for entry in entries),
        "silent": sum(entry["silent"] for entry in entries) / count,
        "onset_density": sum(1 for entry in entries if entry["onset"] >= 0.12) / count,
        "accent_density": sum(1 for entry in entries if entry["accent"] >= 0.45) / count,
    }


def _hybrid_section_beats(profile: Dict[str, float], target: float, temperature: float, index: int) -> int:
    if profile["score"] < 0.10 and profile["accent"] < 0.22:
        base = 8
    elif target >= 6.0 or profile["accent_density"] >= 0.16:
        base = 8
    else:
        base = 12
    if temperature >= 0.65:
        base += _deterministic_jitter(index, int(target * 313), 2)
    return max(6, min(16, base))


def _choose_hybrid_module(
    section: Dict[str, Any],
    weights: Dict[str, float],
    ln_bias: float,
    target: float,
    temperature: float,
    previous_module: str,
    previous_run: int,
    index: int,
) -> str:
    profile = section["profile"]
    if profile["silent"] >= 0.70:
        return "break_transition"
    if profile["score"] < 0.09 and profile["accent"] < 0.24:
        return "break_transition"

    non_ln = max(0.0, 1.0 - ln_bias * 0.78)
    high_music = profile["accent_density"] * 1.15 + profile["kick"] * 0.18 + profile["accent"] * 0.12
    fast_music = profile["onset_density"] * 1.15 + profile["energy"] * 0.34 + profile["onset"] * 0.10
    sustain = max(0.0, profile["energy"] * 0.65 + profile["score"] * 0.35 - profile["onset_density"] * 0.18)

    def preferred_ln_module() -> str:
        if (
            weights["tech"] > 0
            and (profile["accent_density"] >= 0.045 or profile["kick"] >= 0.64 or profile["accent"] >= 0.62)
        ):
            return "ln_chord_burst"
        if weights["speed"] >= weights["stream"] and (profile["onset_density"] >= 0.030 or profile["energy"] >= 0.42):
            return "ln_speed_anchor"
        return "ln_stream"

    # The pp/hybrid references lean on LN skeleton phrases, then swap into
    # speed/stream/tech bursts. Force that phrasing rhythm so one loud peak does
    # not turn the whole map into speed or tech.
    phrase_phase = index % 8
    if ln_bias >= 0.42 and previous_run < 2 and (weights["stream"] + weights["speed"] + weights["tech"]) > 0:
        if index == 0 or phrase_phase in [0, 3, 6]:
            return preferred_ln_module()
        if phrase_phase == 2 and (profile["accent_density"] >= 0.035 or profile["kick"] >= 0.58):
            return "ln_chord_burst"

    scores = {
        "ln_stream": weights["stream"] * ln_bias * (0.65 + sustain + profile["score"] * 0.35),
        "ln_speed_anchor": weights["speed"] * ln_bias * (0.45 + fast_music + profile["onset"] * 0.35),
        "ln_chord_burst": weights["tech"] * ln_bias * (0.35 + high_music * 1.35),
        "stream_rice": weights["stream"] * non_ln * (0.55 + fast_music * 0.65 + profile["score"] * 0.35),
        "speed_rice": weights["speed"] * non_ln * (0.45 + fast_music * 0.90),
        "tech_chord_burst": weights["tech"] * (0.36 + high_music * 1.10 + max(0.0, target - 4.5) * 0.05),
        "jack_anchor": weights["jack"] * (0.20 + profile["kick"] * 0.45 + profile["accent_density"] * 0.35),
    }

    if target < 4.5:
        scores["tech_chord_burst"] *= 0.72
        scores["jack_anchor"] *= 0.70
    if previous_run >= 2 and previous_module in scores:
        scores[previous_module] *= 0.36
    if previous_module == "tech_chord_burst":
        scores["tech_chord_burst"] *= 0.30
    if previous_module == "speed_rice":
        scores["speed_rice"] *= 0.58
    if previous_module == "jack_anchor":
        scores["jack_anchor"] *= 0.35
    if profile["accent_density"] < 0.035 and profile["onset_density"] < 0.045:
        scores["tech_chord_burst"] *= 0.42
    if profile["accent"] >= 0.68 or profile["kick"] >= 0.62:
        scores["ln_chord_burst"] *= 1.16
        scores["tech_chord_burst"] *= 1.04

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1] - (_deterministic_jitter(index, int(target * 100), 7) / 100.0 * temperature),
            item[0],
        ),
    )
    return ranked[0][0]


def _hybrid_section_rows(
    analysis: Dict[str, Any],
    config: DifficultyConfig,
    snap_points: List[int],
    accent_snap_points: Set[int],
    section: Dict[str, Any],
    module: str,
    target: float,
    density_bias: float,
) -> List[int]:
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    preferred_divisor = _hybrid_module_divisor(module, target, analysis["bpm"], density_bias, section)
    divisor = _nearest_allowed_divisor(preferred_divisor, config.allowed_subdivisions)
    start = int(section["start"])
    end = int(section["end"])
    silent_regions = analysis.get("silent_regions", [])
    rows: List[int] = []

    if divisor <= 0:
        return rows

    step = beat_length / divisor
    desired = float(start)
    while desired < end:
        nearest = _nearest_snap_point(snap_points, int(round(desired)))
        if (
            nearest is not None
            and start <= nearest < end
            and abs(nearest - desired) <= max(10.0, beat_length / 32.0)
            and not _time_in_regions(nearest, silent_regions)
            and nearest not in rows
        ):
            rows.append(nearest)
        desired += step

    strong_anchors = [
        time_ms
        for time_ms in accent_snap_points
        if start <= time_ms < end
        and not _time_in_regions(time_ms, silent_regions)
    ]
    if module not in ["break_transition", "speed_rice"]:
        for time_ms in strong_anchors:
            if time_ms not in rows:
                rows.append(time_ms)

    rows = sorted(set(rows))
    if module == "break_transition":
        keep_every = 4 if section["profile"]["score"] < 0.08 else 3
        rows = [time_ms for idx, time_ms in enumerate(rows) if idx % keep_every == 0 or time_ms in accent_snap_points]
    elif density_bias < 0.50:
        keep_every = 3 if module.startswith("ln_") else 4
        rows = [
            time_ms
            for idx, time_ms in enumerate(rows)
            if time_ms in accent_snap_points or idx % keep_every == 0
        ]
    elif density_bias < 0.56:
        keep_every = 2 if module.startswith("ln_") else 3
        rows = [
            time_ms
            for idx, time_ms in enumerate(rows)
            if time_ms in accent_snap_points or (idx % keep_every == 0 and idx % 10 != 2)
        ]
    elif density_bias < 0.68:
        keep_every = 2 if module.startswith("ln_") else 3
        rows = [
            time_ms
            for idx, time_ms in enumerate(rows)
            if time_ms in accent_snap_points
            or (idx % keep_every == 0 and idx % (8 if module.startswith("ln_") else 12) != keep_every)
        ]
    elif density_bias < 0.76:
        if module.startswith("ln_"):
            rows = [
                time_ms
                for idx, time_ms in enumerate(rows)
                if time_ms in accent_snap_points or (idx % 2 == 0 and idx % 10 != 2) or idx % 14 == 1
            ]
        else:
            rows = [
                time_ms
                for idx, time_ms in enumerate(rows)
                if time_ms in accent_snap_points or (idx % 3 == 0 and idx % 12 != 3) or idx % 14 == 1
            ]
    elif density_bias < 0.85:
        rows = [
            time_ms
            for idx, time_ms in enumerate(rows)
            if time_ms in accent_snap_points or idx % 2 == 0 or idx % (8 if module.startswith("ln_") else 10) == 3
        ]

    rows = _shape_hybrid_burst_rows(
        rows,
        module,
        divisor,
        section,
        analysis,
        accent_snap_points,
        beat_length,
        target,
    )

    if not rows and section["profile"]["silent"] < 0.70:
        fallback = _hybrid_best_section_time(snap_points, accent_snap_points, analysis, start, end)
        if fallback is not None:
            rows = [fallback]
    return rows


def _shape_hybrid_burst_rows(
    rows: List[int],
    module: str,
    divisor: int,
    section: Dict[str, Any],
    analysis: Dict[str, Any],
    accent_snap_points: Set[int],
    beat_length: float,
    target: float,
) -> List[int]:
    if divisor < 9 or len(rows) < 6:
        return rows
    if module not in ["speed_rice", "tech_chord_burst", "ln_chord_burst", "ln_speed_anchor"]:
        return rows

    start = int(section["start"])
    end = int(section["end"])
    anchors = [time_ms for time_ms in rows if time_ms in accent_snap_points]
    if not anchors and section["profile"]["accent"] >= 0.40:
        anchors = [rows[len(rows) // 2]]
    if not anchors:
        return rows[:: max(2, divisor // 3)]

    max_windows = 2 if target < 6.2 else 3
    ranked_anchors = sorted(
        anchors,
        key=lambda time_ms: (
            -_music_entry(analysis, [], accent_snap_points, time_ms)["score"],
            abs(time_ms - (start + end) // 2),
        ),
    )[:max_windows]
    half_window = beat_length * (0.32 if target < 6.2 else 0.46)
    skeleton_every = max(3, divisor // 3)
    if module.startswith("ln_"):
        skeleton_every = max(2, divisor // 4)

    shaped: List[int] = []
    for idx, time_ms in enumerate(rows):
        in_burst = any(abs(time_ms - anchor) <= half_window for anchor in ranked_anchors)
        if in_burst or time_ms in accent_snap_points or idx % skeleton_every == 0:
            shaped.append(time_ms)

    return sorted(set(shaped))


def _hybrid_module_divisor(
    module: str,
    target: float,
    bpm: float,
    density_bias: float,
    section: Dict[str, Any],
) -> int:
    profile = section["profile"]
    if module == "break_transition":
        return 2 if profile["score"] < 0.12 else 4
    if module == "jack_anchor":
        return 4 if target < 6.25 else 6
    if module == "stream_rice":
        return _stream_divisor_candidates(target, bpm)[0]
    if module == "speed_rice":
        return _speed_divisor_candidates(target + (0.35 if density_bias >= 1.1 else 0.0), bpm)[0]
    if module == "tech_chord_burst":
        return _tech_divisor_candidates(target, bpm)[0]
    if module == "ln_speed_anchor":
        return _speed_divisor_candidates(target + 0.25, bpm)[0]
    if module == "ln_chord_burst":
        if profile["accent"] >= 0.60 or density_bias >= 1.15:
            return _tech_divisor_candidates(target, bpm)[0]
        return 6
    if module == "ln_stream":
        if target >= 6.0 or density_bias >= 1.15:
            return _stream_divisor_candidates(target, bpm)[0]
        return 4
    return 4


def _materialize_hybrid_section(
    rows: List[int],
    module: str,
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    section: Dict[str, Any],
    density_bias: float,
    chord_bias: float,
    ln_bias: float,
) -> List[NoteObject]:
    if not rows:
        return []

    target = config.target_star or 4.0
    temperature = _pattern_temperature(config)
    max_chord_size = clamp_max_chord_size(config)
    average_chord_size = _hybrid_module_chord_average(module, target, chord_bias, section["profile"], max_chord_size)

    if module == "break_transition":
        return _build_hybrid_break_notes(rows, section, target)
    if module == "speed_rice":
        local_max = min(max_chord_size, _speed_profile_max_chord_size(target))
        sizes = _build_speed_chord_sizes(len(rows), average_chord_size, local_max, rows, accent_snap_points, target, temperature)
        return _build_speed_notes(rows, sizes, target, temperature)
    if module == "stream_rice":
        local_max = min(max_chord_size, 3)
        sizes = _build_stream_chord_sizes(len(rows), average_chord_size, 1, local_max, rows, accent_snap_points, temperature, target)
        return _build_stream_notes(rows, sizes, target, temperature)
    if module == "tech_chord_burst":
        local_max = min(max_chord_size, _tech_profile_max_chord_size(target))
        sizes = _build_tech_chord_sizes(len(rows), average_chord_size, local_max, rows, accent_snap_points, target, temperature)
        notes = _build_tech_notes(rows, sizes, target, temperature)
        notes = _soften_tech_fast_jacks(notes, target, temperature)
        return _reduce_tech_short_jacks(notes, target, local_max)
    if module == "jack_anchor":
        local_max = min(max_chord_size, 4)
        min_size = 1 if target < 5.75 else 2
        sizes = _build_jack_chord_sizes(len(rows), average_chord_size, min_size, local_max, rows, accent_snap_points, analysis, target, temperature)
        return _build_jack_stack_notes(rows, sizes, target, temperature)

    if module == "ln_stream":
        return _build_hybrid_ln_phrase_notes(
            rows, module, config, analysis, snap_points, accent_snap_points, section, average_chord_size, max(ln_bias, 0.52)
        )
    if module == "ln_speed_anchor":
        return _build_hybrid_ln_phrase_notes(
            rows, module, config, analysis, snap_points, accent_snap_points, section, average_chord_size, max(ln_bias, 0.56)
        )
    if module == "ln_chord_burst":
        return _build_hybrid_ln_phrase_notes(
            rows, module, config, analysis, snap_points, accent_snap_points, section, average_chord_size, max(ln_bias, 0.58)
        )

    return _build_hybrid_break_notes(rows, section, target)


def _hybrid_module_chord_average(
    module: str,
    target: float,
    chord_bias: float,
    profile: Dict[str, float],
    max_chord_size: int,
) -> float:
    pressure = profile["accent"] * 0.42 + profile["kick"] * 0.30 + profile["accent_density"] * 0.28
    if module == "break_transition":
        value = 1.0 + pressure * 0.12
    elif module == "speed_rice":
        value = 1.03 + chord_bias * 0.10 + pressure * 0.14
    elif module == "stream_rice":
        value = 1.12 + chord_bias * 0.20 + pressure * 0.20
    elif module == "tech_chord_burst":
        value = 1.18 + chord_bias * 0.26 + pressure * 0.36
    elif module == "jack_anchor":
        value = 1.22 + chord_bias * 0.32 + pressure * 0.30
    elif module == "ln_stream":
        value = 1.08 + chord_bias * 0.16 + pressure * 0.18
    elif module == "ln_speed_anchor":
        value = 1.04 + chord_bias * 0.12 + pressure * 0.14
    elif module == "ln_chord_burst":
        value = 1.24 + chord_bias * 0.28 + pressure * 0.38
    else:
        value = 1.0
    if target < 4.5:
        value -= 0.12
    elif target >= 6.25:
        value += 0.10
    return max(1.0, min(float(max_chord_size), value))


def _build_hybrid_break_notes(rows: List[int], section: Dict[str, Any], target: float) -> List[NoteObject]:
    notes: List[NoteObject] = []
    lane_order = [1, 2, 0, 3]
    for index, time_ms in enumerate(rows):
        lane = lane_order[(index + section["index"]) % len(lane_order)]
        notes.append(NoteObject(time_ms=time_ms, lane=lane))
        if section["profile"]["accent"] >= 0.50 and target >= 5.0 and index % 4 == 0:
            partner = 3 - lane
            if partner != lane:
                notes.append(NoteObject(time_ms=time_ms, lane=partner))
    return notes


def _apply_hybrid_lns(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    style: str,
    ln_ratio: float,
) -> List[NoteObject]:
    if not notes:
        return notes
    local_max = clamp_max_chord_size(config, style)
    ln_config = replace(
        config,
        chart_type="ln",
        key_style=style,  # type: ignore[arg-type]
        ln_ratio=max(0.0, min(1.0, ln_ratio)),
        max_chord_size=min(config.max_chord_size, local_max),
    )
    return _apply_safe_lns(notes, ln_config, analysis, snap_points, style)


def _build_hybrid_ln_phrase_notes(
    rows: List[int],
    module: str,
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    section: Dict[str, Any],
    average_chord_size: float,
    ln_bias: float,
) -> List[NoteObject]:
    if not rows:
        return []

    rows = sorted(set(rows))
    target = config.target_star or 4.0
    temperature = _pattern_temperature(config)
    max_chord_size = clamp_max_chord_size(config)
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    gaps = [b - a for a, b in zip(rows, rows[1:]) if b > a]
    row_step = sorted(gaps)[len(gaps) // 2] if gaps else max(1, int(round(beat_length / 4.0)))
    rows_per_beat = max(1, int(round(beat_length / max(1, row_step))))
    context = _music_context(analysis, snap_points, accent_snap_points)

    if module == "ln_stream":
        lane_cycle = [0, 2, 1, 3]
        base_period = rows_per_beat * (2 if target < 5.2 else 1)
        base_length = rows_per_beat * (2 if target < 6.2 else 1)
    elif module == "ln_speed_anchor":
        lane_cycle = [0, 3, 1, 2]
        base_period = rows_per_beat * (3 if target < 5.2 else 2)
        base_length = rows_per_beat * (3 if target < 5.8 else 2)
    else:
        lane_cycle = [0, 3, 2, 1]
        base_period = rows_per_beat * (2 if target < 5.8 else 1)
        base_length = rows_per_beat * (1 if target >= 6.3 else 2)

    base_period = max(2, base_period)
    base_length = max(2, base_length)
    max_length_rows = max(1, int(round(config.max_ln_ms / max(1, row_step))))
    min_length_rows = max(1, int(math.ceil(config.min_ln_ms / max(1, row_step))))
    hold_until = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    last_tail = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    notes: List[NoteObject] = []
    previous_rice_lanes: Set[int] = set()

    for index, time_ms in enumerate(rows):
        active_holds = {lane for lane, tail in hold_until.items() if time_ms < tail}
        row_lanes: Set[int] = set()
        entry = context.get(time_ms) or _music_entry(analysis, snap_points, accent_snap_points, time_ms)
        is_accent = time_ms in accent_snap_points or entry["accent"] >= 0.52 or entry["kick"] >= 0.58

        if _should_start_hybrid_ln_anchor(index, time_ms, module, base_period, ln_bias, is_accent, target, temperature):
            ln_count = _hybrid_ln_head_count(module, target, is_accent, average_chord_size, max_chord_size)
            for _ in range(ln_count):
                lane = _pick_hybrid_ln_lane(time_ms, index, lane_cycle, row_lanes, active_holds, last_tail)
                if lane is None:
                    break
                length_rows = _hybrid_anchor_length_rows(
                    index,
                    lane,
                    module,
                    base_length,
                    min_length_rows,
                    max_length_rows,
                    target,
                    temperature,
                    is_accent,
                )
                end_index = min(len(rows) - 1, index + length_rows)
                if end_index <= index:
                    continue
                end_time = rows[end_index]
                if end_time - time_ms < config.min_ln_ms:
                    continue
                end_time = min(end_time, time_ms + config.max_ln_ms)
                notes.append(NoteObject(time_ms=time_ms, lane=lane, end_time_ms=end_time))
                row_lanes.add(lane)
                hold_until[lane] = end_time
                last_tail[lane] = end_time
                active_holds.add(lane)

        desired_size = _hybrid_ln_phrase_row_size(
            module,
            average_chord_size,
            target,
            is_accent,
            entry,
            index,
            temperature,
            max_chord_size,
        )
        while len(row_lanes) < desired_size:
            lane = _pick_hybrid_rice_lane(
                time_ms,
                index,
                module,
                row_lanes,
                active_holds,
                previous_rice_lanes,
                entry,
                target,
            )
            if lane is None:
                break
            notes.append(NoteObject(time_ms=time_ms, lane=lane))
            row_lanes.add(lane)
            previous_rice_lanes = {lane}

        if not row_lanes and not active_holds:
            lane = _pick_hybrid_rice_lane(time_ms, index, module, row_lanes, active_holds, previous_rice_lanes, entry, target)
            if lane is not None:
                notes.append(NoteObject(time_ms=time_ms, lane=lane))
                previous_rice_lanes = {lane}

    return sorted(notes, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _should_start_hybrid_ln_anchor(
    row_index: int,
    time_ms: int,
    module: str,
    period_rows: int,
    ln_bias: float,
    is_accent: bool,
    target: float,
    temperature: float,
) -> bool:
    if period_rows <= 0:
        return False
    phase = _deterministic_index(period_rows, int(target * 100), len(module), 0xA17)
    periodic = row_index % period_rows == phase
    if is_accent and module in ["ln_chord_burst", "ln_speed_anchor"]:
        periodic = periodic or row_index % max(2, period_rows // 2) == phase % max(2, period_rows // 2)
    if not periodic and not (is_accent and ln_bias >= 0.58 and row_index % max(2, period_rows // 2) == 0):
        return False
    threshold = int(max(0.12, min(0.92, ln_bias + (0.10 if is_accent else -0.08) + temperature * 0.06)) * 1000)
    roll = _deterministic_index(1000, row_index, time_ms, int(target * 100), len(module))
    return roll < threshold


def _hybrid_ln_head_count(
    module: str,
    target: float,
    is_accent: bool,
    average_chord_size: float,
    max_chord_size: int,
) -> int:
    if max_chord_size <= 1:
        return 1
    if module == "ln_chord_burst" and is_accent and target >= 5.0 and average_chord_size >= 1.35:
        return min(2, max_chord_size)
    if module == "ln_speed_anchor" and is_accent and target >= 6.2 and average_chord_size >= 1.45:
        return min(2, max_chord_size)
    return 1


def _pick_hybrid_ln_lane(
    time_ms: int,
    row_index: int,
    lane_cycle: List[int],
    row_lanes: Set[int],
    active_holds: Set[int],
    last_tail: Dict[int, int],
) -> Optional[int]:
    candidates = [
        lane
        for lane in lane_cycle
        if lane not in row_lanes
        and lane not in active_holds
        and time_ms - last_tail.get(lane, -999999) > 45
    ]
    if not candidates:
        return None
    return candidates[(row_index + _deterministic_index(len(candidates), time_ms, 0x71B)) % len(candidates)]


def _hybrid_anchor_length_rows(
    row_index: int,
    lane: int,
    module: str,
    base_length: int,
    min_length_rows: int,
    max_length_rows: int,
    target: float,
    temperature: float,
    is_accent: bool,
) -> int:
    choices = [base_length, base_length + 1, max(min_length_rows, base_length - 1)]
    if module == "ln_speed_anchor":
        choices.extend([base_length * 2, base_length + 2])
    elif module == "ln_chord_burst":
        choices.extend([max(min_length_rows, base_length // 2), base_length + (1 if is_accent else 0)])
    else:
        choices.append(base_length * 2 if target < 5.5 else base_length + 2)
    index = _deterministic_index(len(choices), row_index, lane, int(target * 100), int(temperature * 1000))
    return max(min_length_rows, min(max_length_rows, choices[index]))


def _hybrid_ln_phrase_row_size(
    module: str,
    average_chord_size: float,
    target: float,
    is_accent: bool,
    entry: Dict[str, float],
    row_index: int,
    temperature: float,
    max_chord_size: int,
) -> int:
    desired = 1
    pressure = max(entry["accent"], entry["kick"] * 0.95, entry["score"] * 1.35)
    extra_chance = max(0.0, min(0.95, average_chord_size - 1.0 + pressure * 0.28))
    roll = _deterministic_index(1000, row_index, int(target * 100), int(pressure * 1000), len(module))
    if is_accent or roll < int(extra_chance * 1000):
        desired += 1
    if module == "ln_chord_burst" and target >= 5.6 and is_accent and average_chord_size >= 1.50:
        desired += 1
    if module == "ln_speed_anchor" and not is_accent:
        desired = min(desired, 1 if target < 5.8 else 2)
    if target < 4.5 and not is_accent:
        desired = 1
    return max(1, min(max_chord_size, desired))


def _pick_hybrid_rice_lane(
    time_ms: int,
    row_index: int,
    module: str,
    row_lanes: Set[int],
    active_holds: Set[int],
    previous_rice_lanes: Set[int],
    entry: Dict[str, float],
    target: float,
) -> Optional[int]:
    available = [lane for lane in [0, 1, 2, 3] if lane not in row_lanes and lane not in active_holds]
    if not available:
        return None
    if module == "ln_speed_anchor":
        order = [1, 2, 0, 3] if row_index % 2 == 0 else [2, 1, 3, 0]
    elif module == "ln_chord_burst":
        order = [0, 3, 1, 2] if entry["kick"] >= 0.55 else [1, 2, 0, 3]
    else:
        order = [0, 1, 2, 3] if (row_index // 2) % 2 == 0 else [3, 2, 1, 0]
    ranked = sorted(
        available,
        key=lambda lane: (
            lane in previous_rice_lanes and target < 6.5,
            order.index(lane) if lane in order else 99,
            _deterministic_index(101, time_ms, row_index, lane, len(module)),
        ),
    )
    return ranked[0]


def _hybrid_best_section_time(
    snap_points: List[int],
    accent_snap_points: Set[int],
    analysis: Dict[str, Any],
    start: int,
    end: int,
) -> Optional[int]:
    candidates = [
        time_ms
        for time_ms in accent_snap_points
        if start <= time_ms < end and not _time_in_regions(time_ms, analysis.get("silent_regions", []))
    ]
    if not candidates:
        candidates = [
            time_ms
            for time_ms in snap_points
            if start <= time_ms < end and not _time_in_regions(time_ms, analysis.get("silent_regions", []))
        ]
    if not candidates:
        return None
    midpoint = (start + end) // 2
    return sorted(
        candidates,
        key=lambda time_ms: (
            -_music_entry(analysis, snap_points, accent_snap_points, time_ms)["accent"],
            abs(time_ms - midpoint),
            time_ms,
        ),
    )[0]


def _repair_hybrid_continuity(
    notes: List[NoteObject],
    sections: List[Dict[str, Any]],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    rows: Dict[int, Set[int]] = {}
    ln_rows: Dict[Tuple[int, int], Optional[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)
        ln_rows[(note.time_ms, note.lane)] = note.end_time_ms

    lane_cycle = [1, 2, 0, 3]
    for section in sections:
        if section["profile"]["silent"] >= 0.70:
            continue
        section_times = [time_ms for time_ms in rows if section["start"] <= time_ms < section["end"]]
        if section_times:
            continue
        fallback = _hybrid_best_section_time(snap_points, accent_snap_points, analysis, section["start"], section["end"])
        if fallback is None:
            continue
        lane = lane_cycle[section["index"] % len(lane_cycle)]
        rows.setdefault(fallback, set()).add(lane)

    repaired = [
        NoteObject(time_ms=time_ms, lane=lane, end_time_ms=ln_rows.get((time_ms, lane)))
        for time_ms in sorted(rows)
        for lane in sorted(rows[time_ms])
    ]
    repaired = sorted(repaired, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))
    return _fill_reasonable_gaps(repaired, analysis, snap_points, accent_snap_points, config)


def _hybrid_style_score(notes: List[NoteObject], sections: List[Dict[str, Any]], target: float) -> float:
    if not notes:
        return 999.0
    rows: Dict[int, Set[int]] = {}
    ln_count = 0
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)
        if note.is_ln:
            ln_count += 1
    module_names = [str(section.get("module", "")) for section in sections]
    module_counts = Counter(module_names)
    ln_ratio = ln_count / max(1, len(notes))
    desired_ln = 0.42 + max(0.0, min(1.0, (target - 3.0) / 4.0)) * 0.12
    diversity_penalty = max(0.0, 4 - len([name for name, count in module_counts.items() if count > 0])) * 0.35
    long_empty_sections = sum(
        1
        for section in sections
        if section.get("profile", {}).get("silent", 0.0) < 0.70
        and not any(section["start"] <= time_ms < section["end"] for time_ms in rows)
    )
    chord_rows = sum(1 for lane_set in rows.values() if len(lane_set) >= 2)
    chord_ratio = chord_rows / max(1, len(rows))
    chord_target = 0.22 + max(0.0, min(1.0, (target - 4.0) / 3.0)) * 0.16
    short_jack_ratio = _hybrid_short_jack_ratio(notes, target)
    return (
        abs(ln_ratio - desired_ln) * 2.8
        + abs(chord_ratio - chord_target) * 1.4
        + short_jack_ratio * 4.2
        + diversity_penalty
        + long_empty_sections * 4.0
    )


def _lift_hybrid_ln_texture(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    target: float,
) -> List[NoteObject]:
    if not notes or config.ln_ratio <= 0.12:
        return notes
    if target < 3.25:
        return notes

    current_ln = sum(1 for note in notes if note.is_ln)
    effective_ln = _hybrid_effective_ln_ratio(config, target)
    if target < 4.5:
        desired_ratio = min(0.05, max(0.03, effective_ln * 0.07))
    elif target < 5.0:
        desired_ratio = min(0.18, max(0.10, effective_ln * 0.20))
    elif target < 6.2:
        desired_ratio = min(0.36, max(0.22, effective_ln * 0.44))
    elif target >= 6.8:
        desired_ratio = min(0.18, max(0.12, effective_ln * 0.18))
    else:
        desired_ratio = min(0.22, max(0.14, effective_ln * 0.20))
    desired_ln = int(round(len(notes) * desired_ratio))
    if current_ln >= desired_ln:
        return notes

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    min_len = max(80, int(config.min_ln_ms))
    max_len = max(min_len, int(min(config.max_ln_ms, beat_length * (2.0 if target < 5.8 else 1.45))))
    tail_gap = 42

    by_lane: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for note in notes:
        by_lane[note.lane].append(note.time_ms)
    next_same: Dict[Tuple[int, int], Optional[int]] = {}
    for lane, lane_times in by_lane.items():
        ordered = sorted(set(lane_times))
        for idx, time_ms in enumerate(ordered):
            next_same[(time_ms, lane)] = ordered[idx + 1] if idx + 1 < len(ordered) else None

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)

    context = _music_context(analysis, snap_points, accent_snap_points)
    lane_tail = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    lifted: List[NoteObject] = []
    ln_count = current_ln

    for row_index, time_ms in enumerate(sorted(rows)):
        entry = context.get(time_ms) or _music_entry(analysis, snap_points, accent_snap_points, time_ms)
        row_notes = sorted(rows[time_ms], key=lambda note: (note.lane, note.end_time_ms or -1))
        row_has_ln = any(note.is_ln for note in row_notes)
        row_is_accent = time_ms in accent_snap_points or entry["accent"] >= 0.50 or entry["kick"] >= 0.56
        phase = _deterministic_index(5, int(target * 100), int(config.ln_ratio * 100), row_index)
        periodic = (row_index + phase) % (4 if target < 5.8 else 3) == 0
        row_lifted = False

        for note in row_notes:
            copied = NoteObject(time_ms=note.time_ms, lane=note.lane, end_time_ms=note.end_time_ms)
            if (
                ln_count < desired_ln
                and not copied.is_ln
                and not row_has_ln
                and not row_lifted
                and time_ms > lane_tail[copied.lane] + tail_gap
                and (periodic or row_is_accent or entry["energy"] >= 0.52)
            ):
                next_time = next_same.get((time_ms, copied.lane))
                ideal_len = int(beat_length * (1.0 if target < 5.8 else 0.75))
                if row_is_accent and target >= 5.2:
                    ideal_len = int(beat_length * 1.15)
                ideal_len = max(min_len, min(max_len, ideal_len))
                latest_tail = time_ms + max_len
                if next_time is not None:
                    latest_tail = min(latest_tail, next_time - tail_gap)
                end_time = min(time_ms + ideal_len, latest_tail)
                if end_time - time_ms >= min_len:
                    copied.end_time_ms = end_time
                    lane_tail[copied.lane] = end_time
                    row_lifted = True
                    ln_count += 1

            if copied.is_ln and copied.end_time_ms is not None:
                lane_tail[copied.lane] = max(lane_tail[copied.lane], copied.end_time_ms)
            lifted.append(copied)

    return sorted(lifted, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _reduce_hybrid_short_jacks(notes: List[NoteObject], target: float, max_chord_size: int) -> List[NoteObject]:
    if not notes:
        return notes

    guard_ms = _hybrid_short_jack_guard_ms(target)
    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)

    last_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    lane_block_until = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    fixed: List[NoteObject] = []
    last_output_time = -999999

    for row_index, time_ms in enumerate(sorted(rows)):
        original = sorted(
            rows[time_ms],
            key=lambda note: (
                not note.is_ln,
                note.lane,
                note.end_time_ms or 0,
            ),
        )[:max_chord_size]
        used: Set[int] = set()
        row_out: List[NoteObject] = []

        for note in original:
            lane = _hybrid_reassigned_lane(
                note.lane,
                time_ms,
                used,
                last_time,
                lane_block_until,
                guard_ms,
                row_index,
                target,
            )
            if lane is None:
                # Do not keep extra notes that can only be placed as a naked short jack.
                # If the row would become empty, only keep a fallback when the row gap
                # itself is already outside the hybrid short-jack guard. Otherwise the
                # map reaches SR by repeated same-lane taps instead of hybrid texture.
                if row_out:
                    continue
                if time_ms - last_output_time <= guard_ms:
                    continue
                lane = _least_bad_hybrid_lane(time_ms, used, last_time, lane_block_until)
                if lane is None:
                    continue

            copied = NoteObject(time_ms=time_ms, lane=lane, end_time_ms=note.end_time_ms)
            row_out.append(copied)
            used.add(lane)

        for note in sorted(row_out, key=lambda item: item.lane):
            fixed.append(note)
            last_time[note.lane] = time_ms
            lane_block_until[note.lane] = note.end_time_ms if note.is_ln and note.end_time_ms is not None else time_ms
        if row_out:
            last_output_time = time_ms

    return fixed


def _reinforce_hybrid_safe_chords(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    target: float,
) -> List[NoteObject]:
    if not notes:
        return notes

    max_chord_size = clamp_max_chord_size(config)
    if max_chord_size <= 1:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)

    context = _music_context(analysis, snap_points, accent_snap_points)
    guard_ms = _hybrid_short_jack_guard_ms(target)
    last_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    lane_block_until = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    reinforced: List[NoteObject] = []

    for row_index, time_ms in enumerate(sorted(rows)):
        row_notes = sorted(rows[time_ms], key=lambda note: (note.lane, note.end_time_ms or -1))[:max_chord_size]
        row_lanes = {note.lane for note in row_notes}
        entry = context.get(time_ms) or _music_entry(analysis, snap_points, accent_snap_points, time_ms)
        pressure = max(entry["accent"], entry["kick"] * 0.95, entry["score"] * 1.55)
        desired = len(row_lanes)

        chord_threshold = 0.62 if target < 3.5 else 0.54 if target < 4.75 else 0.45
        if pressure >= chord_threshold or (time_ms in accent_snap_points and target >= 4.0):
            desired = max(desired, 2)
        if target >= 5.25 and (pressure >= 0.58 or row_index % 8 == 0):
            desired = max(desired, 3)
        if 5.75 <= target < 6.25 and (pressure >= 0.52 or row_index % 6 == 0):
            desired = max(desired, min(3, max_chord_size))
        if 5.75 <= target < 6.25 and pressure >= 0.62:
            desired = max(desired, min(4, max_chord_size))
        if 4.75 <= target < 5.25 and pressure >= 0.62:
            desired = max(desired, min(3, max_chord_size))
        if target >= 6.8 and pressure >= 0.70:
            desired = max(desired, min(4, max_chord_size))
        if target >= 6.8:
            if pressure >= 0.56 or time_ms in accent_snap_points:
                desired = max(desired, min(4, max_chord_size))
            elif row_index % 6 == 0:
                desired = max(desired, min(3, max_chord_size))

        desired = min(max_chord_size, desired)
        while len(row_lanes) < desired:
            lane = _hybrid_safe_chord_lane(time_ms, row_lanes, last_time, lane_block_until, guard_ms, entry, row_index)
            if lane is None:
                break
            row_notes.append(NoteObject(time_ms=time_ms, lane=lane))
            row_lanes.add(lane)

        for note in sorted(row_notes, key=lambda item: (item.lane, item.end_time_ms or -1)):
            reinforced.append(note)
            last_time[note.lane] = time_ms
            lane_block_until[note.lane] = note.end_time_ms if note.is_ln and note.end_time_ms is not None else time_ms

    return reinforced


def _hybrid_safe_chord_lane(
    time_ms: int,
    row_lanes: Set[int],
    last_time: Dict[int, int],
    lane_block_until: Dict[int, int],
    guard_ms: int,
    entry: Dict[str, float],
    row_index: int,
) -> Optional[int]:
    candidates = [
        lane
        for lane in [0, 1, 2, 3]
        if lane not in row_lanes
        and time_ms > lane_block_until.get(lane, -999999)
        and time_ms - last_time.get(lane, -999999) >= guard_ms
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda lane: (
            -abs(lane - 1.5) if entry["kick"] >= 0.62 else abs(lane - 1.5),
            -(time_ms - last_time.get(lane, -999999)),
            _deterministic_index(113, time_ms, row_index, lane, int(entry["accent"] * 1000)),
        ),
    )[0]


def _hybrid_reassigned_lane(
    preferred_lane: int,
    time_ms: int,
    used: Set[int],
    last_time: Dict[int, int],
    lane_block_until: Dict[int, int],
    guard_ms: int,
    row_index: int,
    target: float,
) -> Optional[int]:
    if (
        preferred_lane not in used
        and time_ms > lane_block_until.get(preferred_lane, -999999)
        and time_ms - last_time.get(preferred_lane, -999999) >= guard_ms
    ):
        return preferred_lane

    candidates = [
        lane
        for lane in [0, 1, 2, 3]
        if lane not in used
        and time_ms > lane_block_until.get(lane, -999999)
        and time_ms - last_time.get(lane, -999999) >= guard_ms
    ]
    if candidates:
        return sorted(
            candidates,
            key=lambda lane: (
                -(time_ms - last_time.get(lane, -999999)),
                abs(lane - 1.5),
                _deterministic_index(97, time_ms, row_index, lane, int(target * 100)),
            ),
        )[0]

    return None


def _least_bad_hybrid_lane(
    time_ms: int,
    used: Set[int],
    last_time: Dict[int, int],
    lane_block_until: Dict[int, int],
) -> Optional[int]:
    candidates = [
        lane
        for lane in [0, 1, 2, 3]
        if lane not in used and time_ms > lane_block_until.get(lane, -999999)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda lane: (
            -(time_ms - last_time.get(lane, -999999)),
            abs(lane - 1.5),
            lane,
        ),
    )[0]


def _hybrid_short_jack_guard_ms(target: float) -> int:
    if target < 4.5:
        return 145
    if target < 5.75:
        return 130
    return 105


def _hybrid_short_jack_ratio(notes: List[NoteObject], target: float) -> float:
    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    lane_times: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for note in notes:
        lane_times[note.lane].append(note.time_ms)

    guard_ms = _hybrid_short_jack_guard_ms(target)
    short_hits = 0
    for lane, times in lane_times.items():
        ordered = sorted(times)
        for prev_time, next_time in zip(ordered, ordered[1:]):
            if next_time - prev_time > guard_ms:
                continue
            if any(
                start_time <= prev_time
                and candidate_lane == lane
                and end_time is not None
                and end_time > next_time
                for start_time, row_notes in rows.items()
                for candidate in row_notes
                for candidate_lane, end_time in [(candidate.lane, candidate.end_time_ms)]
            ):
                continue
            short_hits += 1
    return short_hits / max(1, len(rows))


def _tech_burst_density_center(target: float, chart_type: str) -> float:
    if target < 3.5:
        center = 0.24
    elif target < 4.5:
        center = 0.34
    elif target < 5.5:
        center = 0.58
    elif target < 6.5:
        center = 1.12
    else:
        center = 2.35
    if chart_type == "ln":
        center *= 0.92
    return center


def _tech_burst_chord_center(target: float, chart_type: str) -> float:
    if target < 3.5:
        center = 1.02
    elif target < 4.5:
        center = 1.08
    elif target < 5.5:
        center = 1.20
    elif target < 6.5:
        center = 1.52
    else:
        center = 2.55
    if chart_type == "ln":
        center -= 0.10
    return max(1.0, center)


def _tech_search_values(center: float, lower: float, upper: float, step: float) -> List[float]:
    values: List[float] = []
    for offset in [0.0, -0.20, 0.20, -0.36, 0.36, -0.52, 0.52, 0.76]:
        value = round(max(lower, min(upper, center + offset)), 3)
        if value not in values:
            values.append(value)
    value = round(lower, 3)
    while value <= upper + 0.0001:
        rounded = round(value, 3)
        if abs(rounded - center) <= step * 1.2 and rounded not in values:
            values.append(rounded)
        value += step
    return values


def _build_tech_burst_cell_notes(
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    target: float,
    temperature: float,
    density_bias: float,
    chord_bias: float,
    max_chord_size: int,
    chart_type: str,
) -> List[NoteObject]:
    if not snap_points:
        return []

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    duration_ms = int(analysis["duration_ms"])
    silent_regions = analysis.get("silent_regions", [])
    snap_points = sorted(snap_points)
    snap_set = set(snap_points)
    row_lanes: Dict[int, Set[int]] = {}
    phrase_index = 0
    desired_time = float(max(analysis["offset_ms"], snap_points[0]))
    last_cell_name = ""
    last_focus = ""

    while desired_time < duration_ms:
        snapped_start = _nearest_snap_point(snap_points, int(round(desired_time)))
        if snapped_start is None:
            break
        if snapped_start < desired_time - max(4.0, beat_length / 96.0):
            idx = bisect.bisect_right(snap_points, snapped_start)
            if idx >= len(snap_points):
                break
            snapped_start = snap_points[idx]
        if _time_in_regions(snapped_start, silent_regions):
            desired_time = snapped_start + beat_length * 0.5
            continue

        energy = _energy_score_at(analysis, snapped_start)
        cell = _choose_tech_burst_cell(
            target,
            temperature,
            density_bias,
            energy,
            phrase_index,
            last_cell_name,
            last_focus,
        )
        transformed_lanes = _transform_tech_cell_lanes(cell["lanes"], phrase_index, target, temperature)
        cell_rows = _materialize_tech_cell(
            start_time=snapped_start,
            beat_length=beat_length,
            snap_points=snap_points,
            snap_set=snap_set,
            accent_snap_points=accent_snap_points,
            silent_regions=silent_regions,
            lanes=transformed_lanes,
            gaps=cell["gaps"],
            target=target,
            temperature=temperature,
            chord_bias=chord_bias,
            max_chord_size=max_chord_size,
            cell_name=str(cell["name"]),
            phrase_index=phrase_index,
        )
        for time_ms, lanes in cell_rows:
            existing = row_lanes.setdefault(time_ms, set())
            for lane in lanes:
                if len(existing) >= max_chord_size:
                    break
                existing.add(lane)

        cell_span = _tech_cell_span_ms(cell["gaps"], beat_length)
        release = _tech_cell_release_ms(cell, beat_length, target, density_bias, energy, chart_type)
        desired_time = snapped_start + cell_span + release
        last_cell_name = str(cell["name"])
        last_focus = str(cell.get("focus", ""))
        phrase_index += 1

    _anchor_tech_musical_hits(
        row_lanes,
        analysis,
        snap_points,
        accent_snap_points,
        target,
        temperature,
        chord_bias,
        max_chord_size,
        density_bias,
    )

    notes = [
        NoteObject(time_ms=time_ms, lane=lane)
        for time_ms in sorted(row_lanes)
        for lane in sorted(row_lanes[time_ms])
    ]
    return notes


def _anchor_tech_musical_hits(
    row_lanes: Dict[int, Set[int]],
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    target: float,
    temperature: float,
    chord_bias: float,
    max_chord_size: int,
    density_bias: float,
) -> None:
    if not row_lanes or not snap_points:
        return

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    silent_regions = analysis.get("silent_regions", [])
    anchor_points = _tech_anchor_candidates(analysis, snap_points, accent_snap_points)
    if not anchor_points:
        return

    max_gap = _tech_anchor_gap_threshold_ms(beat_length, target, density_bias)
    guard = _tech_anchor_guard_ms(beat_length, target)
    row_times = sorted(row_lanes)

    for prev_time, next_time in zip(row_times, row_times[1:]):
        gap = next_time - prev_time
        if gap <= max_gap:
            continue

        inside = [
            time_ms
            for time_ms in anchor_points
            if prev_time + guard <= time_ms <= next_time - guard
            and not _time_in_regions(time_ms, silent_regions)
            and _energy_score_at(analysis, time_ms) >= _tech_anchor_energy_floor(target)
        ]
        if not inside:
            midpoint = (prev_time + next_time) // 2
            hard_gap = max(int(max_gap * 0.88), int(beat_length * 0.82))
            if gap > hard_gap or _energy_score_at(analysis, midpoint) >= _tech_anchor_energy_floor(target) + 0.10:
                nearest = _nearest_snap_point(snap_points, midpoint)
                if (
                    nearest is not None
                    and prev_time + guard <= nearest <= next_time - guard
                    and not _time_in_regions(nearest, silent_regions)
                ):
                    inside = [nearest]
        if not inside:
            continue

        needed = max(1, min(3, int(gap // max(1, max_gap)) ))
        selected = _select_tech_anchor_points(inside, needed, prev_time, next_time, target, temperature)
        for time_ms in selected:
            existing = row_lanes.setdefault(time_ms, set())
            if len(existing) >= max_chord_size:
                continue
            lanes = _tech_anchor_lanes(time_ms, target, chord_bias, max_chord_size, len(existing))
            for lane in lanes:
                if len(existing) >= max_chord_size:
                    break
                existing.add(lane)


def _tech_anchor_candidates(
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[int]:
    snap_set = set(snap_points)
    candidates: Set[int] = {time_ms for time_ms in accent_snap_points if time_ms in snap_set}
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    max_distance = max(35, min(70, int(beat_length * 0.12)))

    for source in ["beat_times_ms", "onset_times_ms"]:
        for raw_time in analysis.get(source, []):
            time_ms = int(raw_time)
            nearest = _nearest_snap_point(snap_points, time_ms)
            if nearest is not None and abs(nearest - time_ms) <= max_distance:
                candidates.add(nearest)

    return sorted(candidates)


def _tech_anchor_gap_threshold_ms(beat_length: float, target: float, density_bias: float) -> int:
    if target < 4.0:
        beats = 1.02
    elif target < 5.0:
        beats = 0.88
    elif target < 6.25:
        beats = 0.70
    else:
        beats = 0.56
    beats -= max(0.0, density_bias - 1.0) * 0.10
    return max(120, int(beat_length * beats))


def _tech_anchor_guard_ms(beat_length: float, target: float) -> int:
    if target < 4.0:
        return max(95, int(beat_length * 0.22))
    if target < 5.5:
        return max(70, int(beat_length * 0.16))
    return max(45, int(beat_length * 0.10))


def _tech_anchor_energy_floor(target: float) -> float:
    if target < 4.0:
        return 0.22
    if target < 5.5:
        return 0.18
    return 0.14


def _select_tech_anchor_points(
    values: List[int],
    count: int,
    prev_time: int,
    next_time: int,
    target: float,
    temperature: float,
) -> List[int]:
    if count >= len(values):
        return values
    selected: List[int] = []
    for index in range(1, count + 1):
        desired = prev_time + int(round((next_time - prev_time) * index / (count + 1)))
        ranked = sorted(
            values,
            key=lambda time_ms: (
                abs(time_ms - desired),
                _deterministic_index(127, time_ms, int(target * 100), int(temperature * 1000)),
            ),
        )
        for time_ms in ranked:
            if time_ms not in selected:
                selected.append(time_ms)
                break
    return sorted(selected)


def _tech_anchor_lanes(
    time_ms: int,
    target: float,
    chord_bias: float,
    max_chord_size: int,
    existing_size: int,
) -> List[int]:
    base_patterns = [[0, 3], [1, 2], [0, 1], [2, 3], [0, 2], [1, 3]]
    single_patterns = [[0], [3], [1], [2]]
    use_chord = target >= 4.0 or chord_bias >= 1.18
    if target < 4.0 and _deterministic_index(1000, time_ms, int(target * 100)) > 420:
        use_chord = False

    patterns = base_patterns if use_chord and max_chord_size - existing_size >= 2 else single_patterns
    choice = _deterministic_index(len(patterns), time_ms, int(target * 100), int(chord_bias * 1000))
    return list(patterns[choice])


def _tech_continuity_lanes(
    time_ms: int,
    rows: Dict[int, Set[int]],
    target: float,
    chord_bias: float,
    max_chord_size: int,
    existing_size: int,
    entry: Dict[str, float],
) -> List[int]:
    if existing_size >= max_chord_size:
        return []

    desired_size = 1
    if target >= 4.6 and (entry["accent"] >= 0.48 or entry["kick"] >= 0.48 or chord_bias >= 1.34):
        desired_size = 2
    if target >= 6.2 and max_chord_size >= 3 and (entry["accent"] >= 0.70 or chord_bias >= 1.70):
        desired_size = 3
    desired_size = min(max_chord_size, max(existing_size + 1, desired_size))

    row_times = sorted(rows)
    idx = bisect.bisect_left(row_times, time_ms)
    prev_time = row_times[idx - 1] if idx > 0 else None
    next_time = row_times[idx] if idx < len(row_times) else None
    prev_lanes = rows.get(prev_time, set()) if prev_time is not None else set()
    next_lanes = rows.get(next_time, set()) if next_time is not None else set()
    soft_window = _tech_short_jack_soft_window_ms(target)
    selected: Set[int] = set()

    while len(selected) + existing_size < desired_size:
        available = [lane for lane in [0, 1, 2, 3] if lane not in selected and lane not in rows.get(time_ms, set())]
        if not available:
            break
        ranked = sorted(
            available,
            key=lambda lane: (
                prev_time is not None and lane in prev_lanes and time_ms - prev_time <= soft_window,
                next_time is not None and lane in next_lanes and next_time - time_ms <= soft_window,
                lane in prev_lanes,
                lane in next_lanes,
                abs(lane - 1.5) if entry["kick"] < 0.62 else -abs(lane - 1.5),
                _deterministic_index(113, time_ms, lane, int(target * 100), len(selected)),
            ),
        )
        lane = ranked[0]
        selected.add(lane)

    return sorted(selected)


def _tech_cell_span_ms(gaps: List[int], beat_length: float) -> float:
    return sum(beat_length / max(1, gap) for gap in gaps)


def _choose_tech_burst_cell(
    target: float,
    temperature: float,
    density_bias: float,
    energy: float,
    phrase_index: int,
    last_cell_name: str,
    last_focus: str,
) -> Dict[str, Any]:
    cells = _tech_burst_cells()
    pressure = energy * 0.62 + min(1.0, max(0.0, (target - 3.0) / 4.0)) * 0.28 + (density_bias - 1.0) * 0.32
    pressure += 0.10 if phrase_index % 4 in [1, 2] else -0.06
    pressure += _deterministic_jitter(phrase_index, int(target * 1000), 8) / 100.0

    if target < 4.25 and pressure < 0.62:
        names = ["sparse_gate", "slow_cross", "stair_gate"]
    elif target < 5.0 and pressure < 0.46:
        names = ["sparse_gate", "slow_cross", "gate_walk", "cross_gate"]
    elif pressure < 0.34:
        names = ["gate_walk", "stair_gate", "cross_gate"]
    elif pressure < 0.56:
        names = ["gate_walk", "cross_gate", "left_hand_burst", "right_hand_burst", "lane_focus_burst"]
    elif pressure < 0.78:
        names = ["full_burst", "left_hand_burst", "right_hand_burst", "shape_burst", "lane_focus_burst", "cross_gate"]
    else:
        names = ["irregular_burst", "full_burst", "shape_burst", "left_hand_burst", "right_hand_burst", "lane_focus_burst"]

    if target >= 6.75:
        names = ["tenth_control_burst", "dense_gate_burst", "dense_split_burst", "lane_cycle_burst", "shape_burst", "irregular_burst"]
    elif target >= 6.25:
        names = list(dict.fromkeys(names + ["tenth_control_burst", "dense_gate_burst", "dense_split_burst", "lane_cycle_burst", "controlled_lane_burst", "irregular_burst", "shape_burst"]))
    if temperature >= 0.70:
        names = list(dict.fromkeys(names + ["pivot_burst", "split_burst"]))

    filtered = [
        name
        for name in names
        if name != last_cell_name and str(cells[name].get("focus", "")) != last_focus
    ]
    if not filtered:
        filtered = [name for name in names if name != last_cell_name] or names

    choice = _deterministic_index(len(filtered), phrase_index, int(target * 100), int(temperature * 1000), int(density_bias * 1000))
    return cells[filtered[choice]]


def _tech_burst_cells() -> Dict[str, Dict[str, Any]]:
    return {
        "sparse_gate": {
            "name": "sparse_gate",
            "focus": "full",
            "gaps": [4, 4, 2, 4, 4],
            "lanes": [[0], [1], [2], [0, 3], [3], [1]],
            "release": 0.85,
        },
        "slow_cross": {
            "name": "slow_cross",
            "focus": "full",
            "gaps": [4, 4, 4, 2],
            "lanes": [[0, 3], [1], [2], [0], [2, 3]],
            "release": 0.95,
        },
        "gate_walk": {
            "name": "gate_walk",
            "focus": "full",
            "gaps": [8, 8, 4, 8, 8, 4],
            "lanes": [[0, 1], [2], [3], [0, 3], [1], [2], [2, 3]],
            "release": 0.35,
        },
        "stair_gate": {
            "name": "stair_gate",
            "focus": "full",
            "gaps": [8, 8, 8, 8, 4],
            "lanes": [[0], [1], [2], [3], [0, 2], [1, 3]],
            "release": 0.45,
        },
        "cross_gate": {
            "name": "cross_gate",
            "focus": "full",
            "gaps": [8, 8, 8, 8, 8, 4],
            "lanes": [[0, 3], [1], [2], [0, 2], [3], [1, 2], [0]],
            "release": 0.25,
        },
        "full_burst": {
            "name": "full_burst",
            "focus": "full",
            "gaps": [12, 12, 8, 12, 12, 6, 8],
            "lanes": [[0, 1], [2], [3], [0], [1, 3], [2], [0, 2], [3]],
            "release": 0.18,
        },
        "left_hand_burst": {
            "name": "left_hand_burst",
            "focus": "left",
            "gaps": [12, 12, 8, 12, 12, 4],
            "lanes": [[0, 1], [2], [0], [1], [0, 3], [1], [2]],
            "release": 0.22,
        },
        "right_hand_burst": {
            "name": "right_hand_burst",
            "focus": "right",
            "gaps": [12, 12, 8, 12, 12, 4],
            "lanes": [[2, 3], [1], [3], [2], [0, 3], [2], [1]],
            "release": 0.22,
        },
        "lane_focus_burst": {
            "name": "lane_focus_burst",
            "focus": "lane",
            "gaps": [10, 8, 10, 8, 6, 8],
            "lanes": [[1], [3], [1], [0, 2], [1], [3], [0, 3]],
            "release": 0.20,
        },
        "shape_burst": {
            "name": "shape_burst",
            "focus": "shape",
            "gaps": [6, 6, 6, 6, 4],
            "lanes": [[0, 1], [2, 3], [0, 3], [1, 2], [0, 2], [1, 3]],
            "release": 0.16,
        },
        "irregular_burst": {
            "name": "irregular_burst",
            "focus": "rhythm",
            "gaps": [16, 16, 12, 8, 16, 16, 8, 6],
            "lanes": [[0, 1], [2], [3], [0], [1], [2, 3], [0], [2], [1, 3]],
            "release": 0.12,
        },
        "dense_gate_burst": {
            "name": "dense_gate_burst",
            "focus": "chord_gate",
            "gaps": [8, 8, 8, 8, 8, 8, 8],
            "lanes": [[0, 1], [2, 3], [0, 3], [1, 2], [0, 2], [1, 3], [0, 1], [2, 3]],
            "release": 0.04,
        },
        "dense_split_burst": {
            "name": "dense_split_burst",
            "focus": "split_gate",
            "gaps": [8, 8, 8, 8, 8, 8, 8],
            "lanes": [[0, 3], [1, 2], [0, 2], [1, 3], [0, 1], [2, 3], [0, 3], [1, 2]],
            "release": 0.04,
        },
        "controlled_lane_burst": {
            "name": "controlled_lane_burst",
            "focus": "lane_burst",
            "gaps": [5, 5, 5, 5, 5, 5, 5],
            "lanes": [[0], [0], [2, 3], [2], [1], [1], [0, 3], [3]],
            "release": 0.06,
        },
        "tenth_control_burst": {
            "name": "tenth_control_burst",
            "focus": "tenth_burst",
            "gaps": [10, 10, 10, 10, 10, 10, 10, 10, 10],
            "lanes": [[0], [1], [0], [2], [3], [2], [1, 3], [0], [1], [3]],
            "release": 0.04,
        },
        "lane_cycle_burst": {
            "name": "lane_cycle_burst",
            "focus": "cycle_burst",
            "gaps": [16, 16, 16, 16, 16, 16, 16, 16],
            "lanes": [[0], [1], [2], [3], [0], [1], [2], [3], [0, 3]],
            "release": 0.05,
        },
        "pivot_burst": {
            "name": "pivot_burst",
            "focus": "pivot",
            "gaps": [8, 12, 12, 8, 12, 12, 6],
            "lanes": [[0, 3], [1], [2], [1, 2], [3], [0], [2, 3], [1]],
            "release": 0.16,
        },
        "split_burst": {
            "name": "split_burst",
            "focus": "split",
            "gaps": [8, 8, 12, 12, 8, 8, 4],
            "lanes": [[0, 1], [3], [2], [2, 3], [0], [1], [0, 3], [2]],
            "release": 0.18,
        },
    }


def _transform_tech_cell_lanes(
    lanes: List[List[int]],
    phrase_index: int,
    target: float,
    temperature: float,
) -> List[List[int]]:
    mode = _deterministic_index(4, phrase_index, int(target * 100), int(temperature * 1000))
    transformed: List[List[int]] = []
    for row in lanes:
        new_row = []
        for lane in row:
            mapped = lane
            if mode == 1:
                mapped = 3 - lane
            elif mode == 2:
                mapped = (lane + 1) % 4
            elif mode == 3:
                mapped = [2, 0, 3, 1][lane]
            if mapped not in new_row:
                new_row.append(mapped)
        transformed.append(sorted(new_row))
    return transformed


def _materialize_tech_cell(
    start_time: int,
    beat_length: float,
    snap_points: List[int],
    snap_set: Set[int],
    accent_snap_points: Set[int],
    silent_regions: List[Tuple[int, int]],
    lanes: List[List[int]],
    gaps: List[int],
    target: float,
    temperature: float,
    chord_bias: float,
    max_chord_size: int,
    cell_name: str,
    phrase_index: int,
) -> List[Tuple[int, List[int]]]:
    rows: List[Tuple[int, List[int]]] = []
    desired = float(start_time)
    for local_index, base_lanes in enumerate(lanes):
        snapped = _nearest_snap_point(snap_points, int(round(desired)))
        if snapped is not None and snapped in snap_set and abs(snapped - desired) <= max(10.0, beat_length / 40.0):
            if not _time_in_regions(snapped, silent_regions):
                is_accent = snapped in accent_snap_points or local_index == 0 or local_index == len(lanes) - 1
                fitted = _fit_tech_cell_lanes(
                    base_lanes,
                    max_chord_size,
                    target,
                    temperature,
                    chord_bias,
                    is_accent,
                    cell_name,
                    phrase_index,
                    local_index,
                )
                if fitted:
                    rows.append((snapped, fitted))
        if local_index < len(gaps):
            desired += beat_length / max(1, gaps[local_index])
    return rows


def _fit_tech_cell_lanes(
    lanes: List[int],
    max_chord_size: int,
    target: float,
    temperature: float,
    chord_bias: float,
    is_accent: bool,
    cell_name: str,
    phrase_index: int,
    local_index: int,
) -> List[int]:
    chosen = []
    for lane in lanes:
        if lane in [0, 1, 2, 3] and lane not in chosen:
            chosen.append(lane)
    if not chosen:
        return []

    if len(chosen) > max_chord_size:
        chosen = chosen[:max_chord_size]

    if target < 4.75 and len(chosen) > 1 and not is_accent:
        keep = _deterministic_index(len(chosen), phrase_index, local_index, int(target * 100))
        chosen = [chosen[keep]]
    elif target < 4.0 and len(chosen) > 1 and is_accent and local_index not in [0]:
        keep = _deterministic_index(len(chosen), phrase_index, local_index, int(target * 211))
        chosen = [chosen[keep]]

    if len(chosen) >= max_chord_size:
        return sorted(chosen)

    chord_pressure = max(0.0, chord_bias - 1.0)
    if is_accent:
        chord_pressure += 0.22
    if "shape" in cell_name or "gate" in cell_name:
        chord_pressure += 0.10
    if target >= 6.25 and ("burst" in cell_name or "pivot" in cell_name):
        chord_pressure += 0.08
    chord_pressure *= 0.80 + max(0.0, min(1.0, temperature)) * 0.45

    roll = _deterministic_index(1000, phrase_index, local_index, len(chosen), int(target * 100), int(chord_bias * 1000))
    if len(chosen) == 1 and roll < int(min(0.92, chord_pressure) * 1000):
        partner = _tech_chord_partner(chosen[0], phrase_index, local_index, cell_name)
        if partner not in chosen:
            chosen.append(partner)

    triple_pressure = max(0.0, chord_bias - 1.55)
    if target >= 6.2 and is_accent:
        triple_pressure += 0.10
    triple_roll = _deterministic_index(1000, phrase_index, local_index, int(target * 137), int(chord_bias * 997))
    if max_chord_size >= 3 and len(chosen) == 2 and triple_roll < int(min(0.22, triple_pressure) * 1000):
        for lane in _tech_patterns(3)[_deterministic_index(len(_tech_patterns(3)), phrase_index, local_index, int(target * 100))]:
            if lane not in chosen:
                chosen.append(lane)
            if len(chosen) >= 3:
                break

    return sorted(chosen[:max_chord_size])


def _tech_chord_partner(lane: int, phrase_index: int, local_index: int, cell_name: str) -> int:
    if "left" in cell_name:
        preferred = {0: 1, 1: 0, 2: 0, 3: 1}
    elif "right" in cell_name:
        preferred = {0: 2, 1: 3, 2: 3, 3: 2}
    elif "lane" in cell_name:
        preferred = {0: 2, 1: 3, 2: 0, 3: 1}
    else:
        preferred = {0: 3, 1: 2, 2: 1, 3: 0}
    partner = preferred.get(lane, 3 - lane)
    if _deterministic_index(3, phrase_index, local_index, lane) == 0:
        alternatives = [candidate for candidate in [0, 1, 2, 3] if candidate != lane and candidate != partner]
        if alternatives:
            partner = alternatives[_deterministic_index(len(alternatives), phrase_index, local_index, lane, 77)]
    return partner


def _tech_cell_release_ms(
    cell: Dict[str, Any],
    beat_length: float,
    target: float,
    density_bias: float,
    energy: float,
    chart_type: str,
) -> float:
    release_beats = float(cell.get("release", 0.2))
    release_beats += max(0.0, 0.45 - energy) * 0.55
    release_beats += max(0.0, 4.9 - target) * 0.24
    release_beats += max(0.0, 1.0 - density_bias) * 0.85
    release_beats -= max(0.0, target - 5.0) * 0.035
    release_beats -= max(0.0, density_bias - 1.0) * 0.28
    if chart_type == "ln":
        release_beats += 0.06
    return max(0.0, release_beats) * beat_length


def _break_tech_speed_runs(notes: List[NoteObject], target: float, max_chord_size: int) -> List[NoteObject]:
    if max_chord_size < 2 or not notes:
        return notes

    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)
    times = sorted(rows)
    if len(times) < 8:
        return notes

    for start in range(0, len(times) - 7):
        window = times[start:start + 8]
        lane_rows = [rows[time_ms] for time_ms in window]
        if any(len(lanes) != 1 for lanes in lane_rows):
            continue
        gaps = [window[i] - window[i - 1] for i in range(1, len(window))]
        if max(gaps) - min(gaps) > 4:
            continue
        sequence = [next(iter(lanes)) for lanes in lane_rows]
        if _looks_like_speed_cycle(sequence):
            gate_index = start + 3 + _deterministic_index(2, start, int(target * 100))
            lane = sequence[gate_index - start]
            partner = _tech_chord_partner(lane, start, gate_index, "anti_speed")
            rows[times[gate_index]].add(partner)

    return [
        NoteObject(time_ms=time_ms, lane=lane)
        for time_ms in sorted(rows)
        for lane in sorted(rows[time_ms])
    ]


def _looks_like_speed_cycle(sequence: List[int]) -> bool:
    if len(sequence) < 8:
        return False
    asc = [0, 1, 2, 3]
    desc = [3, 2, 1, 0]
    common_cycles = [
        asc,
        desc,
        [0, 2, 1, 3],
        [1, 0, 3, 2],
        [1, 2, 3, 0],
        [2, 1, 0, 3],
    ]
    for cycle in common_cycles:
        for offset in range(len(cycle)):
            if all(sequence[index] == cycle[(index + offset) % len(cycle)] for index in range(len(sequence))):
                return True
    return len(set(sequence)) == 4 and all(sequence[index] != sequence[index - 1] for index in range(1, len(sequence)))


def _tech_burst_style_score(notes: List[NoteObject], target: float) -> float:
    if not notes:
        return 999.0
    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)
    times = sorted(rows)
    if len(times) < 2:
        return 999.0

    gaps = [times[index] - times[index - 1] for index in range(1, len(times))]
    fixed_gap_runs = 0
    current_run = 1
    for index in range(1, len(gaps)):
        if abs(gaps[index] - gaps[index - 1]) <= 3:
            current_run += 1
        else:
            if current_run >= 12:
                fixed_gap_runs += current_run
            current_run = 1
    if current_run >= 12:
        fixed_gap_runs += current_run

    single_rows = sum(1 for lanes in rows.values() if len(lanes) == 1)
    chord_rows = len(times) - single_rows
    speed_cycle_runs = 0
    singles = [(time_ms, next(iter(rows[time_ms]))) for time_ms in times if len(rows[time_ms]) == 1]
    for index in range(0, max(0, len(singles) - 7)):
        if _looks_like_speed_cycle([lane for _, lane in singles[index:index + 8]]):
            speed_cycle_runs += 1

    chord_ratio = chord_rows / max(1, len(times))
    desired_chord_ratio = 0.18 + max(0.0, min(1.0, (target - 4.0) / 3.0)) * 0.24
    chord_penalty = abs(chord_ratio - desired_chord_ratio) * 3.0

    short_jack_hits = 0
    near_jack_hits = 0
    last_lane_time = {0: None, 1: None, 2: None, 3: None}
    for time_ms in times:
        for lane in rows[time_ms]:
            previous = last_lane_time[lane]
            if previous is not None:
                gap = time_ms - previous
                if gap <= 75:
                    short_jack_hits += 1
                elif gap <= 90:
                    near_jack_hits += 1
            last_lane_time[lane] = time_ms

    return (
        fixed_gap_runs / max(1, len(times)) * 4.0
        + speed_cycle_runs / max(1, len(times)) * 8.0
        + chord_penalty
        + short_jack_hits / max(1, len(times)) * 160.0
        + near_jack_hits / max(1, len(times)) * 1.2
    )


def _ensure_tech_musical_continuity(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    if not notes or config.key_style != "tech" or not snap_points:
        return notes

    target = config.target_star or 4.0
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    threshold = _tech_final_gap_threshold_ms(beat_length, target)
    guard = _tech_anchor_guard_ms(beat_length, target)
    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)

    sorted_times = sorted(rows)
    if len(sorted_times) < 2:
        return notes

    snap_points = sorted(snap_points)
    snap_set = set(snap_points)
    silent_regions = analysis.get("silent_regions", [])
    anchor_points = _tech_anchor_candidates(analysis, snap_points, accent_snap_points)
    additions: List[NoteObject] = []
    chord_bias = _tech_burst_chord_center(target, config.chart_type)
    max_chord_size = max(1, min(4, config.max_chord_size))

    for prev_time, next_time in zip(sorted_times, sorted_times[1:]):
        gap = next_time - prev_time
        midpoint = (prev_time + next_time) // 2
        midpoint_entry = _music_entry(analysis, snap_points, accent_snap_points, midpoint)
        effective_threshold = _tech_effective_gap_threshold_ms(threshold, beat_length, target, midpoint_entry)
        if gap <= effective_threshold:
            continue
        if _time_in_regions(midpoint, silent_regions):
            continue

        needed = max(1, min(3, int(gap // max(1, effective_threshold))))
        candidates = [
            time_ms
            for time_ms in anchor_points
            if prev_time + guard <= time_ms <= next_time - guard
            and time_ms in snap_set
            and time_ms not in rows
            and not _time_in_regions(time_ms, silent_regions)
        ]
        selected = _select_tech_anchor_points(candidates, needed, prev_time, next_time, target, config.pattern_temperature)
        if len(selected) < needed:
            selected_set = set(selected)
            for index in range(1, needed + 1):
                desired = prev_time + int(round(gap * index / (needed + 1)))
                nearest = _nearest_snap_point(snap_points, desired)
                if (
                    nearest is not None
                    and prev_time + guard <= nearest <= next_time - guard
                    and nearest not in rows
                    and nearest not in selected_set
                    and not _time_in_regions(nearest, silent_regions)
                ):
                    selected.append(nearest)
                    selected_set.add(nearest)
        if len(selected) < needed and midpoint_entry["score"] >= 0.38:
            selected_set = set(selected)
            scored = sorted(
                [
                    time_ms
                    for time_ms in snap_points
                    if prev_time + guard <= time_ms <= next_time - guard
                    and time_ms not in rows
                    and time_ms not in selected_set
                    and not _time_in_regions(time_ms, silent_regions)
                ],
                key=lambda time_ms: (
                    -_music_entry(analysis, snap_points, accent_snap_points, time_ms)["score"],
                    abs(time_ms - midpoint),
                    time_ms,
                ),
            )
            for time_ms in scored:
                selected.append(time_ms)
                selected_set.add(time_ms)
                if len(selected) >= needed:
                    break

        for time_ms in sorted(selected):
            existing_size = len(rows.get(time_ms, set()))
            entry = _music_entry(analysis, snap_points, accent_snap_points, time_ms)
            lanes = _tech_continuity_lanes(time_ms, rows, target, chord_bias, max_chord_size, existing_size, entry)
            for lane in lanes:
                additions.append(NoteObject(time_ms=time_ms, lane=lane))
            rows.setdefault(time_ms, set()).update(lanes)

    if not additions:
        return notes
    return notes + additions


def _tech_final_gap_threshold_ms(beat_length: float, target: float) -> int:
    if target < 4.0:
        return max(300, int(beat_length * 0.88))
    if target < 5.0:
        return max(260, int(beat_length * 0.74))
    if target < 6.25:
        return max(220, int(beat_length * 0.58))
    return max(180, int(beat_length * 0.46))


def _tech_effective_gap_threshold_ms(
    base_threshold: int,
    beat_length: float,
    target: float,
    entry: Dict[str, float],
) -> int:
    if entry["score"] >= 0.58 or entry["accent"] >= 0.62:
        return max(105, min(base_threshold, int(beat_length * (0.34 if target < 6.25 else 0.28))))
    if entry["score"] >= 0.38 or entry["energy"] >= 0.52:
        return max(125, min(base_threshold, int(beat_length * (0.42 if target < 6.25 else 0.34))))
    if entry["score"] <= 0.10 and entry["energy"] <= 0.12:
        return int(base_threshold * 1.22)
    return base_threshold


def _reduce_tech_short_jacks(notes: List[NoteObject], target: float, max_chord_size: int) -> List[NoteObject]:
    if not notes:
        return notes

    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)

    hard_window = _tech_short_jack_hard_window_ms(target)
    soft_window = _tech_short_jack_soft_window_ms(target)
    last_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    last_row_lanes: Set[int] = set()
    fixed_rows: Dict[int, Set[int]] = {}
    last_output_time: Optional[int] = None

    for time_ms in sorted(rows):
        original = sorted(lane for lane in rows[time_ms] if lane in [0, 1, 2, 3])
        chosen: Set[int] = set()
        for lane in original:
            gap = time_ms - last_time.get(lane, -999999)
            repeated = gap <= hard_window or (gap <= soft_window and (len(original) >= 2 or lane in last_row_lanes))
            if repeated:
                replacement = _tech_short_jack_replacement(
                    chosen,
                    time_ms,
                    last_time,
                    hard_window,
                    soft_window,
                    original,
                )
                if replacement is not None:
                    lane = replacement
                else:
                    folded_lane = _fold_tech_short_jack_into_previous(
                        fixed_rows,
                        last_output_time,
                        lane,
                        max_chord_size,
                        last_time,
                        hard_window,
                    )
                    if folded_lane is not None:
                        if last_output_time is not None:
                            last_time[folded_lane] = last_output_time
                        continue
                    continue

            if lane in chosen:
                continue
            if len(chosen) >= max_chord_size:
                break
            chosen.add(lane)

        if not chosen:
            lane = _tech_short_jack_replacement(set(), time_ms, last_time, hard_window, soft_window, original)
            if lane is None:
                folded_lane = None
                for candidate in original:
                    folded_lane = _fold_tech_short_jack_into_previous(
                        fixed_rows,
                        last_output_time,
                        candidate,
                        max_chord_size,
                        last_time,
                        hard_window,
                    )
                    if folded_lane is not None:
                        break
                if folded_lane is None:
                    folded_lane = _fold_tech_short_jack_into_previous(
                        fixed_rows,
                        last_output_time,
                        None,
                        max_chord_size,
                        last_time,
                        hard_window,
                    )
                if folded_lane is not None:
                    if last_output_time is not None:
                        last_time[folded_lane] = last_output_time
                    continue
                continue
            chosen.add(lane)

        fixed_rows[time_ms] = chosen
        for lane in chosen:
            last_time[lane] = time_ms
        last_row_lanes = set(chosen)
        last_output_time = time_ms

    return [
        NoteObject(time_ms=time_ms, lane=lane)
        for time_ms in sorted(fixed_rows)
        for lane in sorted(fixed_rows[time_ms])
    ]


def _fold_tech_short_jack_into_previous(
    fixed_rows: Dict[int, Set[int]],
    last_output_time: Optional[int],
    lane: Optional[int],
    max_chord_size: int,
    last_time: Dict[int, int],
    hard_window: int,
) -> Optional[int]:
    if last_output_time is None:
        return None
    previous = fixed_rows.get(last_output_time)
    if previous is None or len(previous) >= max_chord_size:
        return None

    preferred = [lane] if lane is not None and lane not in previous else []
    candidates = preferred + [
        candidate
        for candidate in [0, 1, 2, 3]
        if candidate not in preferred
    ]
    ranked = sorted(
        [candidate for candidate in candidates if candidate not in previous],
        key=lambda candidate: (
            last_output_time - last_time.get(candidate, -999999) <= hard_window,
            -last_output_time + last_time.get(candidate, -999999),
            abs(candidate - 1.5),
            candidate,
        ),
    )
    for candidate in ranked:
        if last_output_time - last_time.get(candidate, -999999) > hard_window:
            previous.add(candidate)
            return candidate
    return None


def _reinforce_tech_safe_chords(notes: List[NoteObject], target: float, max_chord_size: int) -> List[NoteObject]:
    if target < 6.45 or max_chord_size < 2 or not notes:
        return notes

    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)

    times = sorted(rows)
    if len(times) < 2:
        return notes

    soft_window = _tech_short_jack_soft_window_ms(target)
    lane_times = {
        lane: [time_ms for time_ms in times if lane in rows[time_ms]]
        for lane in [0, 1, 2, 3]
    }
    last_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    reinforced: Dict[int, Set[int]] = {}

    for index, time_ms in enumerate(times):
        current = set(rows[time_ms])
        prev_gap = time_ms - times[index - 1] if index > 0 else 999999
        next_gap = times[index + 1] - time_ms if index + 1 < len(times) else 999999
        safe_lanes = [
            lane
            for lane in [0, 1, 2, 3]
            if lane not in current
            and time_ms - last_time.get(lane, -999999) > soft_window
            and _tech_next_lane_gap(lane_times[lane], time_ms) > soft_window
        ]

        if safe_lanes and len(current) < max_chord_size:
            desired_size = len(current)
            if prev_gap >= soft_window or next_gap >= soft_window:
                desired_size = max(desired_size, 2)
            if target >= 6.45 and (prev_gap >= soft_window * 1.08 or next_gap >= soft_window * 1.08):
                if _deterministic_index(1000, time_ms, index, int(target * 131)) < 260:
                    desired_size = max(desired_size, 3)
            if target >= 6.9 and (prev_gap >= soft_window * 1.2 or next_gap >= soft_window * 1.2):
                if _deterministic_index(1000, time_ms, index, int(target * 100)) < 420:
                    desired_size = max(desired_size, 3)
            if target >= 7.2 and _deterministic_index(1000, time_ms, index, 701) < 120:
                desired_size = max(desired_size, min(4, max_chord_size))

            ranked = sorted(
                safe_lanes,
                key=lambda lane: (
                    abs(lane - 1.5),
                    _deterministic_index(127, time_ms, index, lane),
                    lane,
                ),
            )
            for lane in ranked:
                if len(current) >= min(max_chord_size, desired_size):
                    break
                current.add(lane)

        reinforced[time_ms] = current
        for lane in current:
            last_time[lane] = time_ms

    return [
        NoteObject(time_ms=time_ms, lane=lane)
        for time_ms in sorted(reinforced)
        for lane in sorted(reinforced[time_ms])
    ]


def _prioritize_tech_music_anchors(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    max_chord_size: int,
) -> List[NoteObject]:
    if not notes or max_chord_size <= 1:
        return notes

    target = config.target_star or 4.0
    context = _music_context(analysis, snap_points, accent_snap_points)
    if not context:
        return notes

    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)
    row_times = sorted(rows)
    if not row_times:
        return notes

    soft_window = _tech_short_jack_soft_window_ms(target)
    move_limit = max(1, int(round(len(row_times) * 0.035 * _music_influence(config))))
    moves = 0
    anchors = sorted(
        (
            (time_ms, entry)
            for time_ms, entry in context.items()
            if time_ms in rows
            and entry["silent"] <= 0.0
            and entry["accent"] >= 0.56
            and (entry["protected"] > 0.0 or entry["kick"] >= 0.50 or entry["score"] >= 0.50)
        ),
        key=lambda item: (-item[1]["accent"], -item[1]["score"], item[0]),
    )

    for time_ms, entry in anchors:
        if moves >= move_limit:
            break
        current = rows.get(time_ms, set())
        desired = _music_desired_chord_size(entry, max_chord_size, "tech", target)
        if len(current) >= desired:
            continue

        for lane in _tech_anchor_priority_lanes(time_ms, current, entry, target):
            if moves >= move_limit or len(current) >= desired:
                break
            if lane in current:
                continue
            conflict_time = _tech_close_same_lane_conflict(row_times, rows, time_ms, lane, soft_window)
            if conflict_time is not None:
                conflict_entry = _music_entry(analysis, snap_points, accent_snap_points, conflict_time)
                if (
                    len(rows.get(conflict_time, set())) <= 1
                    or conflict_entry["protected"] > 0.0
                    or conflict_entry["accent"] > entry["accent"] - 0.06
                ):
                    continue
                rows[conflict_time].discard(lane)
            current.add(lane)
            moves += 1

    if moves <= 0:
        return notes

    return [
        NoteObject(time_ms=time_ms, lane=lane)
        for time_ms in sorted(rows)
        for lane in sorted(rows[time_ms])
    ]


def _tech_anchor_priority_lanes(
    time_ms: int,
    current: Set[int],
    entry: Dict[str, float],
    target: float,
) -> List[int]:
    available = [lane for lane in [0, 1, 2, 3] if lane not in current]
    return sorted(
        available,
        key=lambda lane: (
            -abs(lane - 1.5) if entry["kick"] >= 0.62 else abs(lane - 1.5),
            _deterministic_index(127, time_ms, lane, int(target * 100), int(entry["accent"] * 1000)),
            lane,
        ),
    )


def _tech_close_same_lane_conflict(
    row_times: List[int],
    rows: Dict[int, Set[int]],
    time_ms: int,
    lane: int,
    soft_window: int,
) -> Optional[int]:
    idx = bisect.bisect_left(row_times, time_ms)
    for cursor in range(idx - 1, -1, -1):
        other_time = row_times[cursor]
        if time_ms - other_time > soft_window:
            break
        if lane in rows.get(other_time, set()):
            return other_time
    for cursor in range(idx + 1, len(row_times)):
        other_time = row_times[cursor]
        if other_time - time_ms > soft_window:
            break
        if lane in rows.get(other_time, set()):
            return other_time
    return None


def _tech_next_lane_gap(times: List[int], time_ms: int) -> int:
    idx = bisect.bisect_right(times, time_ms)
    if idx >= len(times):
        return 999999
    return times[idx] - time_ms


def _tech_short_jack_hard_window_ms(target: float) -> int:
    if target < 4.5:
        return 78
    if target < 6.25:
        return 82
    return 76


def _tech_short_jack_soft_window_ms(target: float) -> int:
    if target < 4.5:
        return 104
    if target < 6.25:
        return 96
    return 84


def _tech_short_jack_replacement(
    chosen: Set[int],
    time_ms: int,
    last_time: Dict[int, int],
    hard_window: int,
    soft_window: int,
    original: List[int],
) -> Optional[int]:
    available = [lane for lane in [0, 1, 2, 3] if lane not in chosen and lane not in original]
    if not available:
        available = [lane for lane in [0, 1, 2, 3] if lane not in chosen]
    if not available:
        return None

    ranked = sorted(
        available,
        key=lambda lane: (
            time_ms - last_time.get(lane, -999999) <= hard_window,
            time_ms - last_time.get(lane, -999999) <= soft_window,
            -time_ms + last_time.get(lane, -999999),
            abs(lane - 1.5),
            lane,
        ),
    )
    best = ranked[0]
    if time_ms - last_time.get(best, -999999) <= hard_window:
        return None
    return best


def _tech_profile_max_chord_size(target: float) -> int:
    if target < 4.0:
        return 2
    if target < 5.75:
        return 3
    return 4


def _tech_profile_average_upper(target: float, max_chord_size: int, chart_type: str) -> float:
    if max_chord_size <= 1:
        return 1.0
    if target < 3.5:
        upper = 1.18
    elif target < 4.5:
        upper = 1.34
    elif target < 5.5:
        upper = 1.55
    elif target < 6.5:
        upper = 1.78
    else:
        upper = 3.40
    if chart_type == "ln":
        upper -= 0.12
    return max(1.0, min(float(max_chord_size), upper))


def _tech_style_score(divisor: int, average_chord_size: float, target: float, bpm: float, chart_type: str) -> float:
    beat_length = 60000.0 / bpm if bpm > 0 else 0.0
    interval = beat_length / divisor if divisor > 0 and beat_length > 0 else 999.0
    if target < 3.25:
        desired_interval = 86.0
    elif target < 4.25:
        desired_interval = 72.0
    elif target < 5.75:
        desired_interval = 54.0
    elif target < 6.75:
        desired_interval = 43.0
    else:
        desired_interval = 36.0 if chart_type == "ln" else 43.0

    if target < 5.75:
        chord_target = 1.18 + max(0.0, target - 3.0) * 0.08
        interval_weight = 0.38
    else:
        chord_target = 1.35 + max(0.0, target - 5.75) * 0.10
        interval_weight = 0.12
    if chart_type == "ln":
        chord_target -= 0.10

    interval_penalty = abs(interval - desired_interval) * interval_weight
    chord_penalty = abs(average_chord_size - chord_target) * 2.5
    return interval_penalty + chord_penalty


def _tech_divisor_candidates(target: float, bpm: float = 0.0) -> List[int]:
    beat_length = 60000.0 / bpm if bpm > 0 else 0.0
    if target < 3.25:
        desired_interval = 86.0
    elif target < 4.25:
        desired_interval = 72.0
    elif target < 5.75:
        desired_interval = 54.0
    elif target < 6.75:
        desired_interval = 43.0
    else:
        desired_interval = 36.0

    divisors = [3, 4, 5, 6, 8, 10, 12, 16]
    if beat_length <= 0:
        return divisors

    return sorted(
        divisors,
        key=lambda divisor: (
            abs((beat_length / divisor) - desired_interval),
            divisor < 4,
            divisor,
        ),
    )


def _build_tech_chord_sizes(
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

    if max_chord_size >= 4 and target >= 6.55 and extra_units >= 3:
        quad_budget = min(extra_units // 3, int(round(row_count * _tech_quad_ratio(target, temperature))))
        quad_candidates = _tech_chord_candidates(row_count, rows, accent_snap_points, target, for_heavy=True)
        quads = _pick_non_adjacent(
            quad_candidates,
            quad_budget,
            temperature,
            salt=int(target * 1013) + row_count,
            radius=6,
        )
        for index in quads:
            sizes[index] = 4
        extra_units -= len(quads) * 3

    if max_chord_size >= 3 and target >= 4.75 and extra_units >= 2:
        triple_budget = min(extra_units // 2, int(round(row_count * _tech_triple_ratio(target, temperature))))
        triple_candidates = [
            index
            for index in _tech_chord_candidates(row_count, rows, accent_snap_points, target, for_heavy=True)
            if sizes[index] == 1
        ]
        triples = _pick_non_adjacent(
            triple_candidates,
            triple_budget,
            temperature,
            salt=int(target * 1061) + row_count,
            radius=4,
        )
        for index in triples:
            sizes[index] = 3
        extra_units -= len(triples) * 2

    double_candidates = [
        index
        for index in _tech_chord_candidates(row_count, rows, accent_snap_points, target, for_heavy=False)
        if sizes[index] == 1
    ]
    selected = _spread_pick(
        double_candidates,
        min(extra_units, len(double_candidates)),
        temperature,
        salt=int(target * 1097) + desired_total,
    )
    for index in selected:
        sizes[index] = 2
    extra_units -= len(selected)

    if extra_units > 0:
        fallback = [index for index, size in enumerate(sizes) if size == 1]
        selected = _spread_pick(fallback, min(extra_units, len(fallback)), temperature, salt=int(target * 1151))
        for index in selected:
            sizes[index] = 2

    _thin_tech_chord_runs(sizes, target)
    return sizes


def _tech_triple_ratio(target: float, temperature: float) -> float:
    if target < 4.75:
        return 0.0
    if target < 5.75:
        base = 0.025
    elif target < 6.75:
        base = 0.055
    else:
        base = 0.085
    return max(0.0, min(0.12, base * (0.75 + temperature * 0.50)))


def _tech_quad_ratio(target: float, temperature: float) -> float:
    if target < 6.55:
        return 0.0
    base = 0.006 if target < 7.0 else 0.012
    return max(0.0, min(0.025, base * (0.75 + temperature * 0.50)))


def _tech_chord_candidates(
    row_count: int,
    rows: List[int],
    accent_snap_points: Set[int],
    target: float,
    for_heavy: bool,
) -> List[int]:
    groups: List[List[int]] = [
        [index for index, time_ms in enumerate(rows) if time_ms in accent_snap_points],
        [index for index in range(row_count) if index % 16 in [0, 8]],
        [index for index in range(row_count) if index % 12 in [4, 10]],
    ]
    if not for_heavy:
        groups.append([index for index in range(row_count) if index % 8 in [2, 6]])
    if target >= 5.25:
        groups.append([index for index in range(row_count) if index % 6 in [1, 4]])
    if target >= 6.25 and not for_heavy:
        groups.append([index for index in range(row_count) if index % 3 == 1])

    ordered: List[int] = []
    seen: Set[int] = set()
    for group in groups:
        for index in group:
            if index in seen:
                continue
            ordered.append(index)
            seen.add(index)
    return ordered


def _thin_tech_chord_runs(sizes: List[int], target: float) -> None:
    max_run = 2 if target < 4.75 else 4 if target < 6.25 else 6
    run = 0
    demoted = 0
    for index, size in enumerate(sizes):
        if size >= 2:
            run += 1
            if run > max_run:
                demoted += size - 1
                sizes[index] = 1
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
        if index > 0 and sizes[index - 1] >= 2 and index + 1 < len(sizes) and sizes[index + 1] >= 2:
            continue
        sizes[index] = 2
        demoted -= 1


def _build_tech_notes(
    rows: List[int],
    sizes: List[int],
    target: float,
    temperature: float,
) -> List[NoteObject]:
    notes: List[NoteObject] = []
    previous_lanes: List[int] = []
    recent_lanes: List[List[int]] = []
    phrase_index = -1
    phrase_start = 0
    phrase_len = 0
    module_tokens: List[List[int]] = []

    for row_index, (time_ms, size) in enumerate(zip(rows, sizes)):
        if row_index >= phrase_start + phrase_len:
            phrase_index += 1
            phrase_start = row_index
            phrase_len = _tech_phrase_length(target, temperature, phrase_index)
            module_tokens = _tech_module_tokens(target, temperature, phrase_index)

        local_index = row_index - phrase_start
        token = module_tokens[local_index % len(module_tokens)] if module_tokens else [row_index % 4]
        row_gap_ms = time_ms - rows[row_index - 1] if row_index > 0 else 999999
        lanes = _fit_tech_token_to_size(
            token,
            size,
            previous_lanes,
            recent_lanes,
            row_index,
            phrase_index,
            target,
            temperature,
            row_gap_ms,
        )
        previous_lanes = lanes
        recent_lanes.append(lanes)
        if len(recent_lanes) > 10:
            recent_lanes.pop(0)

        for lane in lanes:
            notes.append(NoteObject(time_ms=time_ms, lane=lane))

    return notes


def _soften_tech_fast_jacks(notes: List[NoteObject], target: float, temperature: float) -> List[NoteObject]:
    if not notes:
        return notes

    rows: Dict[int, List[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note.lane)

    row_times = sorted(rows)
    fast_window = _tech_fast_jack_window_ms(target)
    fast_pair_budget = int(round(len(row_times) * _tech_fast_pair_ratio(target, temperature)))
    last_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    fast_streak = {0: 0, 1: 0, 2: 0, 3: 0}
    softened: List[NoteObject] = []

    for row_index, time_ms in enumerate(row_times):
        original_lanes = sorted(set(rows[time_ms]))
        chosen: List[int] = []

        for lane in original_lanes:
            lane = _resolve_tech_fast_lane(
                lane,
                chosen,
                time_ms,
                row_index,
                target,
                fast_window,
                fast_pair_budget,
                last_time,
                fast_streak,
            )
            if lane in chosen:
                continue

            if last_time[lane] > -999999 and time_ms - last_time[lane] <= fast_window:
                if fast_pair_budget > 0 and fast_streak[lane] < 1:
                    fast_pair_budget -= 1
                    fast_streak[lane] += 1
                else:
                    replacement = _find_tech_replacement_lane(chosen, time_ms, fast_window, last_time)
                    if replacement is not None:
                        lane = replacement
                        fast_streak[lane] = 0
                    elif chosen or len(original_lanes) > 1:
                        continue
            else:
                fast_streak[lane] = 0

            chosen.append(lane)
            last_time[lane] = time_ms

        for lane in sorted(chosen):
            softened.append(NoteObject(time_ms=time_ms, lane=lane))

    return softened


def _tech_fast_pair_ratio(target: float, temperature: float) -> float:
    if target < 4.5:
        base = 0.006
    elif target < 6.25:
        base = 0.012
    else:
        base = 0.018
    return max(0.0, min(0.026, base * (0.75 + max(0.0, min(1.0, temperature)) * 0.50)))


def _resolve_tech_fast_lane(
    lane: int,
    chosen: List[int],
    time_ms: int,
    row_index: int,
    target: float,
    fast_window: int,
    fast_pair_budget: int,
    last_time: Dict[int, int],
    fast_streak: Dict[int, int],
) -> int:
    if lane in chosen:
        replacement = _find_tech_replacement_lane(chosen, time_ms, fast_window, last_time)
        return replacement if replacement is not None else lane

    recent_gap = time_ms - last_time.get(lane, -999999)
    if recent_gap > fast_window:
        return lane

    replacement = _find_tech_replacement_lane(chosen, time_ms, fast_window, last_time)
    return replacement if replacement is not None else lane


def _find_tech_replacement_lane(
    chosen: List[int],
    time_ms: int,
    fast_window: int,
    last_time: Dict[int, int],
) -> Optional[int]:
    available = [lane for lane in [0, 1, 2, 3] if lane not in chosen]
    if not available:
        return None
    ranked = sorted(
        available,
        key=lambda lane: (
            time_ms - last_time.get(lane, -999999) <= fast_window,
            -time_ms + last_time.get(lane, -999999),
            abs(lane - 1.5),
            lane,
        ),
    )
    if time_ms - last_time.get(ranked[0], -999999) <= fast_window:
        return None
    return ranked[0]


def _tech_phrase_length(target: float, temperature: float, phrase_index: int) -> int:
    base = 16 if target < 4.5 else 12 if target < 6.25 else 8
    if temperature < 0.30:
        return base
    jitter_span = 2 if temperature < 0.75 else 4
    return max(6, base + _deterministic_jitter(phrase_index, int(target * 313), jitter_span))


def _tech_module_tokens(target: float, temperature: float, phrase_index: int) -> List[List[int]]:
    modules = _tech_modules()
    if target < 3.75:
        names = ["stair", "cross", "anchor_return"]
    elif target < 5.25:
        names = ["cross", "stair", "chord_tech", "anchor_return"]
    elif target < 6.5:
        names = ["chord_tech", "cross_chord", "stair", "burst", "anchor_return"]
    else:
        names = ["chord_tech", "cross_chord", "burst", "stair", "anchor_return"]

    if temperature >= 0.75:
        names = list(dict.fromkeys(names + ["cross_chord", "burst"]))
    choice = _deterministic_index(len(names), phrase_index, int(target * 100), int(temperature * 1000))
    return [list(token) for token in modules[names[choice]]]


def _tech_modules() -> Dict[str, List[List[int]]]:
    return {
        "stair": [[0], [1], [2], [3], [0], [1], [3], [2], [1], [3], [0], [2], [2], [0], [3], [1]],
        "cross": [[0], [2], [1], [3], [0], [3], [1], [2], [3], [1], [2], [0], [1], [0], [2], [3]],
        "anchor_return": [[0], [2], [0], [1], [3], [0], [2], [1], [2], [0], [3], [2], [1], [0], [3], [1]],
        "burst": [[0], [1], [2], [3], [0], [2], [1], [3], [0, 1], [2], [3], [0, 3], [1], [2, 3], [0], [1, 2]],
        "chord_tech": [[0, 1], [2], [3], [0, 3], [1], [2, 3], [0], [1, 3], [0, 2], [3], [1], [0, 1], [2], [0, 3], [1, 2], [3]],
        "cross_chord": [[0, 2], [3], [1], [0, 3], [1, 2], [0], [3], [1], [0, 1], [2], [3], [1, 3], [0], [2, 3], [1], [0, 2]],
    }


def _fit_tech_token_to_size(
    token: List[int],
    size: int,
    previous_lanes: List[int],
    recent_lanes: List[List[int]],
    row_index: int,
    phrase_index: int,
    target: float,
    temperature: float,
    row_gap_ms: int,
) -> List[int]:
    size = max(1, min(4, size))
    base = [lane for lane in token if lane in [0, 1, 2, 3]]
    if not base:
        base = [row_index % 4]

    patterns = _tech_patterns(size)
    previous_set = set(previous_lanes)
    base_set = set(base)
    fast_context = row_gap_ms <= _tech_fast_jack_window_ms(target)
    base_candidates = [lanes for lanes in patterns if base_set & set(lanes)]
    if fast_context:
        # In tech, anchors should usually mean returning to a finger after movement,
        # not adjacent high-resolution jacks. Treat the module token as a preference
        # instead of a hard constraint when the row gap is very small.
        candidates = patterns
    else:
        candidates = base_candidates or patterns

    ranked = sorted(
        candidates,
        key=lambda lanes: (
            _tech_lane_penalty(lanes, previous_set, recent_lanes, target, row_gap_ms),
            -len(set(lanes) & base_set),
            abs(sum(lanes) - 3),
            _deterministic_index(127, row_index, phrase_index, size, int(target * 100), int(temperature * 1000)),
            patterns.index(lanes),
        ),
    )
    if fast_context:
        pool_size = 1
    else:
        pool_size = 1 + int(round(max(0.0, min(1.0, temperature)) * min(3, len(ranked) - 1)))
    pool = ranked[: max(1, pool_size)]
    choice = _deterministic_index(
        len(pool),
        row_index,
        phrase_index,
        size,
        sum(previous_lanes) if previous_lanes else 0,
    )
    return list(pool[choice])


def _tech_fast_jack_window_ms(target: float) -> int:
    if target < 4.5:
        return 122
    if target < 6.25:
        return 116
    if target < 6.75:
        return 110
    return 110


def _tech_lane_penalty(
    lanes: List[int],
    previous_set: Set[int],
    recent_lanes: List[List[int]],
    target: float,
    row_gap_ms: int,
) -> float:
    lane_set = set(lanes)
    overlap = len(lane_set & previous_set)
    fast_window = _tech_fast_jack_window_ms(target)
    fast_context = row_gap_ms <= fast_window
    penalty = overlap * (1.5 if target < 5.5 else 1.0)
    if fast_context:
        penalty += overlap * (11.0 if target < 6.5 else 8.0)
    if not overlap:
        penalty -= 3.0 if fast_context else 2.0
    if lane_set == previous_set:
        penalty += 18.0 if fast_context else 8.0
    if fast_context and row_gap_ms > 0:
        lookback_rows = max(1, min(len(recent_lanes), fast_window // max(1, row_gap_ms)))
        for distance in range(2, lookback_rows + 1):
            old_set = set(recent_lanes[-distance])
            old_overlap = len(lane_set & old_set)
            penalty += old_overlap * (5.5 / distance)
    if len(recent_lanes) >= 2 and lane_set == set(recent_lanes[-2]):
        penalty += 5.0 if fast_context else 2.5
    if len(recent_lanes) >= 5:
        penalty += sum(1 for old in recent_lanes[-5:] if set(old) == lane_set) * 0.8
    return penalty


def _tech_patterns(size: int) -> List[List[int]]:
    if size <= 1:
        return [[0], [1], [2], [3]]
    if size == 2:
        return [[0, 1], [2, 3], [0, 3], [0, 2], [1, 3], [1, 2]]
    if size == 3:
        return [[0, 1, 3], [0, 2, 3], [0, 1, 2], [1, 2, 3]]
    return [[0, 1, 2, 3]]


def _apply_safe_lns(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    style: str,
) -> List[NoteObject]:
    if config.chart_type != "ln" or config.ln_ratio <= 0 or not notes:
        return notes
    accent_snap_points = build_accent_snap_points(analysis, snap_points)
    if style == "stream":
        return _apply_stream_lns(notes, config, analysis, snap_points, accent_snap_points)
    if style == "speed":
        return _apply_speed_lns(notes, config, analysis, snap_points, accent_snap_points)
    if style == "tech":
        return _apply_tech_lns(notes, config, analysis, snap_points, accent_snap_points)

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    sorted_notes = sorted(notes, key=lambda note: (note.time_ms, note.lane))
    next_same_times = _next_same_lane_time_by_index(sorted_notes)
    row_times = sorted({note.time_ms for note in sorted_notes})
    row_index_by_time = {time_ms: index for index, time_ms in enumerate(row_times)}
    ln_interval = max(2, int(round(1.0 / max(0.02, min(1.0, config.ln_ratio)))))
    tail_gap_ms = 35
    lane_tail_until = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    result: List[NoteObject] = []
    influence = _music_influence(config)

    for index, note in enumerate(sorted_notes):
        copied = NoteObject(time_ms=note.time_ms, lane=note.lane)
        row_index = row_index_by_time.get(note.time_ms, 0)

        if note.time_ms > lane_tail_until[note.lane] + tail_gap_ms:
            next_same = next_same_times.get(index)
            max_length = config.max_ln_ms
            if next_same is not None:
                max_length = min(max_length, max(0, next_same - note.time_ms - tail_gap_ms))
            else:
                max_length = min(max_length, int(round(beat_length * (2.0 if style == "stream" else 1.5))))

            if max_length >= config.min_ln_ms and _should_place_music_ln(
                row_index,
                note.lane,
                note.time_ms,
                ln_interval,
                config.ln_ratio,
                target=config.target_star or 4.0,
                style=style,
                analysis=analysis,
                snap_points=snap_points,
                accent_snap_points=accent_snap_points,
                influence=influence,
            ):
                preferred_length = _safe_ln_length(
                    config.min_ln_ms,
                    max_length,
                    beat_length,
                    row_index,
                    note.lane,
                    style,
                )
                length = _music_rank_generic_ln_length(
                    note.time_ms,
                    preferred_length,
                    config.min_ln_ms,
                    max_length,
                    beat_length,
                    row_times,
                    row_index,
                    analysis,
                    snap_points,
                    accent_snap_points,
                    influence,
                )
                copied.end_time_ms = note.time_ms + length
                lane_tail_until[note.lane] = copied.end_time_ms

        result.append(copied)

    return result


def _apply_stream_lns(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    sorted_notes = sorted(notes, key=lambda note: (note.time_ms, note.lane))
    row_times = sorted({note.time_ms for note in sorted_notes})
    if len(row_times) < 2:
        return sorted_notes

    gaps = [row_times[index + 1] - row_times[index] for index in range(len(row_times) - 1)]
    row_step = int(round(sorted(gaps)[len(gaps) // 2]))
    min_tail_gap = max(31, int(round(row_step * 0.75)))
    row_index_by_time = {time_ms: index for index, time_ms in enumerate(row_times)}
    next_same_times = _next_same_lane_time_by_index(sorted_notes)
    target = config.target_star or 4.0
    ln_ratio = _stream_effective_ln_ratio(config, target)
    max_lns_per_row = 1 if target < 6.5 else 2
    influence = _music_influence(config)

    lane_tail_until = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    row_ln_counts: Dict[int, int] = {}
    result: List[NoteObject] = []

    for index, note in enumerate(sorted_notes):
        copied = NoteObject(time_ms=note.time_ms, lane=note.lane)
        row_index = row_index_by_time.get(note.time_ms, 0)

        if (
            note.time_ms > lane_tail_until[note.lane] + min_tail_gap
            and row_ln_counts.get(row_index, 0) < max_lns_per_row
            and _should_place_music_ln(
                row_index,
                note.lane,
                note.time_ms,
                None,
                ln_ratio,
                target,
                "stream",
                analysis,
                snap_points,
                accent_snap_points,
                influence,
            )
        ):
            next_same = next_same_times.get(index)
            for length_rows in _rank_ln_length_rows_by_music(
                _stream_ln_length_rows(row_index, note.lane, target),
                row_times,
                row_index,
                note.time_ms,
                target,
                "stream",
                analysis,
                snap_points,
                accent_snap_points,
                influence,
            ):
                tail_row_index = row_index + length_rows
                if tail_row_index >= len(row_times):
                    continue
                tail_time = row_times[tail_row_index]
                if next_same is not None and tail_time > next_same - min_tail_gap:
                    continue
                if tail_time <= note.time_ms:
                    continue

                copied.end_time_ms = tail_time
                lane_tail_until[note.lane] = tail_time
                row_ln_counts[row_index] = row_ln_counts.get(row_index, 0) + 1
                break

        result.append(copied)

    return result


def _stream_effective_ln_ratio(config: DifficultyConfig, target: float) -> float:
    floor = 0.36 + max(0.0, min(1.0, (target - 3.0) / 4.0)) * 0.24
    return max(0.0, min(0.72, max(config.ln_ratio, floor)))


def _should_place_stream_ln(row_index: int, lane: int, ln_ratio: float, target: float) -> bool:
    threshold = int(round(max(0.0, min(1.0, ln_ratio)) * 1000))
    roll = _deterministic_index(1000, row_index, lane * 37, int(target * 100), 0x51A7)
    return roll < threshold


def _stream_ln_length_rows(row_index: int, lane: int, target: float) -> List[int]:
    if target < 4.5:
        options = [1, 1, 2, 1, 2, 4]
    elif target < 6.25:
        options = [1, 1, 1, 2, 1, 2, 4]
    else:
        options = [1, 1, 1, 1, 2, 1, 2, 4]

    start = _deterministic_index(len(options), row_index, lane, int(target * 100), 0x6D2B)
    return options[start:] + options[:start]


def _apply_speed_lns(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    sorted_notes = sorted(notes, key=lambda note: (note.time_ms, note.lane))
    row_times = sorted({note.time_ms for note in sorted_notes})
    if len(row_times) < 2:
        return sorted_notes

    gaps = [row_times[index + 1] - row_times[index] for index in range(len(row_times) - 1)]
    row_step = int(round(sorted(gaps)[len(gaps) // 2]))
    min_tail_gap = max(31, int(round(row_step * 0.70)))
    row_index_by_time = {time_ms: index for index, time_ms in enumerate(row_times)}
    next_same_times = _next_same_lane_time_by_index(sorted_notes)
    target = config.target_star or 4.0
    ln_ratio = _speed_effective_ln_ratio(config, target)
    max_lns_per_row = 1 if target < 6.75 else 2
    influence = _music_influence(config)

    lane_tail_until = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    row_ln_counts: Dict[int, int] = {}
    result: List[NoteObject] = []

    for index, note in enumerate(sorted_notes):
        copied = NoteObject(time_ms=note.time_ms, lane=note.lane)
        row_index = row_index_by_time.get(note.time_ms, 0)

        if (
            note.time_ms > lane_tail_until[note.lane] + min_tail_gap
            and row_ln_counts.get(row_index, 0) < max_lns_per_row
            and _should_place_music_ln(
                row_index,
                note.lane,
                note.time_ms,
                None,
                ln_ratio,
                target,
                "speed",
                analysis,
                snap_points,
                accent_snap_points,
                influence,
            )
        ):
            next_same = next_same_times.get(index)
            for length_rows in _rank_ln_length_rows_by_music(
                _speed_ln_length_rows(row_index, note.lane, target),
                row_times,
                row_index,
                note.time_ms,
                target,
                "speed",
                analysis,
                snap_points,
                accent_snap_points,
                influence,
            ):
                tail_row_index = row_index + length_rows
                if tail_row_index >= len(row_times):
                    continue
                tail_time = row_times[tail_row_index]
                if next_same is not None and tail_time > next_same - min_tail_gap:
                    continue
                if tail_time <= note.time_ms:
                    continue

                copied.end_time_ms = tail_time
                lane_tail_until[note.lane] = tail_time
                row_ln_counts[row_index] = row_ln_counts.get(row_index, 0) + 1
                break

        result.append(copied)

    return result


def _speed_effective_ln_ratio(config: DifficultyConfig, target: float) -> float:
    floor = 0.44 + max(0.0, min(1.0, (target - 3.0) / 4.0)) * 0.36
    return max(0.0, min(0.88, max(config.ln_ratio, floor)))


def _should_place_speed_ln(row_index: int, lane: int, ln_ratio: float, target: float) -> bool:
    threshold = int(round(max(0.0, min(1.0, ln_ratio)) * 1000))
    roll = _deterministic_index(1000, row_index, lane * 41, int(target * 100), 0x7C95)
    return roll < threshold


def _speed_ln_length_rows(row_index: int, lane: int, target: float) -> List[int]:
    if target < 4.5:
        options = [1, 1, 2, 1, 2, 3]
    elif target < 6.25:
        options = [1, 1, 1, 2, 1, 2, 3]
    else:
        options = [1, 1, 1, 1, 2, 1, 2, 3]

    start = _deterministic_index(len(options), row_index, lane, int(target * 100), 0x2E49)
    return options[start:] + options[:start]


def _apply_tech_lns(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    sorted_notes = sorted(notes, key=lambda note: (note.time_ms, note.lane))
    row_times = sorted({note.time_ms for note in sorted_notes})
    if len(row_times) < 2:
        return sorted_notes

    gaps = [row_times[index + 1] - row_times[index] for index in range(len(row_times) - 1)]
    row_step = int(round(sorted(gaps)[len(gaps) // 2]))
    min_tail_gap = max(31, int(round(row_step * 0.55)))
    row_index_by_time = {time_ms: index for index, time_ms in enumerate(row_times)}
    next_same_times = _next_same_lane_time_by_index(sorted_notes)
    target = config.target_star or 4.0
    ln_ratio = _tech_effective_ln_ratio(config, target)
    max_lns_per_row = 1 if target < 4.5 else 2 if target < 6.25 else 3
    influence = _music_influence(config)

    lane_tail_until = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    row_ln_counts: Dict[int, int] = {}
    result: List[NoteObject] = []

    for index, note in enumerate(sorted_notes):
        copied = NoteObject(time_ms=note.time_ms, lane=note.lane)
        row_index = row_index_by_time.get(note.time_ms, 0)

        if (
            note.time_ms > lane_tail_until[note.lane] + min_tail_gap
            and row_ln_counts.get(row_index, 0) < max_lns_per_row
            and _should_place_music_ln(
                row_index,
                note.lane,
                note.time_ms,
                None,
                ln_ratio,
                target,
                "tech",
                analysis,
                snap_points,
                accent_snap_points,
                influence,
            )
        ):
            next_same = next_same_times.get(index)
            for length_rows in _rank_ln_length_rows_by_music(
                _tech_ln_length_rows(row_index, note.lane, target),
                row_times,
                row_index,
                note.time_ms,
                target,
                "tech",
                analysis,
                snap_points,
                accent_snap_points,
                influence,
            ):
                tail_row_index = row_index + length_rows
                if tail_row_index >= len(row_times):
                    continue
                tail_time = row_times[tail_row_index]
                if next_same is not None and tail_time > next_same - min_tail_gap:
                    continue
                if tail_time <= note.time_ms:
                    continue

                copied.end_time_ms = tail_time
                lane_tail_until[note.lane] = tail_time
                row_ln_counts[row_index] = row_ln_counts.get(row_index, 0) + 1
                break

        result.append(copied)

    return result


def _tech_effective_ln_ratio(config: DifficultyConfig, target: float) -> float:
    floor = 0.50 + max(0.0, min(1.0, (target - 3.0) / 4.0)) * 0.30
    return max(0.0, min(0.86, max(config.ln_ratio, floor)))


def _should_place_music_ln(
    row_index: int,
    lane: int,
    time_ms: int,
    interval: Optional[int],
    ln_ratio: float,
    target: float,
    style: str,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    influence: float,
) -> bool:
    if influence <= 0.01:
        if style == "stream":
            return _should_place_stream_ln(row_index, lane, ln_ratio, target)
        if style == "speed":
            return _should_place_speed_ln(row_index, lane, ln_ratio, target)
        if style == "tech":
            return _should_place_tech_ln(row_index, lane, ln_ratio, target)
        return _should_place_safe_ln(row_index, lane, interval or 2, style)

    entry = _music_entry(analysis, snap_points, accent_snap_points, time_ms)
    base_threshold = int(round(max(0.0, min(1.0, ln_ratio)) * 1000))
    if interval is not None:
        base_threshold = max(base_threshold, int(round(1000.0 / max(1, interval))))

    sustain_head = max(entry.get("sustain", 0.0), entry.get("vocal", 0.0))
    head_score = (
        entry["score"] * 0.48
        + entry["onset"] * 0.18
        + entry["kick"] * 0.12
        + entry["beat"] * 0.06
        + sustain_head * 0.16
    )
    multiplier = 0.64 + influence * (0.42 + head_score * 0.82)
    if entry["protected"] > 0.0:
        multiplier += 0.16 * influence
    if entry["score"] < 0.22 and entry["onset"] < 0.10 and sustain_head < 0.18:
        multiplier *= 1.0 - 0.42 * influence
    if style == "speed" and entry["score"] < 0.34:
        multiplier *= 1.0 - 0.18 * influence

    threshold = int(round(base_threshold * max(0.25, min(1.55, multiplier))))
    if style == "tech" and target >= 6.0 and row_index % 16 in [0, 1, 8, 9]:
        threshold = min(1000, threshold + int(120 * (0.35 + influence * 0.65)))

    salt = {"jack": 0x31D1, "stream": 0x51A7, "speed": 0x7C95, "tech": 0x4A6F}.get(style, 0x2473)
    roll = _deterministic_index(1000, row_index, lane * 43, int(target * 100), salt)
    return roll < max(0, min(1000, threshold))


def _rank_ln_length_rows_by_music(
    length_rows: List[int],
    row_times: List[int],
    row_index: int,
    head_time: int,
    target: float,
    style: str,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    influence: float,
) -> List[int]:
    if influence <= 0.05 or not length_rows:
        return length_rows

    unique_lengths = list(dict.fromkeys(length_rows))
    original_rank = {length: index for index, length in enumerate(unique_lengths)}
    ranked = sorted(
        unique_lengths,
        key=lambda length: (
            -_ln_tail_music_score(
                row_times,
                row_index,
                length,
                head_time,
                target,
                style,
                analysis,
                snap_points,
                accent_snap_points,
                influence,
            ),
            original_rank[length],
        ),
    )
    return ranked


def _ln_tail_music_score(
    row_times: List[int],
    row_index: int,
    length_rows: int,
    head_time: int,
    target: float,
    style: str,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    influence: float,
) -> float:
    tail_index = row_index + length_rows
    if tail_index >= len(row_times):
        return -999.0
    tail_time = row_times[tail_index]
    head = _music_entry(analysis, snap_points, accent_snap_points, head_time)
    tail = _music_entry(analysis, snap_points, accent_snap_points, tail_time)
    next_entry = _music_entry(analysis, snap_points, accent_snap_points, row_times[tail_index + 1]) if tail_index + 1 < len(row_times) else tail

    body_sustain = _analysis_curve_average(analysis, "sustain_curve", head_time, tail_time, tail.get("sustain", 0.0))
    body_vocal = _analysis_curve_average(analysis, "vocal_sustain_curve", head_time, tail_time, tail.get("vocal", 0.0))
    release_score = tail.get("release", 0.0) * 0.34 + max(0.0, head.get("sustain", 0.0) - tail.get("sustain", 0.0)) * 0.12
    body_score = max(body_sustain, body_vocal * 1.08) * 0.30
    beat_release = tail["beat"] * 0.08 + tail["score"] * 0.04
    onset_penalty = tail["onset"] * (0.16 if tail.get("release", 0.0) < 0.32 else 0.05)
    next_accent_preparation = next_entry["accent"] * 0.16 if style in ["stream", "tech"] else next_entry["accent"] * 0.08
    length_preference = 0.0
    if style == "stream":
        length_preference = 0.18 if length_rows in [1, 2] else 0.04
    elif style == "speed":
        length_preference = 0.20 if length_rows == 1 else 0.08 if length_rows == 2 else 0.0
    elif style == "tech":
        length_preference = 0.16 if length_rows in [1, 2] else 0.08
    else:
        length_preference = 0.10 if length_rows in [1, 2, 3] else 0.03

    if tail["silent"] > 0.0:
        return -50.0
    if target < 4.5 and length_rows <= 1 and style != "speed":
        length_preference -= 0.08

    return (beat_release + release_score + body_score + next_accent_preparation + length_preference - onset_penalty) * influence


def _music_rank_generic_ln_length(
    head_time: int,
    preferred_length: int,
    min_ln_ms: int,
    max_length: int,
    beat_length: float,
    row_times: List[int],
    row_index: int,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    influence: float,
) -> int:
    if influence <= 0.05:
        return preferred_length

    candidates = [
        preferred_length,
        int(round(beat_length * 0.50)),
        int(round(beat_length * 0.75)),
        int(round(beat_length)),
        int(round(beat_length * 1.50)),
        int(round(beat_length * 2.00)),
    ]
    candidates = [
        length
        for length in dict.fromkeys(candidates)
        if min_ln_ms <= length <= max_length
    ]
    if not candidates:
        return preferred_length

    head = _music_entry(analysis, snap_points, accent_snap_points, head_time)
    ranked = sorted(
        candidates,
        key=lambda length: (
            -_generic_ln_tail_score(
                head,
                head_time,
                head_time + length,
                analysis,
                snap_points,
                accent_snap_points,
                influence,
            ),
            abs(length - preferred_length),
        ),
    )
    return ranked[0]


def _generic_ln_tail_score(
    head: Dict[str, float],
    head_time: int,
    tail_time: int,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    influence: float,
) -> float:
    nearest = _nearest_snap_point(snap_points, tail_time)
    if nearest is None:
        return -999.0
    distance = abs(nearest - tail_time)
    snap_penalty = min(1.0, distance / 80.0) * 0.20
    tail = _music_entry(analysis, snap_points, accent_snap_points, nearest)
    body_sustain = _analysis_curve_average(analysis, "sustain_curve", head_time, nearest, tail.get("sustain", 0.0))
    body_vocal = _analysis_curve_average(analysis, "vocal_sustain_curve", head_time, nearest, tail.get("vocal", 0.0))
    release_score = tail.get("release", 0.0) * 0.36 + max(0.0, head.get("sustain", 0.0) - tail.get("sustain", 0.0)) * 0.12
    onset_penalty = tail["onset"] * (0.14 if tail.get("release", 0.0) < 0.32 else 0.04)
    return (
        tail["beat"] * 0.08
        + tail["score"] * 0.04
        + max(body_sustain, body_vocal * 1.08) * 0.28
        + release_score
        - onset_penalty
        - snap_penalty
    ) * influence


def _should_place_tech_ln(row_index: int, lane: int, ln_ratio: float, target: float) -> bool:
    threshold = int(round(max(0.0, min(1.0, ln_ratio)) * 1000))
    roll = _deterministic_index(1000, row_index, lane * 43, int(target * 100), 0x4A6F)
    if target >= 6.0 and row_index % 16 in [0, 1, 8, 9]:
        threshold = min(1000, threshold + 120)
    return roll < threshold


def _tech_ln_length_rows(row_index: int, lane: int, target: float) -> List[int]:
    if target < 4.5:
        options = [1, 2, 1, 2, 3]
    elif target < 6.25:
        options = [2, 1, 2, 3, 1, 4]
    else:
        options = [2, 1, 2, 1, 3, 2, 4]

    start = _deterministic_index(len(options), row_index, lane, int(target * 100), 0x5B73)
    return options[start:] + options[:start]


def _next_same_lane_time_by_index(notes: List[NoteObject]) -> Dict[int, int]:
    next_times: Dict[int, int] = {}
    last_seen: Dict[int, int] = {}
    for index in range(len(notes) - 1, -1, -1):
        lane = notes[index].lane
        if lane in last_seen:
            next_times[index] = last_seen[lane]
        last_seen[lane] = notes[index].time_ms
    return next_times


def _refine_lns_to_sustain_and_hits(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> List[NoteObject]:
    if config.chart_type not in ["ln", "hybrid"] or not notes or not any(note.is_ln for note in notes):
        return notes

    context = _music_context(analysis, snap_points, accent_snap_points)
    target = config.target_star or 4.0
    max_chord_size = clamp_max_chord_size(config)
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    tail_gap = max(34, int(min(58, beat_length / 12.0)))
    min_ln_ms = max(30, int(config.min_ln_ms))
    if config.key_style in ["stream", "speed", "tech"]:
        min_ln_ms = max(30, min(min_ln_ms, int(max(45, beat_length / 8.0))))

    sorted_notes = sorted(notes, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))
    lane_times: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for note in sorted_notes:
        lane_times[note.lane].append(note.time_ms)
    for lane in lane_times:
        lane_times[lane] = sorted(set(lane_times[lane]))

    accepted_lns: List[Tuple[int, int, int]] = []
    refined: List[NoteObject] = []
    for note in sorted_notes:
        if not note.is_ln or note.end_time_ms is None:
            refined.append(NoteObject(time_ms=note.time_ms, lane=note.lane))
            continue

        head_time = note.time_ms
        latest_tail = min(head_time + int(config.max_ln_ms), max(head_time, int(note.end_time_ms) + int(beat_length * 0.75)))
        same_lane_times = lane_times.get(note.lane, [])
        next_idx = bisect.bisect_right(same_lane_times, head_time)
        if next_idx < len(same_lane_times):
            latest_tail = min(latest_tail, same_lane_times[next_idx] - tail_gap)
        if latest_tail - head_time < min_ln_ms:
            refined.append(NoteObject(time_ms=head_time, lane=note.lane))
            continue

        tail_time = _choose_sustain_release_tail(
            head_time,
            int(note.end_time_ms),
            latest_tail,
            min_ln_ms,
            analysis,
            snap_points,
            accent_snap_points,
            target,
            beat_length,
        )
        if not _ln_has_sustain_reason(head_time, tail_time, analysis, snap_points, accent_snap_points):
            refined.append(NoteObject(time_ms=head_time, lane=note.lane))
            continue
        tail_time = _limit_ln_tail_for_music_hits(
            head_time,
            tail_time,
            note.lane,
            accepted_lns,
            min_ln_ms,
            tail_gap,
            config,
            context,
            snap_points,
            target,
        )

        if tail_time - head_time < min_ln_ms:
            refined.append(NoteObject(time_ms=head_time, lane=note.lane))
            continue

        refined_note = NoteObject(time_ms=head_time, lane=note.lane, end_time_ms=tail_time)
        refined.append(refined_note)
        accepted_lns.append((head_time, tail_time, note.lane))

    refined = _restore_music_hits_after_ln_refine(
        refined,
        config,
        analysis,
        snap_points,
        accent_snap_points,
        context,
        target,
        max_chord_size,
        tail_gap,
    )
    refined = _release_ln_blocks_for_music_hits(
        refined,
        config,
        analysis,
        snap_points,
        accent_snap_points,
        context,
        target,
        max_chord_size,
        min_ln_ms,
        tail_gap,
    )
    refined = _restore_music_hits_after_ln_refine(
        refined,
        config,
        analysis,
        snap_points,
        accent_snap_points,
        context,
        target,
        max_chord_size,
        tail_gap,
    )
    return sorted(refined, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _choose_sustain_release_tail(
    head_time: int,
    preferred_tail: int,
    latest_tail: int,
    min_ln_ms: int,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    target: float,
    beat_length: float,
) -> int:
    earliest_tail = head_time + min_ln_ms
    latest_tail = max(earliest_tail, latest_tail)
    candidates: Set[int] = {max(earliest_tail, min(latest_tail, preferred_tail))}

    start_idx = bisect.bisect_left(snap_points, earliest_tail)
    end_idx = bisect.bisect_right(snap_points, latest_tail)
    window_snaps = snap_points[start_idx:end_idx]
    if len(window_snaps) <= 96:
        candidates.update(window_snaps)
    else:
        step = max(1, len(window_snaps) // 96)
        candidates.update(window_snaps[::step])
        candidates.update(window_snaps[-8:])

    for release in analysis.get("release_points_ms", []):
        if isinstance(release, dict):
            release_time = int(release.get("time_ms", 0))
        else:
            release_time = int(release)
        if earliest_tail <= release_time <= latest_tail:
            snapped = _nearest_snap_point(snap_points, release_time)
            if snapped is not None and earliest_tail <= snapped <= latest_tail:
                candidates.add(snapped)

    sustain_segments = analysis.get("sustain_segments", [])
    for segment in sustain_segments:
        try:
            start = int(segment.get("start_ms", 0))
            end = int(segment.get("end_ms", 0))
        except AttributeError:
            continue
        if start - 120 <= head_time <= end + 80:
            snapped = _nearest_snap_point(snap_points, min(end, latest_tail))
            if snapped is not None and earliest_tail <= snapped <= latest_tail:
                candidates.add(snapped)

    best_tail = max(earliest_tail, min(latest_tail, preferred_tail))
    best_score = -999.0
    for tail_time in sorted(candidates):
        tail = _music_entry(analysis, snap_points, accent_snap_points, tail_time)
        if tail.get("silent", 0.0) > 0.0:
            continue
        body_sustain = _analysis_curve_average(analysis, "sustain_curve", head_time, tail_time, tail.get("sustain", 0.0))
        body_vocal = _analysis_curve_average(analysis, "vocal_sustain_curve", head_time, tail_time, tail.get("vocal", 0.0))
        body_score = max(body_sustain, body_vocal * 1.10)
        release_score = tail.get("release", 0.0)
        preferred_penalty = min(1.0, abs(tail_time - preferred_tail) / max(1.0, beat_length * 1.50)) * 0.10
        length_penalty = 0.0
        if target < 5.0 and tail_time - head_time > beat_length * 1.25:
            length_penalty = 0.12
        onset_penalty = tail.get("onset", 0.0) * (0.20 if release_score < 0.34 else 0.06)
        weak_body_penalty = 0.18 if body_score < 0.13 and release_score < 0.18 else 0.0
        segment_bonus = _sustain_segment_bonus(sustain_segments, head_time, tail_time)
        score = (
            release_score * 0.42
            + body_score * 0.34
            + tail.get("beat", 0.0) * 0.04
            + segment_bonus * 0.16
            - onset_penalty
            - preferred_penalty
            - length_penalty
            - weak_body_penalty
        )
        if score > best_score:
            best_score = score
            best_tail = tail_time

    return int(best_tail)


def _sustain_segment_bonus(sustain_segments: List[Dict[str, Any]], head_time: int, tail_time: int) -> float:
    best = 0.0
    for segment in sustain_segments:
        try:
            start = int(segment.get("start_ms", 0))
            end = int(segment.get("end_ms", 0))
            score = float(segment.get("score", 0.0))
            vocal = float(segment.get("vocal", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue
        if start - 120 <= head_time <= end + 80 and start < tail_time <= end + 140:
            best = max(best, min(1.0, score * 0.72 + vocal * 0.40))
    return best


def _ln_has_sustain_reason(
    head_time: int,
    tail_time: int,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> bool:
    head = _music_entry(analysis, snap_points, accent_snap_points, head_time)
    tail = _music_entry(analysis, snap_points, accent_snap_points, tail_time)
    body_sustain = _analysis_curve_average(analysis, "sustain_curve", head_time, tail_time, tail.get("sustain", 0.0))
    body_vocal = _analysis_curve_average(analysis, "vocal_sustain_curve", head_time, tail_time, tail.get("vocal", 0.0))
    body_reason = max(body_sustain, body_vocal * 1.10)
    release_reason = tail.get("release", 0.0)
    segment_reason = _sustain_segment_bonus(analysis.get("sustain_segments", []), head_time, tail_time)
    attack_reason = max(head.get("onset", 0.0), head.get("kick", 0.0), head.get("accent", 0.0))
    if body_reason >= 0.13 or release_reason >= 0.22 or segment_reason >= 0.16:
        return True
    return attack_reason >= 0.72 and tail_time - head_time <= 170


def _limit_ln_tail_for_music_hits(
    head_time: int,
    tail_time: int,
    lane: int,
    accepted_lns: List[Tuple[int, int, int]],
    min_ln_ms: int,
    tail_gap: int,
    config: DifficultyConfig,
    context: Dict[int, Dict[str, float]],
    snap_points: List[int],
    target: float,
) -> int:
    max_chord_size = clamp_max_chord_size(config)
    start_idx = bisect.bisect_right(snap_points, head_time + tail_gap)
    end_idx = bisect.bisect_left(snap_points, tail_time)
    for time_ms in snap_points[start_idx:end_idx]:
        entry = context.get(time_ms)
        if not entry:
            continue
        required = _ln_required_hits_at(entry, config, target, max_chord_size)
        if required <= 0:
            continue
        active_lanes = {
            active_lane
            for start, end, active_lane in accepted_lns
            if start < time_ms <= end + tail_gap
        }
        if head_time < time_ms <= tail_time + tail_gap:
            active_lanes.add(lane)
        free_lane_count = 4 - len(active_lanes)
        if free_lane_count >= required:
            continue
        trimmed = _previous_snap_at_or_before(snap_points, time_ms - tail_gap)
        if trimmed is None or trimmed - head_time < min_ln_ms:
            return head_time
        return min(tail_time, trimmed)
    return tail_time


def _restore_music_hits_after_ln_refine(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    context: Dict[int, Dict[str, float]],
    target: float,
    max_chord_size: int,
    tail_gap: int,
) -> List[NoteObject]:
    rows: Dict[int, Dict[int, NoteObject]] = {}
    lns: List[NoteObject] = []
    for note in notes:
        rows.setdefault(note.time_ms, {})[note.lane] = note
        if note.is_ln and note.end_time_ms is not None:
            lns.append(note)

    for time_ms in snap_points:
        entry = context.get(time_ms)
        if not entry or entry.get("silent", 0.0) > 0.0:
            continue
        required = _ln_required_hits_at(entry, config, target, max_chord_size)
        if required <= 0:
            continue

        row = rows.setdefault(time_ms, {})
        required = min(required, max_chord_size)
        if len(row) >= required or len(row) >= max_chord_size:
            continue

        active_lanes = {
            note.lane
            for note in lns
            if note.end_time_ms is not None and note.time_ms < time_ms <= note.end_time_ms + tail_gap
        }
        available = [
            lane
            for lane in [0, 1, 2, 3]
            if lane not in row and lane not in active_lanes
        ]
        while available and len(row) < required and len(row) < max_chord_size:
            lane = _pick_restored_music_hit_lane(time_ms, available, entry, row)
            row[lane] = NoteObject(time_ms=time_ms, lane=lane)
            available = [candidate for candidate in available if candidate != lane]

    restored: List[NoteObject] = []
    for time_ms in sorted(rows):
        for lane in sorted(rows[time_ms]):
            restored.append(rows[time_ms][lane])
    return restored


def _release_ln_blocks_for_music_hits(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    context: Dict[int, Dict[str, float]],
    target: float,
    max_chord_size: int,
    min_ln_ms: int,
    tail_gap: int,
) -> List[NoteObject]:
    adjusted = [NoteObject(note.time_ms, note.lane, note.end_time_ms) for note in notes]
    strong_times = [
        time_ms
        for time_ms in snap_points
        if _ln_required_hits_at(context.get(time_ms, {}), config, target, max_chord_size) > 0
    ]
    if not strong_times:
        return adjusted

    for time_ms in strong_times:
        required = _ln_required_hits_at(context.get(time_ms, {}), config, target, max_chord_size)
        for _ in range(4):
            active_indices = [
                index
                for index, note in enumerate(adjusted)
                if note.is_ln and note.end_time_ms is not None and note.time_ms < time_ms <= note.end_time_ms + tail_gap
            ]
            active_lanes = {adjusted[index].lane for index in active_indices}
            if len(active_lanes) <= 4 - required:
                break
            if not active_indices:
                break

            trim_index = sorted(
                active_indices,
                key=lambda index: (
                    _ln_keep_score(adjusted[index], analysis, snap_points, accent_snap_points),
                    -(time_ms - adjusted[index].time_ms),
                ),
            )[0]
            note = adjusted[trim_index]
            trimmed_tail = _previous_snap_at_or_before(snap_points, time_ms - tail_gap)
            if trimmed_tail is not None and trimmed_tail - note.time_ms >= min_ln_ms:
                adjusted[trim_index] = NoteObject(note.time_ms, note.lane, trimmed_tail)
            else:
                adjusted[trim_index] = NoteObject(note.time_ms, note.lane)

    return sorted(adjusted, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _ln_required_hits_at(
    entry: Dict[str, float],
    config: DifficultyConfig,
    target: float,
    max_chord_size: int,
) -> int:
    if entry.get("silent", 0.0) > 0.0:
        return 0
    if max_chord_size <= 1:
        return 1 if entry.get("protected", 0.0) > 0.0 or entry.get("accent", 0.0) >= 0.62 else 0

    strongest = max(entry.get("accent", 0.0), entry.get("kick", 0.0) * 1.05, entry.get("stack", 0.0) * 0.95)
    if strongest >= 0.88 and target >= 6.4 and max_chord_size >= 2:
        desired = 2
    elif strongest >= 0.66 or (entry.get("protected", 0.0) > 0.0 and entry.get("kick", 0.0) >= 0.48):
        desired = 1
    elif entry.get("onset", 0.0) >= 0.24 and entry.get("energy", 0.0) >= 0.40:
        desired = 1
    else:
        return 0

    if config.key_style == "speed":
        desired = min(desired, 2)
    return max(1, min(max_chord_size, desired))


def _pick_restored_music_hit_lane(
    time_ms: int,
    available: List[int],
    entry: Dict[str, float],
    row: Dict[int, NoteObject],
) -> int:
    if not available:
        return 0
    center_bias = entry.get("kick", 0.0) >= 0.58 or entry.get("stack", 0.0) >= 0.55
    return sorted(
        available,
        key=lambda lane: (
            abs(lane - 1.5) if center_bias else -abs(lane - 1.5),
            lane in row,
            _deterministic_index(101, time_ms, lane, int(entry.get("accent", 0.0) * 1000), 0xA91),
        ),
    )[0]


def _previous_snap_at_or_before(snap_points: List[int], time_ms: int) -> Optional[int]:
    idx = bisect.bisect_right(snap_points, time_ms)
    if idx <= 0:
        return None
    return snap_points[idx - 1]


def _rebalance_lns_for_target(
    notes: List[NoteObject],
    target: float,
    tolerance: float,
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> Tuple[List[NoteObject], float]:
    current_sr = DifficultyEstimator.estimate_sr(notes, analysis["duration_ms"])
    upper_bound = target + max(0.0, tolerance - 0.02)
    if config.chart_type not in ["ln", "hybrid"] or current_sr <= upper_bound or not any(note.is_ln for note in notes):
        return notes, current_sr

    working = list(notes)
    ln_indices = [index for index, note in enumerate(working) if note.is_ln and note.end_time_ms is not None]
    if not ln_indices:
        return notes, current_sr

    ranked = sorted(
        ln_indices,
        key=lambda index: _ln_keep_score(working[index], analysis, snap_points, accent_snap_points),
    )
    best_notes = working
    best_sr = current_sr
    best_diff = abs(current_sr - target)
    batch_size = max(1, min(24, len(ranked) // 18 or 1))
    min_keep_lns = _minimum_lns_to_keep(notes, config)
    converted_lns = 0

    for offset in range(0, len(ranked), batch_size):
        remaining_after_full_batch = len(ranked) - converted_lns - batch_size
        if remaining_after_full_batch < min_keep_lns:
            batch_size = max(0, len(ranked) - converted_lns - min_keep_lns)
        if batch_size <= 0:
            break

        for index in ranked[offset:offset + batch_size]:
            note = working[index]
            working[index] = NoteObject(time_ms=note.time_ms, lane=note.lane)
            converted_lns += 1

        candidate = Validator.validate_and_fix(working, config, analysis["silent_regions"], snap_points)
        context = _music_context(analysis, snap_points, accent_snap_points)
        beat_length = 60000.0 / max(1.0, analysis["bpm"])
        tail_gap = max(34, int(min(58, beat_length / 12.0)))
        min_ln_ms = max(30, int(config.min_ln_ms))
        if config.key_style in ["stream", "speed", "tech"]:
            min_ln_ms = max(30, min(min_ln_ms, int(max(45, beat_length / 8.0))))
        candidate = _release_ln_blocks_for_music_hits(
            candidate,
            config,
            analysis,
            snap_points,
            accent_snap_points,
            context,
            target,
            clamp_max_chord_size(config),
            min_ln_ms,
            tail_gap,
        )
        candidate = _restore_music_hits_after_ln_refine(
            candidate,
            config,
            analysis,
            snap_points,
            accent_snap_points,
            context,
            target,
            clamp_max_chord_size(config),
            tail_gap,
        )
        candidate = _release_ln_blocks_for_music_hits(
            candidate,
            config,
            analysis,
            snap_points,
            accent_snap_points,
            context,
            target,
            clamp_max_chord_size(config),
            min_ln_ms,
            tail_gap,
        )
        candidate = Validator.validate_and_fix(candidate, config, analysis["silent_regions"], snap_points)
        candidate_sr = DifficultyEstimator.estimate_sr(candidate, analysis["duration_ms"])
        candidate_diff = abs(candidate_sr - target)
        if candidate_diff < best_diff:
            best_notes = candidate
            best_sr = candidate_sr
            best_diff = candidate_diff
        if candidate_sr <= upper_bound:
            return candidate, candidate_sr

    if best_sr > upper_bound:
        best_notes, best_sr = _trim_low_music_notes_for_target(
            best_notes,
            target,
            tolerance,
            config,
            analysis,
            snap_points,
            accent_snap_points,
            best_sr,
        )

    return best_notes, best_sr


def _minimum_lns_to_keep(notes: List[NoteObject], config: DifficultyConfig) -> int:
    ln_count = sum(1 for note in notes if note.is_ln)
    if ln_count <= 0:
        return 0
    if config.chart_type == "hybrid":
        ratio_floor = 0.20
    elif config.key_style == "speed":
        ratio_floor = 0.10
    else:
        ratio_floor = 0.16
    requested_floor = max(0.06, min(0.28, config.ln_ratio * 0.32))
    ratio_floor = max(ratio_floor, requested_floor)
    return min(ln_count, max(24, int(round(len(notes) * ratio_floor))))


def _ln_keep_score(
    note: NoteObject,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> float:
    if not note.is_ln or note.end_time_ms is None:
        return 1.0
    head = _music_entry(analysis, snap_points, accent_snap_points, note.time_ms)
    tail = _music_entry(analysis, snap_points, accent_snap_points, note.end_time_ms)
    body_sustain = _analysis_curve_average(analysis, "sustain_curve", note.time_ms, note.end_time_ms, tail.get("sustain", 0.0))
    body_vocal = _analysis_curve_average(analysis, "vocal_sustain_curve", note.time_ms, note.end_time_ms, tail.get("vocal", 0.0))
    segment = _sustain_segment_bonus(analysis.get("sustain_segments", []), note.time_ms, note.end_time_ms)
    return (
        max(body_sustain, body_vocal * 1.10) * 0.42
        + tail.get("release", 0.0) * 0.32
        + head.get("accent", 0.0) * 0.12
        + head.get("kick", 0.0) * 0.08
        + segment * 0.20
    )


def _trim_low_music_notes_for_target(
    notes: List[NoteObject],
    target: float,
    tolerance: float,
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    current_sr: float,
) -> Tuple[List[NoteObject], float]:
    upper_bound = target + max(0.0, tolerance - 0.02)
    if current_sr <= upper_bound or not notes:
        return notes, current_sr

    working = list(notes)
    best_notes = working
    best_sr = current_sr
    best_diff = abs(current_sr - target)
    max_chord_size = clamp_max_chord_size(config)

    for _ in range(80):
        rows: Dict[int, List[int]] = {}
        for index, note in enumerate(working):
            rows.setdefault(note.time_ms, []).append(index)

        candidates: List[Tuple[float, int]] = []
        for time_ms, indexes in rows.items():
            entry = _music_entry(analysis, snap_points, accent_snap_points, time_ms)
            if entry.get("silent", 0.0) > 0.0:
                continue
            row_size = len(indexes)
            if row_size <= 1 and (entry.get("score", 0.0) >= 0.18 or entry.get("onset", 0.0) >= 0.12):
                continue
            for index in indexes:
                note = working[index]
                if note.is_ln:
                    continue
                score = _note_keep_score_for_trim(note, row_size, entry, max_chord_size)
                candidates.append((score, index))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0])
        batch = max(1, min(12, len(candidates) // 24 or 1))
        remove_indices = {index for _, index in candidates[:batch]}
        candidate_notes = [note for index, note in enumerate(working) if index not in remove_indices]
        candidate_notes = Validator.validate_and_fix(candidate_notes, config, analysis["silent_regions"], snap_points)
        context = _music_context(analysis, snap_points, accent_snap_points)
        beat_length = 60000.0 / max(1.0, analysis["bpm"])
        tail_gap = max(34, int(min(58, beat_length / 12.0)))
        min_ln_ms = max(30, int(config.min_ln_ms))
        if config.key_style in ["stream", "speed", "tech"]:
            min_ln_ms = max(30, min(min_ln_ms, int(max(45, beat_length / 8.0))))
        candidate_notes = _release_ln_blocks_for_music_hits(
            candidate_notes,
            config,
            analysis,
            snap_points,
            accent_snap_points,
            context,
            target,
            clamp_max_chord_size(config),
            min_ln_ms,
            tail_gap,
        )
        candidate_notes = Validator.validate_and_fix(candidate_notes, config, analysis["silent_regions"], snap_points)
        candidate_sr = DifficultyEstimator.estimate_sr(candidate_notes, analysis["duration_ms"])
        candidate_diff = abs(candidate_sr - target)
        if candidate_diff < best_diff:
            best_notes = candidate_notes
            best_sr = candidate_sr
            best_diff = candidate_diff
        if candidate_sr <= upper_bound:
            return candidate_notes, candidate_sr
        if candidate_sr >= current_sr and batch == 1:
            break
        working = candidate_notes
        current_sr = candidate_sr

    return best_notes, best_sr


def _note_keep_score_for_trim(
    note: NoteObject,
    row_size: int,
    entry: Dict[str, float],
    max_chord_size: int,
) -> float:
    row_weight = 0.0 if row_size > 1 else 4.0
    chord_weight = -0.35 if row_size >= max(2, max_chord_size - 1) else 0.0
    return (
        entry.get("accent", 0.0) * 2.0
        + entry.get("kick", 0.0) * 1.4
        + entry.get("onset", 0.0) * 1.1
        + entry.get("protected", 0.0) * 2.2
        + row_weight
        + chord_weight
        + (0.18 if note.lane in [1, 2] else 0.0)
    )


def _should_place_safe_ln(row_index: int, lane: int, interval: int, style: str) -> bool:
    salt = 7 if style == "stream" else 3
    return ((row_index + lane * 2 + salt) % interval) == 0


def _safe_ln_length(
    min_ln_ms: int,
    max_ln_ms: int,
    beat_length: float,
    row_index: int,
    lane: int,
    style: str,
) -> int:
    if max_ln_ms <= min_ln_ms:
        return min_ln_ms

    preferred = [
        int(round(beat_length * 0.50)),
        int(round(beat_length * 0.75)),
        int(round(beat_length)),
        int(round(beat_length * 1.50)),
    ]
    if style == "jack":
        preferred = [
            int(round(beat_length * 0.50)),
            int(round(beat_length * 0.75)),
            int(round(beat_length)),
        ]

    valid = [length for length in preferred if min_ln_ms <= length <= max_ln_ms]
    if not valid:
        return max(min_ln_ms, min(max_ln_ms, int(round(beat_length * 0.50))))

    choice = _deterministic_index(len(valid), row_index, lane, int(beat_length), len(valid))
    return valid[choice]


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

    if temperature >= 0.75:
        return _spread_pick(
            eligible,
            promote_count,
            temperature,
            salt=next_size * 193 + promote_count,
        )

    eligible_set = set(eligible)
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
        group = [index for index in group if index in eligible_set and index not in selected_set]
        if not group or remaining <= 0:
            continue

        group.sort()
        if remaining >= len(group):
            picks = group
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
    phrase_index = -1
    next_phrase_start = 0

    for index, (time_ms, size) in enumerate(zip(rows, sizes)):
        phrase_boundary = index >= next_phrase_start
        if phrase_boundary:
            phrase_index += 1
            phrase_len = _jack_stack_phrase_length(target, temperature, phrase_index)
            next_phrase_start = index + phrase_len
            current_lanes = _next_jack_stack_pattern(size, current_lanes, phrase_index, temperature)
        else:
            current_lanes = _resize_jack_stack(current_lanes, size, phrase_index, temperature)
            current_lanes = _vary_jack_stack_inside_phrase(
                current_lanes,
                size,
                phrase_index,
                index,
                target,
                temperature,
            )

        for lane in current_lanes[:size]:
            notes.append(NoteObject(time_ms=time_ms, lane=lane))

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
            (patterns.index(lanes) - phrase_index) % len(patterns),
        ),
    )

    pool_size = 1 + int(round(max(0.0, min(1.0, temperature)) * (len(ranked) - 1)))
    pool = ranked[: max(1, pool_size)]
    choice_index = _deterministic_index(
        len(pool),
        phrase_index,
        size,
        int(temperature * 100),
        sum(current_lanes) if current_lanes else 0,
    )
    return list(pool[choice_index])


def _vary_jack_stack_inside_phrase(
    current_lanes: List[int],
    size: int,
    phrase_index: int,
    row_index: int,
    target: float,
    temperature: float,
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

    choice = _deterministic_index(
        len(candidates),
        phrase_index,
        row_index,
        size,
        int(temperature * 1000),
        sum(current_lanes),
    )
    return list(candidates[choice])


def _deterministic_index(modulo: int, *parts: int) -> int:
    if modulo <= 0:
        return 0
    value = 0x345678
    for part in parts:
        value = ((value ^ int(part)) * 1000003) & 0x7FFFFFFF
    return value % modulo


def _resize_jack_stack(
    current_lanes: List[int],
    size: int,
    phrase_index: int,
    temperature: float = 0.35,
) -> List[int]:
    current = [lane for lane in current_lanes if lane in [0, 1, 2, 3]]
    if len(current) == size:
        return current
    if not current:
        return _next_jack_stack_pattern(size, [], phrase_index, temperature)
    if size == 4:
        return [0, 1, 2, 3]
    if len(current) > size:
        if len(current) == 4:
            return _next_jack_stack_pattern(size, current, phrase_index, temperature)
        return current[:size]

    target_pattern = _next_jack_stack_pattern(size, current, phrase_index, temperature)
    expanded = list(current)
    for lane in target_pattern:
        if lane not in expanded:
            expanded.append(lane)
        if len(expanded) >= size:
            break
    for lane in [0, 1, 2, 3]:
        if lane not in expanded:
            expanded.append(lane)
        if len(expanded) >= size:
            break
    return expanded[:size]


def _jack_stack_patterns(size: int) -> List[List[int]]:
    if size <= 1:
        return [[0], [1], [2], [3]]
    if size == 2:
        return [[0, 1], [1, 2], [2, 3], [0, 3], [0, 2], [1, 3]]
    if size == 3:
        return [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
    return [[0, 1, 2, 3]]


def _time_in_regions(time_ms: int, regions: List[Tuple[int, int]]) -> bool:
    return any(start <= time_ms <= end for start, end in regions)


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


def _fill_reasonable_gaps(
    notes: List[NoteObject],
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    config: DifficultyConfig,
) -> List[NoteObject]:
    if not notes:
        return notes

    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    active_style = config.key_style or ("hybrid" if config.chart_type == "hybrid" else "tech")
    if active_style == "jack":
        gap_threshold = int(beat_length * 0.45)
    elif active_style == "stream":
        gap_threshold = int(beat_length * 1.55)
    elif active_style == "speed":
        gap_threshold = int(beat_length * 1.75)
    elif active_style == "hybrid":
        gap_threshold = int(beat_length * 1.20)
    else:
        gap_threshold = int(beat_length * 1.95)

    snap_set = set(snap_points)
    accent_times = sorted(t for t in accent_snap_points if t in snap_set)
    all_snap_times = sorted(snap_set)
    additions: List[NoteObject] = []
    last_lane_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
    for note in sorted(notes, key=lambda n: (n.time_ms, n.lane)):
        last_lane_time[note.lane] = note.time_ms

    sorted_times = sorted(rows)
    for index, (prev_time, next_time) in enumerate(zip(sorted_times, sorted_times[1:])):
        gap = next_time - prev_time
        if gap < gap_threshold:
            continue
        prev_interval = prev_time - sorted_times[index - 1] if index > 0 else None
        next_interval = sorted_times[index + 2] - next_time if (index + 2) < len(sorted_times) else None
        context_active = (
            prev_interval is not None
            and next_interval is not None
            and prev_interval <= int(beat_length * 0.80)
            and next_interval <= int(beat_length * 0.80)
        )
        if _gap_has_reason(prev_time, next_time, analysis, config, active_style, context_active):
            continue

        if active_style == "jack":
            between = [t for t in all_snap_times if prev_time < t < next_time]
        else:
            between = [t for t in accent_times if prev_time < t < next_time]
            if not between:
                between = [t for t in all_snap_times if prev_time < t < next_time]
        if not between:
            continue

        shared = sorted(rows[prev_time] & rows[next_time])
        lane = _choose_gap_lane(rows[prev_time], rows[next_time], shared, last_lane_time, active_style, config)

        if active_style == "jack":
            chosen_times = list(between)
        else:
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
            last_lane_time[lane] = time_ms

    if not additions:
        return notes

    return notes + additions


def _gap_has_reason(
    prev_time: int,
    next_time: int,
    analysis: Dict[str, Any],
    config: DifficultyConfig,
    active_style: str,
    context_active: bool,
) -> bool:
    for start, end in analysis.get("silent_regions", []):
        if max(prev_time, start) < min(next_time, end):
            return True

    center_time = (prev_time + next_time) // 2
    energy_score = _energy_score_at(analysis, center_time)
    if active_style == "jack":
        return False
    if active_style == "stream":
        return energy_score < 0.28
    if active_style == "speed":
        return energy_score < 0.24
    if active_style == "hybrid":
        return False
    return energy_score < 0.30


def _choose_gap_lane(
    prev_lanes: Set[int],
    next_lanes: Set[int],
    shared: List[int],
    last_lane_time: Dict[int, int],
    active_style: str,
    config: DifficultyConfig,
) -> int:
    ordered = sorted(range(4), key=lambda lane: (last_lane_time.get(lane, -999999), lane))
    if active_style == "jack" and config.target_star is not None and config.target_star <= 3.5:
        return ordered[0]

    if shared:
        return shared[0]

    if active_style == "stream":
        stream_order = [lane for lane in ordered if lane not in prev_lanes]
        if stream_order:
            return stream_order[0]
    if active_style == "jack":
        jack_order = [lane for lane in ordered if lane in prev_lanes or lane in next_lanes]
        if jack_order:
            return jack_order[0]
    return ordered[0]


def _music_influence(config: DifficultyConfig) -> float:
    value = getattr(config, "music_influence", 0.65)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.65


def _music_context(
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> Dict[int, Dict[str, float]]:
    if not snap_points:
        return {}

    sample_step = max(1, len(snap_points) // 32)
    snap_fingerprint = sum(snap_points[::sample_step]) % 1000003
    accent_fingerprint = sum(accent_snap_points) % 1000003
    cache_key = (
        len(snap_points),
        snap_points[0],
        snap_points[-1],
        snap_fingerprint,
        accent_fingerprint,
        len(analysis.get("onset_peaks", [])),
        len(accent_snap_points),
        len(analysis.get("sustain_curve", [])),
        len(analysis.get("vocal_sustain_curve", [])),
        len(analysis.get("release_curve", [])),
        len(analysis.get("release_points_ms", [])),
    )
    cache = analysis.setdefault("_music_context_cache", {})
    if cache.get("key") == cache_key:
        return cache.get("value", {})

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    max_distance = max(35, min(70, int(beat_length * 0.12)))
    snap_set = set(snap_points)
    beat_times = [int(time_ms) for time_ms in analysis.get("beat_times_ms", [])]

    energy_curve = [float(value) for value in analysis.get("energy_curve", [])]
    if energy_curve and analysis["duration_ms"] > 0:
        sorted_energy = sorted(energy_curve)
        low = sorted_energy[min(len(sorted_energy) - 1, int(len(sorted_energy) * 0.25))]
        high = sorted_energy[min(len(sorted_energy) - 1, int(len(sorted_energy) * 0.90))]
    else:
        low = 0.0
        high = 1.0

    def energy_at(time_ms: int) -> float:
        if not energy_curve or analysis["duration_ms"] <= 0 or high <= low:
            return 0.5
        idx = int((time_ms / analysis["duration_ms"]) * (len(energy_curve) - 1))
        idx = max(0, min(len(energy_curve) - 1, idx))
        return max(0.0, min(1.0, (energy_curve[idx] - low) / (high - low)))

    onset_by_snap: Dict[int, float] = {}
    bass_by_snap: Dict[int, float] = {}
    stack_by_snap: Dict[int, int] = {}

    raw_peaks = analysis.get("onset_peaks") or [
        {"time_ms": int(time_ms), "strength": 0.72, "bass": 0.0}
        for time_ms in analysis.get("onset_times_ms", [])
    ]
    for peak in raw_peaks:
        if isinstance(peak, dict):
            raw_time = int(peak.get("time_ms", 0))
            strength = float(peak.get("strength", 0.0))
            bass = float(peak.get("bass", 0.0))
        else:
            raw_time = int(peak[0])
            strength = float(peak[1]) if len(peak) > 1 else 0.72
            bass = float(peak[2]) if len(peak) > 2 else 0.0

        nearest = _nearest_snap_point(snap_points, raw_time)
        if nearest is None:
            continue
        distance = abs(nearest - raw_time)
        if distance > max_distance:
            continue

        distance_weight = max(0.0, 1.0 - (distance / max(1.0, max_distance)))
        onset_by_snap[nearest] = max(onset_by_snap.get(nearest, 0.0), max(0.0, min(1.0, strength)) * distance_weight)
        bass_by_snap[nearest] = max(bass_by_snap.get(nearest, 0.0), max(0.0, min(1.0, bass)) * distance_weight)
        stack_by_snap[nearest] = stack_by_snap.get(nearest, 0) + 1

    context: Dict[int, Dict[str, float]] = {}
    for time_ms in snap_points:
        silent = 1.0 if _time_in_regions(time_ms, analysis.get("silent_regions", [])) else 0.0
        onset = onset_by_snap.get(time_ms, 0.0)
        bass = bass_by_snap.get(time_ms, 0.0)
        stack = max(0.0, min(1.0, stack_by_snap.get(time_ms, 0) / 3.0))
        energy = energy_at(time_ms)
        sustain = _analysis_curve_value(analysis, "sustain_curve", time_ms, 0.0)
        vocal = _analysis_curve_value(analysis, "vocal_sustain_curve", time_ms, 0.0)
        release = _analysis_curve_value(analysis, "release_curve", time_ms, 0.0)

        near_beat = _nearest_distance(beat_times, time_ms)
        if near_beat is not None and near_beat <= max_distance:
            beat_score = max(0.0, 1.0 - (near_beat / max(1.0, max_distance)))
        else:
            beat_score = 0.0

        phrase_score = 0.0
        if beat_score > 0.0 and beat_times:
            idx = bisect.bisect_left(beat_times, time_ms)
            beat_candidates = []
            if idx < len(beat_times):
                beat_candidates.append(idx)
            if idx > 0:
                beat_candidates.append(idx - 1)
            if beat_candidates:
                beat_idx = min(beat_candidates, key=lambda candidate: abs(beat_times[candidate] - time_ms))
                if beat_idx % 4 == 0:
                    phrase_score = 1.0
                elif beat_idx % 2 == 0:
                    phrase_score = 0.45

        music_score = (
            onset * 0.34
            + bass * 0.24
            + energy * 0.18
            + beat_score * 0.12
            + stack * 0.08
            + phrase_score * 0.04
            + max(sustain, vocal) * 0.05
        )
        if time_ms in accent_snap_points:
            music_score = max(music_score, 0.35 + energy * 0.20 + bass * 0.12)
        if silent:
            music_score = 0.0

        accent_score = max(music_score, bass * 0.78 + onset * 0.22, stack * 0.68 + energy * 0.20)
        protected = 1.0 if (
            music_score >= 0.58
            or (bass >= 0.50 and onset >= 0.20)
            or (stack >= 0.68 and energy >= 0.42)
        ) and not silent else 0.0

        context[time_ms] = {
            "score": max(0.0, min(1.0, music_score)),
            "accent": max(0.0, min(1.0, accent_score)),
            "kick": max(0.0, min(1.0, bass)),
            "energy": max(0.0, min(1.0, energy)),
            "onset": max(0.0, min(1.0, onset)),
            "stack": max(0.0, min(1.0, stack)),
            "beat": max(0.0, min(1.0, beat_score)),
            "sustain": max(0.0, min(1.0, sustain)),
            "vocal": max(0.0, min(1.0, vocal)),
            "release": max(0.0, min(1.0, release)),
            "protected": protected,
            "silent": silent,
        }

    cache["key"] = cache_key
    cache["value"] = context
    return context


def _music_entry(
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    time_ms: int,
) -> Dict[str, float]:
    context = _music_context(analysis, snap_points, accent_snap_points)
    if time_ms in context:
        return context[time_ms]

    energy = _energy_score_at(analysis, time_ms)
    return {
        "score": energy,
        "accent": energy,
        "kick": 0.0,
        "energy": energy,
        "onset": 0.0,
        "stack": 0.0,
        "beat": 0.0,
        "sustain": _analysis_curve_value(analysis, "sustain_curve", time_ms, 0.0),
        "vocal": _analysis_curve_value(analysis, "vocal_sustain_curve", time_ms, 0.0),
        "release": _analysis_curve_value(analysis, "release_curve", time_ms, 0.0),
        "protected": 0.0,
        "silent": 0.0,
    }


def _apply_music_influence(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    active_style: Optional[str],
) -> List[NoteObject]:
    influence = _music_influence(config)
    if influence <= 0.01 or not notes:
        return notes

    adjusted = _ensure_music_anchor_notes(notes, config, analysis, snap_points, accent_snap_points, active_style, influence)
    adjusted = _align_rows_to_music_anchors(adjusted, config, analysis, snap_points, accent_snap_points, active_style, influence)
    adjusted = _fortify_music_anchor_rows(adjusted, config, analysis, snap_points, accent_snap_points, active_style, influence)
    adjusted = _retarget_chords_to_music(adjusted, config, analysis, snap_points, accent_snap_points, active_style, influence)
    adjusted = _shape_music_dynamics(adjusted, config, analysis, snap_points, accent_snap_points, active_style, influence)
    adjusted = _fortify_music_anchor_rows(adjusted, config, analysis, snap_points, accent_snap_points, active_style, influence)
    return adjusted


def _align_rows_to_music_anchors(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    active_style: Optional[str],
    influence: float,
) -> List[NoteObject]:
    if active_style not in ["stream", "tech", "speed"] or influence < 0.45 or not notes:
        return notes

    context = _music_context(analysis, snap_points, accent_snap_points)
    if not context:
        return notes

    snap_set = set(snap_points)
    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    row_times = sorted(rows)
    if len(row_times) < 3:
        return notes

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    max_moves = max(1, int(round(len(row_times) * _anchor_align_ratio(active_style) * influence)))
    threshold = _anchor_align_threshold(active_style, influence)
    min_gap = _anchor_align_min_gap_ms(active_style, beat_length, config.target_star or 4.0, config.chart_type)
    moves = 0

    anchors = sorted(
        (
            (time_ms, entry)
            for time_ms, entry in context.items()
            if time_ms in snap_set
            and entry["silent"] <= 0.0
            and entry["accent"] >= threshold
            and (entry["protected"] > 0.0 or entry["kick"] >= 0.52 or entry["score"] >= threshold - 0.06)
        ),
        key=lambda item: (-item[1]["accent"], -item[1]["score"], item[0]),
    )

    for anchor_time, entry in anchors:
        if moves >= max_moves:
            break
        if anchor_time in rows:
            continue

        nearest = _nearest_music_row_time(row_times, anchor_time, active_style, beat_length)
        if nearest is None or nearest == anchor_time:
            continue
        distance = abs(nearest - anchor_time)
        if distance < 14:
            continue
        moving_notes = rows.get(nearest, [])
        if not moving_notes:
            continue
        if any(note.is_ln for note in moving_notes) and active_style == "tech":
            continue

        idx = bisect.bisect_left(row_times, nearest)
        prev_time = row_times[idx - 1] if idx > 0 else None
        next_time = row_times[idx + 1] if idx + 1 < len(row_times) else None
        if prev_time is not None and anchor_time - prev_time < min_gap:
            if not _drop_weak_close_anchor_neighbor(rows, row_times, prev_time, anchor_time, entry, analysis, snap_points, accent_snap_points):
                continue
            idx = bisect.bisect_left(row_times, nearest)
            prev_time = row_times[idx - 1] if idx > 0 else None
            next_time = row_times[idx + 1] if idx + 1 < len(row_times) else None
        if next_time is not None and next_time - anchor_time < min_gap:
            if not _drop_weak_close_anchor_neighbor(rows, row_times, next_time, anchor_time, entry, analysis, snap_points, accent_snap_points):
                continue
            idx = bisect.bisect_left(row_times, nearest)

        delta = anchor_time - nearest
        moved = [
            NoteObject(
                time_ms=anchor_time,
                lane=note.lane,
                end_time_ms=(note.end_time_ms + delta if note.end_time_ms is not None else None),
            )
            for note in moving_notes
            if note.end_time_ms is None or note.end_time_ms + delta > anchor_time
        ]
        if not moved:
            continue
        del rows[nearest]
        rows[anchor_time] = moved
        row_times.pop(idx)
        bisect.insort(row_times, anchor_time)
        moves += 1

    if moves <= 0:
        return notes
    return [
        note
        for time_ms in sorted(rows)
        for note in sorted(rows[time_ms], key=lambda item: (item.lane, item.end_time_ms or -1))
    ]


def _anchor_align_threshold(active_style: Optional[str], influence: float) -> float:
    if active_style == "stream":
        base = 0.58
    elif active_style == "tech":
        base = 0.60
    elif active_style == "speed":
        base = 0.66
    else:
        base = 0.62
    return max(0.44, base - influence * 0.06)


def _anchor_align_ratio(active_style: Optional[str]) -> float:
    if active_style == "stream":
        return 0.030
    if active_style == "tech":
        return 0.024
    if active_style == "speed":
        return 0.014
    return 0.018


def _anchor_align_min_gap_ms(active_style: Optional[str], beat_length: float, target: float, chart_type: str = "rice") -> int:
    if active_style == "stream":
        if chart_type == "ln":
            return max(32, int(beat_length / (13.5 if target >= 5.0 else 12.0)))
        return max(52, int(beat_length / (8.6 if target >= 5.0 else 8.0)))
    if active_style == "tech":
        return max(34, int(beat_length / (14.0 if target >= 5.5 else 12.0)))
    if active_style == "speed":
        return max(28, int(beat_length / 15.0))
    return max(36, int(beat_length / 12.0))


def _drop_weak_close_anchor_neighbor(
    rows: Dict[int, List[NoteObject]],
    row_times: List[int],
    neighbor_time: int,
    anchor_time: int,
    anchor_entry: Dict[str, float],
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> bool:
    neighbor_notes = rows.get(neighbor_time, [])
    if not neighbor_notes or any(note.is_ln for note in neighbor_notes):
        return False

    neighbor_entry = _music_entry(analysis, snap_points, accent_snap_points, neighbor_time)
    if neighbor_entry["protected"] > 0.0:
        return False
    if neighbor_entry["accent"] >= anchor_entry["accent"] - 0.08 and neighbor_entry["score"] >= 0.34:
        return False

    del rows[neighbor_time]
    idx = bisect.bisect_left(row_times, neighbor_time)
    if idx < len(row_times) and row_times[idx] == neighbor_time:
        row_times.pop(idx)
    return True


def _fortify_music_anchor_rows(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    active_style: Optional[str],
    influence: float,
) -> List[NoteObject]:
    context = _music_context(analysis, snap_points, accent_snap_points)
    if not context or not notes:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    row_lanes: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
        row_lanes.setdefault(note.time_ms, set()).add(note.lane)

    row_times = sorted(row_lanes)
    if not row_times:
        return notes

    target = config.target_star or 4.0
    max_chord_size = clamp_max_chord_size(config, active_style)
    if max_chord_size <= 1:
        return notes

    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    threshold = _anchor_fortify_threshold(active_style, influence)
    max_upgrades = max(1, int(round(len(row_times) * _anchor_fortify_ratio(active_style) * influence)))
    additions: List[NoteObject] = []
    upgrades = 0

    anchors = sorted(
        (
            (time_ms, entry)
            for time_ms, entry in context.items()
            if entry["silent"] <= 0.0
            and entry["accent"] >= threshold
            and (entry["protected"] > 0.0 or entry["score"] >= threshold - 0.08 or entry["kick"] >= 0.52)
        ),
        key=lambda item: (-item[1]["accent"], -item[1]["score"], item[0]),
    )

    for anchor_time, entry in anchors:
        if upgrades >= max_upgrades:
            break

        target_time = _nearest_music_row_time(row_times, anchor_time, active_style, beat_length)
        if target_time is None:
            continue
        lanes_at_time = row_lanes.setdefault(target_time, set())
        desired_size = _music_desired_chord_size(entry, max_chord_size, active_style, target)
        if desired_size <= len(lanes_at_time):
            continue

        while len(lanes_at_time) < desired_size and upgrades < max_upgrades:
            lane = _music_receiver_lane_for_style(
                target_time,
                row_lanes,
                row_times,
                entry,
                max_chord_size,
                active_style,
                target,
                beat_length,
            )
            if lane is None:
                break
            lanes_at_time.add(lane)
            additions.append(NoteObject(time_ms=target_time, lane=lane))
            upgrades += 1

    if not additions:
        return notes
    return sorted(notes + additions, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _shape_music_dynamics(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    active_style: Optional[str],
    influence: float,
) -> List[NoteObject]:
    if influence < 0.20 or not notes:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    row_times = sorted(rows)
    if not row_times:
        return notes

    target = config.target_star or 4.0
    kept: List[NoteObject] = []
    previous_lanes: Set[int] = set()

    for row_index, time_ms in enumerate(row_times):
        row_notes = sorted(rows[time_ms], key=lambda note: (note.lane, note.end_time_ms or -1))
        entry = _music_entry(analysis, snap_points, accent_snap_points, time_ms)
        desired_size = _music_dynamic_desired_size(
            current_size=len(row_notes),
            entry=entry,
            active_style=active_style,
            target=target,
            influence=influence,
            row_index=row_index,
        )
        if desired_size >= len(row_notes):
            selected = row_notes
        else:
            selected = _select_dynamic_row_notes(row_notes, desired_size, previous_lanes, active_style, target)

        kept.extend(selected)
        previous_lanes = {note.lane for note in selected}

    return sorted(kept, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _anchor_fortify_threshold(active_style: Optional[str], influence: float) -> float:
    if active_style == "jack":
        base = 0.52
    elif active_style == "stream":
        base = 0.54
    elif active_style == "tech":
        base = 0.55
    elif active_style == "speed":
        base = 0.62
    else:
        base = 0.58
    return max(0.38, base - influence * 0.08)


def _anchor_fortify_ratio(active_style: Optional[str]) -> float:
    if active_style == "jack":
        return 0.075
    if active_style == "stream":
        return 0.060
    if active_style == "tech":
        return 0.080
    if active_style == "speed":
        return 0.026
    return 0.045


def _nearest_music_row_time(
    row_times: List[int],
    anchor_time: int,
    active_style: Optional[str],
    beat_length: float,
) -> Optional[int]:
    if not row_times:
        return None
    idx = bisect.bisect_left(row_times, anchor_time)
    candidates = []
    if idx < len(row_times):
        candidates.append(row_times[idx])
    if idx > 0:
        candidates.append(row_times[idx - 1])
    if not candidates:
        return None

    nearest = min(candidates, key=lambda time_ms: abs(time_ms - anchor_time))
    if active_style == "jack":
        max_distance = max(12, int(beat_length / 20.0))
    elif active_style == "stream":
        max_distance = max(34, int(beat_length / 11.0))
    elif active_style == "tech":
        max_distance = max(55, int(beat_length / 8.0))
    elif active_style == "speed":
        max_distance = max(26, int(beat_length / 14.0))
    else:
        max_distance = max(34, int(beat_length / 12.0))
    return nearest if abs(nearest - anchor_time) <= max_distance else None


def _music_receiver_lane_for_style(
    time_ms: int,
    row_lanes: Dict[int, Set[int]],
    row_times: List[int],
    entry: Dict[str, float],
    max_chord_size: int,
    active_style: Optional[str],
    target: float,
    beat_length: float,
) -> Optional[int]:
    lanes_at_time = row_lanes.get(time_ms, set())
    if len(lanes_at_time) >= max_chord_size:
        return None
    available = [lane for lane in [0, 1, 2, 3] if lane not in lanes_at_time]
    if not available:
        return None

    idx = bisect.bisect_left(row_times, time_ms)
    prev_time = row_times[idx - 1] if idx > 0 else None
    next_time = row_times[idx + 1] if idx + 1 < len(row_times) and row_times[idx] == time_ms else (
        row_times[idx] if idx < len(row_times) and row_times[idx] != time_ms else None
    )
    prev_lanes = row_lanes.get(prev_time, set()) if prev_time is not None else set()
    next_lanes = row_lanes.get(next_time, set()) if next_time is not None else set()

    if active_style == "stream":
        ranked = sorted(
            available,
            key=lambda lane: (
                lane in prev_lanes,
                lane in next_lanes,
                abs(lane - 1.5) if entry["kick"] < 0.62 else -abs(lane - 1.5),
                _deterministic_index(97, time_ms, lane, int(entry["accent"] * 1000)),
            ),
        )
        return ranked[0]

    if active_style == "tech":
        soft_window = _tech_short_jack_soft_window_ms(target)
        ranked = sorted(
            available,
            key=lambda lane: (
                prev_time is not None and lane in prev_lanes and time_ms - prev_time <= soft_window,
                next_time is not None and lane in next_lanes and next_time - time_ms <= soft_window,
                lane in prev_lanes,
                abs(lane - 1.5) if entry["kick"] < 0.62 else -abs(lane - 1.5),
                _deterministic_index(101, time_ms, lane, int(target * 100)),
            ),
        )
        return ranked[0]

    if active_style == "jack":
        sticky = [lane for lane in available if lane in prev_lanes or lane in next_lanes]
        if sticky:
            return sorted(sticky, key=lambda lane: (_deterministic_index(71, time_ms, lane), lane))[0]

    return _music_receiver_lane(time_ms, lanes_at_time, entry, max_chord_size)


def _music_dynamic_desired_size(
    current_size: int,
    entry: Dict[str, float],
    active_style: Optional[str],
    target: float,
    influence: float,
    row_index: int,
) -> int:
    if current_size <= 1 or entry["protected"] > 0.0:
        return current_size

    calm_score = max(entry["score"], entry["accent"] * 0.82, entry["kick"] * 0.92)
    if calm_score >= 0.42:
        return current_size

    roll = _deterministic_index(1000, row_index, int(target * 100), int(calm_score * 1000), current_size)
    demote_chance = int((0.32 + max(0.0, 0.36 - calm_score) * 1.25) * influence * 1000)
    if roll > min(940, demote_chance):
        return current_size

    if active_style == "jack":
        if calm_score < 0.16:
            return min(current_size, 2 if target >= 6.15 else 1)
        if calm_score < 0.30:
            return min(current_size, 2 if target >= 4.9 else 1)
        return min(current_size, 3)

    if active_style == "stream":
        if calm_score < 0.18:
            return 1
        if calm_score < 0.32:
            return min(current_size, 2)
        return current_size

    if active_style == "tech":
        if calm_score < 0.16:
            return 1
        if calm_score < 0.30:
            return min(current_size, 2)
        return current_size

    if active_style == "speed":
        return 1 if calm_score < 0.30 else min(current_size, 2)

    return min(current_size, 2 if calm_score < 0.26 else current_size)


def _select_dynamic_row_notes(
    row_notes: List[NoteObject],
    desired_size: int,
    previous_lanes: Set[int],
    active_style: Optional[str],
    target: float,
) -> List[NoteObject]:
    desired_size = max(1, min(len(row_notes), desired_size))
    if desired_size >= len(row_notes):
        return row_notes

    ln_notes = [note for note in row_notes if note.is_ln]
    rice_notes = [note for note in row_notes if not note.is_ln]
    if len(ln_notes) >= desired_size:
        pool = ln_notes
    else:
        pool = ln_notes + rice_notes

    if active_style == "jack" and previous_lanes:
        ranked = sorted(
            pool,
            key=lambda note: (
                note.lane not in previous_lanes,
                note.lane,
            ),
        )
    elif active_style in ["stream", "tech"]:
        ranked = sorted(
            pool,
            key=lambda note: (
                note.lane in previous_lanes,
                abs(note.lane - 1.5),
                note.lane,
            ),
        )
    else:
        ranked = sorted(pool, key=lambda note: (not note.is_ln, abs(note.lane - 1.5), note.lane))

    selected = ranked[:desired_size]
    return sorted(selected, key=lambda note: (note.lane, note.end_time_ms or -1))


def _ensure_music_anchor_notes(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    active_style: Optional[str],
    influence: float,
) -> List[NoteObject]:
    if influence < 0.35 or active_style == "jack":
        return notes

    context = _music_context(analysis, snap_points, accent_snap_points)
    if not context:
        return notes

    row_lanes: Dict[int, Set[int]] = {}
    for note in notes:
        row_lanes.setdefault(note.time_ms, set()).add(note.lane)

    row_times = sorted(row_lanes)
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    guard_ms = max(18, min(52, int(beat_length * 0.10)))
    if active_style == "tech":
        threshold = 0.58 - influence * 0.10
        addition_ratio = 0.032
    elif active_style == "speed":
        threshold = 0.62 - influence * 0.09
        addition_ratio = 0.018
    elif active_style == "stream":
        threshold = 0.64 - influence * 0.10
        addition_ratio = 0.018
    else:
        threshold = 0.78 - influence * 0.16
        addition_ratio = 0.018
    max_additions = max(1, int(round(len(row_times) * addition_ratio * influence)))
    max_chord_size = clamp_max_chord_size(config, active_style)
    additions: List[NoteObject] = []

    anchors = sorted(
        (
            (time_ms, entry)
            for time_ms, entry in context.items()
            if entry["score"] >= threshold and entry["silent"] <= 0.0
        ),
        key=lambda item: (-item[1]["score"], item[0]),
    )

    for time_ms, entry in anchors:
        if len(additions) >= max_additions:
            break
        if time_ms in row_lanes:
            continue
        nearest = _nearest_distance(row_times, time_ms)
        if nearest is not None and nearest <= guard_ms:
            continue

        prev_gap = _previous_row_gap(row_times, time_ms)
        next_gap = _next_row_gap(row_times, time_ms)
        min_gap = _music_anchor_min_gap_ms(active_style, beat_length)
        if (prev_gap is not None and prev_gap < min_gap) or (next_gap is not None and next_gap < min_gap):
            continue

        lane = _music_anchor_lane(time_ms, row_lanes, entry, max_chord_size)
        if lane is None:
            continue
        row_lanes[time_ms] = {lane}
        bisect.insort(row_times, time_ms)
        additions.append(NoteObject(time_ms=time_ms, lane=lane))

    if not additions:
        return notes
    return sorted(notes + additions, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _retarget_chords_to_music(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    active_style: Optional[str],
    influence: float,
) -> List[NoteObject]:
    max_chord_size = clamp_max_chord_size(config, active_style)
    if max_chord_size <= 1 or influence <= 0.05:
        return notes

    context = _music_context(analysis, snap_points, accent_snap_points)
    if not context:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    row_lanes: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
        row_lanes.setdefault(note.time_ms, set()).add(note.lane)

    target = config.target_star or 4.0
    donor_threshold = 0.48 + (1.0 - influence) * 0.08
    receiver_threshold = 0.56 - influence * 0.10
    min_delta = 0.12 - influence * 0.06
    move_limit = max(1, int(round(len(notes) * 0.075 * influence)))

    donors = sorted(
        (
            time_ms
            for time_ms, lane_set in row_lanes.items()
            if len(lane_set) > 1
            and _music_entry(analysis, snap_points, accent_snap_points, time_ms)["protected"] <= 0.0
            and _music_entry(analysis, snap_points, accent_snap_points, time_ms)["accent"] <= donor_threshold
        ),
        key=lambda time_ms: (
            _music_entry(analysis, snap_points, accent_snap_points, time_ms)["accent"],
            -len(row_lanes[time_ms]),
            time_ms,
        ),
    )
    receivers = sorted(
        (
            time_ms
            for time_ms, lane_set in row_lanes.items()
            if len(lane_set) < max_chord_size
            and _music_entry(analysis, snap_points, accent_snap_points, time_ms)["accent"] >= receiver_threshold
            and _music_desired_chord_size(
                _music_entry(analysis, snap_points, accent_snap_points, time_ms),
                max_chord_size,
                active_style,
                target,
            )
            > len(lane_set)
        ),
        key=lambda time_ms: (
            -_music_entry(analysis, snap_points, accent_snap_points, time_ms)["accent"],
            len(row_lanes[time_ms]),
            time_ms,
        ),
    )

    if not donors or not receivers:
        return notes

    removed: Set[int] = set()
    additions: List[NoteObject] = []
    donor_cursor = 0
    moves = 0
    row_times = sorted(row_lanes)
    beat_length = 60000.0 / max(1.0, analysis["bpm"])

    for receiver_time in receivers:
        if moves >= move_limit:
            break

        receiver_entry = _music_entry(analysis, snap_points, accent_snap_points, receiver_time)
        desired_size = _music_desired_chord_size(receiver_entry, max_chord_size, active_style, target)
        while len(row_lanes[receiver_time]) < desired_size and moves < move_limit:
            donor_time = _next_music_donor_time(
                donors,
                donor_cursor,
                rows,
                row_lanes,
                removed,
                receiver_entry,
                min_delta,
                analysis,
                snap_points,
                accent_snap_points,
            )
            if donor_time is None:
                break
            donor_cursor = max(donor_cursor, donors.index(donor_time))

            note_to_move = _pick_music_donor_note(rows[donor_time], removed)
            if note_to_move is None:
                donor_cursor += 1
                continue
            lane = _music_receiver_lane_for_style(
                receiver_time,
                row_lanes,
                row_times,
                receiver_entry,
                max_chord_size,
                active_style,
                target,
                beat_length,
            )
            if lane is None:
                break

            removed.add(id(note_to_move))
            row_lanes[donor_time].discard(note_to_move.lane)
            row_lanes[receiver_time].add(lane)
            additions.append(NoteObject(time_ms=receiver_time, lane=lane))
            moves += 1

    if moves <= 0:
        return notes

    kept = [note for note in notes if id(note) not in removed]
    return sorted(kept + additions, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _music_desired_chord_size(
    entry: Dict[str, float],
    max_chord_size: int,
    active_style: Optional[str],
    target: float,
) -> int:
    accent = entry["accent"]
    desired = 1
    if accent >= 0.48:
        desired = 2
    if max_chord_size >= 3 and (accent >= 0.70 or entry["kick"] >= 0.60 or entry["stack"] >= 0.68):
        desired = 3
    if (
        max_chord_size >= 4
        and active_style in ["jack", "tech"]
        and target >= 5.25
        and (entry["kick"] >= 0.76 or entry["stack"] >= 0.84 or entry["accent"] >= 0.86)
    ):
        desired = 4
    if active_style == "speed" and target < 6.25:
        desired = min(desired, 2)
    if active_style == "stream":
        desired = min(desired, 3)
    return max(1, min(max_chord_size, desired))


def _next_music_donor_time(
    donors: List[int],
    start_index: int,
    rows: Dict[int, List[NoteObject]],
    row_lanes: Dict[int, Set[int]],
    removed: Set[int],
    receiver_entry: Dict[str, float],
    min_delta: float,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> Optional[int]:
    for index in range(max(0, start_index), len(donors)):
        time_ms = donors[index]
        if len(row_lanes.get(time_ms, set())) <= 1:
            continue
        donor_entry = _music_entry(analysis, snap_points, accent_snap_points, time_ms)
        if receiver_entry["accent"] - donor_entry["accent"] < min_delta:
            continue
        if _pick_music_donor_note(rows.get(time_ms, []), removed) is None:
            continue
        return time_ms
    return None


def _pick_music_donor_note(notes: List[NoteObject], removed: Set[int]) -> Optional[NoteObject]:
    rice_notes = [note for note in notes if id(note) not in removed and not note.is_ln]
    if rice_notes:
        return sorted(rice_notes, key=lambda note: note.lane)[-1]
    return None


def _music_receiver_lane(
    time_ms: int,
    lanes_at_time: Set[int],
    entry: Dict[str, float],
    max_chord_size: int,
) -> Optional[int]:
    if len(lanes_at_time) >= max_chord_size:
        return None
    available = [lane for lane in [0, 1, 2, 3] if lane not in lanes_at_time]
    if not available:
        return None
    return sorted(
        available,
        key=lambda lane: (
            -abs(lane - 1.5) if entry["kick"] >= 0.65 else abs(lane - 1.5),
            _deterministic_index(97, time_ms, lane, int(entry["accent"] * 1000)),
        ),
    )[0]


def _music_anchor_lane(
    time_ms: int,
    row_lanes: Dict[int, Set[int]],
    entry: Dict[str, float],
    max_chord_size: int,
) -> Optional[int]:
    return _music_receiver_lane(time_ms, row_lanes.get(time_ms, set()), entry, max_chord_size)


def _music_anchor_min_gap_ms(active_style: Optional[str], beat_length: float) -> int:
    if active_style == "stream":
        return max(28, int(beat_length / 12.0))
    if active_style == "speed":
        return max(24, int(beat_length / 14.0))
    if active_style == "tech":
        return max(24, int(beat_length / 16.0))
    return max(35, int(beat_length / 10.0))


def _previous_row_gap(row_times: List[int], time_ms: int) -> Optional[int]:
    idx = bisect.bisect_left(row_times, time_ms)
    if idx <= 0:
        return None
    return time_ms - row_times[idx - 1]


def _next_row_gap(row_times: List[int], time_ms: int) -> Optional[int]:
    idx = bisect.bisect_right(row_times, time_ms)
    if idx >= len(row_times):
        return None
    return row_times[idx] - time_ms


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


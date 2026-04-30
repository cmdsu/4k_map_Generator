import math
from collections import Counter
from itertools import combinations
from typing import Dict, List, Set, Tuple

from ..core.models import NoteObject
from ..core.calibration_utils import _deterministic_index, _flatten_note_rows


def _rotate_jack_anchor_runs(
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
    if len(row_times) < 10:
        return notes

    anchor_limit = _jack_anchor_run_soft_limit(target, temperature)
    pattern_limit = max(4, anchor_limit // 2)
    lane_counts = Counter(note.lane for note in notes if 0 <= note.lane <= 3)
    running = [0, 0, 0, 0]
    previous_pattern: Tuple[int, ...] = tuple()
    pattern_run = 0
    changes = 0
    max_changes = max(1, int(round(len(row_times) * (0.10 + max(0.0, min(1.0, temperature)) * 0.06))))

    for row_pos, time_ms in enumerate(row_times):
        row_notes = sorted(rows.get(time_ms, []), key=lambda item: (item.lane, item.end_time_ms or -1))
        current_lanes = sorted({note.lane for note in row_notes if 0 <= note.lane <= 3})
        current_pattern = tuple(current_lanes)
        if not current_lanes:
            continue
        projected_pattern_run = pattern_run + 1 if current_pattern == previous_pattern else 1
        if any(note.is_ln for note in row_notes):
            current_set = set(current_lanes)
            for lane in range(4):
                if lane in current_set:
                    running[lane] += 1
                else:
                    running[lane] = 0
            if current_pattern == previous_pattern:
                pattern_run = projected_pattern_run
            else:
                previous_pattern = current_pattern
                pattern_run = 1
            continue

        overloaded = [
            lane
            for lane in current_lanes
            if running[lane] + 1 > anchor_limit
        ]
        repeated_pattern = projected_pattern_run > pattern_limit

        if changes < max_changes and len(current_lanes) < 4 and (overloaded or repeated_pattern):
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
            replacement = _choose_jack_anchor_rotation(
                current_lanes,
                prev_lanes,
                next_lanes,
                running,
                lane_counts,
                anchor_limit,
                row_pos,
                target,
                repeated_pattern,
            )
            if replacement != current_lanes:
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
                current_lanes = replacement
                current_pattern = tuple(current_lanes)
                projected_pattern_run = 1 if current_pattern != previous_pattern else min(projected_pattern_run, pattern_limit)
                changes += 1

        current_set = set(current_lanes)
        for lane in range(4):
            if lane in current_set:
                running[lane] += 1
            else:
                running[lane] = 0
        if current_pattern == previous_pattern:
            pattern_run = projected_pattern_run
        else:
            previous_pattern = current_pattern
            pattern_run = 1

    if changes <= 0:
        return notes
    return _flatten_note_rows(rows)


def _shape_jack_anchor_contrast(
    notes: List[NoteObject],
    target: float,
    temperature: float,
) -> List[NoteObject]:
    if not notes or target < 4.75:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    ln_intervals: List[Tuple[int, int, int]] = []
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
        if note.is_ln and note.end_time_ms is not None:
            ln_intervals.append((note.lane, note.time_ms, note.end_time_ms))
    if ln_intervals:
        return notes
    row_times = sorted(rows)
    if len(row_times) < 32:
        return notes

    window_size = _jack_anchor_contrast_window_size(target)
    light_sections = _jack_anchor_light_sections(len(row_times), window_size, target, temperature)
    if not light_sections:
        return notes

    lane_counts = Counter(note.lane for note in notes if 0 <= note.lane <= 3)
    changed = 0
    max_changes = max(1, int(round(len(row_times) * (0.18 + max(0.0, min(1.0, temperature)) * 0.08))))

    for section_index in light_sections:
        start = section_index * window_size
        end = min(len(row_times), start + window_size)
        if end - start < 8:
            continue
        section_sizes = [
            len({note.lane for note in rows[row_times[index]] if 0 <= note.lane <= 3})
            for index in range(start, end)
        ]
        average_size = sum(section_sizes) / max(1, len(section_sizes))
        if average_size < 1.12 and target < 4.75:
            continue

        local_previous: List[int] = []
        for row_pos in range(start, end):
            if changed >= max_changes:
                break
            time_ms = row_times[row_pos]
            row_notes = sorted(rows.get(time_ms, []), key=lambda item: (item.lane, item.end_time_ms or -1))
            current_lanes = sorted({note.lane for note in row_notes if 0 <= note.lane <= 3})
            if not current_lanes or len(current_lanes) >= 4:
                local_previous = current_lanes
                continue
            if any(note.is_ln for note in row_notes):
                local_previous = current_lanes
                continue
            if _jack_time_has_active_ln(time_ms, ln_intervals):
                local_previous = current_lanes
                continue
            if _deterministic_index(100, row_pos, section_index, int(target * 100)) >= _jack_anchor_light_apply_rate(target, temperature):
                local_previous = current_lanes
                continue

            prev_lanes = local_previous or (
                sorted({note.lane for note in rows[row_times[row_pos - 1]] if 0 <= note.lane <= 3})
                if row_pos > 0
                else []
            )
            next_lanes = (
                sorted({note.lane for note in rows[row_times[row_pos + 1]] if 0 <= note.lane <= 3})
                if row_pos + 1 < len(row_times)
                else []
            )
            replacement = _choose_jack_anchor_light_pattern(
                current_lanes,
                prev_lanes,
                next_lanes,
                lane_counts,
                row_pos,
                section_index,
                target,
            )
            if replacement == current_lanes:
                local_previous = current_lanes
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
            local_previous = replacement
            changed += 1

    if changed <= 0:
        return notes
    return _flatten_note_rows(rows)


def _jack_anchor_contrast_window_size(target: float) -> int:
    if target < 4.75:
        return 40
    if target < 6.0:
        return 32
    return 28


def _jack_anchor_light_sections(
    row_count: int,
    window_size: int,
    target: float,
    temperature: float,
) -> Set[int]:
    section_count = max(1, math.ceil(row_count / max(1, window_size)))
    light_sections: Set[int] = set()
    cycle = 5 if target < 5.75 else 4
    phase = _deterministic_index(cycle, int(target * 100), int(temperature * 100))
    secondary_phase = (phase + 3) % cycle
    roll_threshold = int(14 + max(0.0, min(1.0, temperature)) * 16)

    for section_index in range(section_count):
        scheduled = section_index % cycle in {phase, secondary_phase}
        rolled = _deterministic_index(100, section_index, int(target * 100), int(temperature * 100), 0xA17) < roll_threshold
        if scheduled or rolled:
            light_sections.add(section_index)

    return light_sections


def _jack_anchor_light_apply_rate(target: float, temperature: float) -> int:
    if target < 4.75:
        base = 56
    elif target < 5.35:
        base = 60
    elif target < 6.0:
        base = 66
    else:
        base = 62
    return max(35, min(88, int(round(base + temperature * 8))))


def _jack_time_has_active_ln(time_ms: int, ln_intervals: List[Tuple[int, int, int]]) -> bool:
    return any(start < time_ms < end for _, start, end in ln_intervals)


def _choose_jack_anchor_light_pattern(
    current_lanes: List[int],
    prev_lanes: List[int],
    next_lanes: List[int],
    lane_counts: Counter,
    row_pos: int,
    section_index: int,
    target: float,
) -> List[int]:
    size = len(current_lanes)
    patterns = _jack_stack_patterns(size)
    if not patterns:
        return current_lanes

    current_set = set(current_lanes)
    prev_set = set(prev_lanes)
    next_set = set(next_lanes)
    max_prev_overlap = _jack_anchor_light_overlap_limit(size, row_pos, section_index)
    candidates = [
        lanes
        for lanes in patterns
        if set(lanes) != current_set
        and len(set(lanes) & prev_set) <= max_prev_overlap
    ]
    if not candidates:
        candidates = [lanes for lanes in patterns if set(lanes) != current_set]
    if not candidates:
        return current_lanes

    ranked = sorted(
        candidates,
        key=lambda lanes: (
            len(set(lanes) & prev_set),
            -len(set(lanes) & next_set) * 0.35,
            sum(lane_counts[lane] for lane in lanes),
            _deterministic_index(1000, row_pos, section_index, sum(lanes), int(target * 100)),
        ),
    )
    return list(ranked[0])


def _jack_anchor_light_overlap_limit(size: int, row_pos: int, section_index: int) -> int:
    if size <= 1:
        return 0
    if size == 2:
        return 0 if (row_pos + section_index) % 4 == 0 else 1
    return 1 if (row_pos + section_index) % 4 == 0 else 2


def _jack_anchor_run_soft_limit(target: float, temperature: float) -> int:
    if target < 4.75:
        base = 11
    elif target < 5.75:
        base = 10
    else:
        base = 9
    if temperature < 0.35:
        base += 2
    elif temperature >= 0.75:
        base -= 1
    return max(8, base)


def _shorten_jack_anchor_runs(
    notes: List[NoteObject],
    max_anchor_length: int,
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

    comfort_length = _jack_anchor_comfort_length(max_anchor_length, temperature)
    lane_runs = [0, 0, 0, 0]
    lane_counts = Counter(note.lane for note in notes if 0 <= note.lane <= 3)
    changed = 0

    for row_pos, time_ms in enumerate(row_times):
        row_notes = sorted(rows.get(time_ms, []), key=lambda item: (item.lane, item.end_time_ms or -1))
        current_lanes = sorted({note.lane for note in row_notes if 0 <= note.lane <= 3})
        if not current_lanes:
            lane_runs = [0, 0, 0, 0]
            continue

        fatigued_lanes = [
            lane
            for lane in current_lanes
            if _jack_anchor_should_rotate(
                lane=lane,
                run_length=lane_runs[lane] + 1,
                comfort_length=comfort_length,
                row_size=len(current_lanes),
                row_pos=row_pos,
                temperature=temperature,
            )
        ]

        if len(current_lanes) >= 4 and fatigued_lanes:
            replacement = _downgrade_jack_overrun_chord(current_lanes, fatigued_lanes, lane_runs, lane_counts, row_pos, max_anchor_length, comfort_length)
            if replacement != current_lanes:
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
                current_lanes = replacement
                changed += 1
        elif len(current_lanes) >= 4:
            current_set = set(current_lanes)
            for lane in range(4):
                lane_runs[lane] = lane_runs[lane] + 1 if lane in current_set else 0
            continue

        if len(current_lanes) >= 3 and not fatigued_lanes:
            current_set = set(current_lanes)
            for lane in range(4):
                lane_runs[lane] = lane_runs[lane] + 1 if lane in current_set else 0
            continue

        fatigued_lanes = [
            lane
            for lane in current_lanes
            if _jack_anchor_should_rotate(
                lane=lane,
                run_length=lane_runs[lane] + 1,
                comfort_length=comfort_length,
                row_size=len(current_lanes),
                row_pos=row_pos,
                temperature=temperature,
            )
        ]
        if fatigued_lanes:
            replacement = _choose_short_jack_anchor_replacement(
                current_lanes,
                fatigued_lanes,
                lane_runs,
                lane_counts,
                row_pos,
                max_anchor_length,
                comfort_length,
            )
            if replacement != current_lanes:
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
                current_lanes = replacement
                changed += 1

        current_set = set(current_lanes)
        for lane in range(4):
            lane_runs[lane] = lane_runs[lane] + 1 if lane in current_set else 0

    if changed <= 0:
        return notes
    return _flatten_note_rows(rows)


def _jack_anchor_comfort_length(max_anchor_length: int, temperature: float) -> int:
    # Kept for config compatibility: this is a comfort point, not a hard cap.
    # Long anchors can still appear, but fatigue pressure makes them rotate naturally.
    base = max(2, min(5, int(max_anchor_length or 3)))
    if temperature >= 0.75:
        base = max(2, base - 1)
    return base


def _jack_anchor_should_rotate(
    lane: int,
    run_length: int,
    comfort_length: int,
    row_size: int,
    row_pos: int,
    temperature: float,
) -> bool:
    if run_length <= comfort_length:
        return False

    excess = run_length - comfort_length
    chance = 58 + excess * 20 + int(max(0.0, min(1.0, temperature)) * 8)
    if row_size >= 3:
        # Big accents may intentionally keep a lane for one more hit.
        chance -= 8
    chance = max(18, min(98, chance))
    roll = _deterministic_index(
        100,
        row_pos,
        lane,
        run_length,
        row_size,
        int(temperature * 1000),
    )
    return roll < chance


def _jack_anchor_fatigue_pressure(run_length: int, comfort_length: int) -> int:
    if run_length <= comfort_length:
        return 0
    excess = run_length - comfort_length
    return excess * excess * 8


def _downgrade_jack_overrun_chord(
    current_lanes: List[int],
    overrun_lanes: List[int],
    lane_runs: List[int],
    lane_counts: Counter,
    row_pos: int,
    max_anchor_length: int,
    comfort_length: int,
) -> List[int]:
    if len(current_lanes) <= 1:
        return current_lanes
    target_size = len(current_lanes) - 1
    patterns = _jack_stack_patterns(target_size)
    overrun_set = set(overrun_lanes)
    candidates = [
        lanes
        for lanes in patterns
        if set(lanes).issubset(set(current_lanes))
    ] or patterns
    no_overrun = [lanes for lanes in candidates if not (set(lanes) & overrun_set)]
    if no_overrun:
        candidates = no_overrun
    ranked = sorted(
        candidates,
        key=lambda lanes: (
            sum(_jack_anchor_fatigue_pressure(lane_runs[lane] + 1, comfort_length) for lane in lanes),
            len(set(lanes) & overrun_set),
            sum(lane_counts[lane] for lane in lanes),
            _deterministic_index(1000, row_pos, sum(lanes), int(max_anchor_length * 100)),
        ),
    )
    return list(ranked[0]) if ranked else current_lanes


def _choose_short_jack_anchor_replacement(
    current_lanes: List[int],
    overrun_lanes: List[int],
    lane_runs: List[int],
    lane_counts: Counter,
    row_pos: int,
    max_anchor_length: int,
    comfort_length: int,
) -> List[int]:
    size = len(current_lanes)
    patterns = _jack_stack_patterns(size)
    current_set = set(current_lanes)
    overrun_set = set(overrun_lanes)
    candidates = [lanes for lanes in patterns if set(lanes) != current_set]
    if size >= 2:
        candidates = [
            lanes
            for lanes in candidates
            if len(set(lanes) & current_set) >= max(1, size - 1)
        ] or candidates
    if not candidates:
        return current_lanes
    no_overrun_candidates = [
        lanes
        for lanes in candidates
        if not (set(lanes) & overrun_set)
    ]
    if no_overrun_candidates:
        candidates = no_overrun_candidates

    ranked = sorted(
        candidates,
        key=lambda lanes: (
            sum(_jack_anchor_fatigue_pressure(lane_runs[lane] + 1, comfort_length) for lane in lanes),
            len(set(lanes) & overrun_set),
            sum(lane_counts[lane] for lane in lanes),
            -len(set(lanes) & current_set) * 0.25,
            _deterministic_index(1000, row_pos, sum(lanes), int(max_anchor_length * 100)),
        ),
    )
    return list(ranked[0])


def _choose_jack_anchor_rotation(
    current_lanes: List[int],
    prev_lanes: List[int],
    next_lanes: List[int],
    running: List[int],
    lane_counts: Counter,
    anchor_limit: int,
    row_pos: int,
    target: float,
    repeated_pattern: bool,
) -> List[int]:
    size = len(current_lanes)
    patterns = _jack_stack_patterns(size)
    current_set = set(current_lanes)
    candidates = [lanes for lanes in patterns if set(lanes) != current_set]
    if size >= 2:
        candidates = [
            lanes
            for lanes in candidates
            if len(set(lanes) & current_set) >= size - 1
        ] or candidates
    if not candidates:
        return current_lanes

    ranked = sorted(
        candidates,
        key=lambda lanes: _jack_anchor_rotation_score(
            lanes,
            current_lanes,
            prev_lanes,
            next_lanes,
            running,
            lane_counts,
            anchor_limit,
            row_pos,
            target,
            repeated_pattern,
        ),
    )
    best = ranked[0]
    current_score = _jack_anchor_rotation_score(
        current_lanes,
        current_lanes,
        prev_lanes,
        next_lanes,
        running,
        lane_counts,
        anchor_limit,
        row_pos,
        target,
        repeated_pattern,
    )
    best_score = _jack_anchor_rotation_score(
        best,
        current_lanes,
        prev_lanes,
        next_lanes,
        running,
        lane_counts,
        anchor_limit,
        row_pos,
        target,
        repeated_pattern,
    )
    return list(best) if best_score + 0.05 < current_score else current_lanes


def _jack_anchor_rotation_score(
    lanes: List[int],
    current_lanes: List[int],
    prev_lanes: List[int],
    next_lanes: List[int],
    running: List[int],
    lane_counts: Counter,
    anchor_limit: int,
    row_pos: int,
    target: float,
    repeated_pattern: bool,
) -> float:
    lane_set = set(lanes)
    current_set = set(current_lanes)
    prev_set = set(prev_lanes)
    next_set = set(next_lanes)
    projected_anchor = sum(max(0, running[lane] + 1 - anchor_limit) for lane in lanes)
    pressure = sum(lane_counts[lane] for lane in lanes) / max(1, len(lanes))
    overlap_current = len(lane_set & current_set)
    overlap_neighbors = len(lane_set & prev_set) + len(lane_set & next_set)
    all_new_penalty = 3.5 if overlap_current == 0 and len(lanes) > 1 else 0.0
    repeated_penalty = 2.4 if repeated_pattern and lane_set == current_set else 0.0
    center_pair_penalty = 0.8 if lane_set == {1, 2} and target < 5.75 else 0.0
    deterministic_tie = _deterministic_index(1000, row_pos, sum(lanes), int(target * 100)) / 1000.0

    return (
        projected_anchor * 7.0
        + pressure * 0.035
        + all_new_penalty
        + repeated_penalty
        + center_pair_penalty
        - overlap_current * 1.2
        - overlap_neighbors * 0.55
        + deterministic_tie * 0.02
    )


def _choose_jack_pressure_replacement(
    current_lanes: List[int],
    prev_lanes: List[int],
    next_lanes: List[int],
    lane_counts: Counter,
    total_notes: int,
    center_cap: float,
    outer_floor: float,
    row_pos: int,
    pass_index: int,
    target: float,
) -> List[int]:
    size = len(current_lanes)
    current_set = set(current_lanes)
    patterns = _jack_stack_patterns(size)
    candidates = [lanes for lanes in patterns if set(lanes) != current_set]
    if size >= 2:
        candidates = [
            lanes
            for lanes in candidates
            if len(set(lanes) & current_set) >= max(1, size - 1)
        ] or candidates
    if not candidates:
        return current_lanes

    current_score = _jack_pressure_replacement_score(
        current_lanes,
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
    ranked = sorted(
        candidates,
        key=lambda lanes: _jack_pressure_replacement_score(
            lanes,
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
        ),
    )
    best = ranked[0]
    best_score = _jack_pressure_replacement_score(
        best,
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
    return list(best) if best_score + 0.01 < current_score else current_lanes


def _jack_pressure_replacement_score(
    lanes: List[int],
    current_lanes: List[int],
    prev_lanes: List[int],
    next_lanes: List[int],
    lane_counts: Counter,
    total_notes: int,
    center_cap: float,
    outer_floor: float,
    row_pos: int,
    pass_index: int,
    target: float,
) -> float:
    adjusted = [lane_counts[lane] for lane in range(4)]
    for lane in current_lanes:
        adjusted[lane] -= 1
    for lane in lanes:
        adjusted[lane] += 1

    center_excess = sum(max(0.0, adjusted[lane] - total_notes * center_cap) for lane in [1, 2])
    outer_shortage = sum(max(0.0, total_notes * outer_floor - adjusted[lane]) for lane in [0, 3])
    lane_spread = max(adjusted) - min(adjusted)
    current_set = set(current_lanes)
    lane_set = set(lanes)
    neighbor_overlap = len(lane_set & set(prev_lanes)) + len(lane_set & set(next_lanes))
    self_overlap = len(lane_set & current_set)
    middle_pair_penalty = 4.0 if lane_set == {1, 2} else 0.0
    center_count = sum(1 for lane in lanes if lane in [1, 2])
    outer_count = len(lanes) - center_count
    deterministic_tie = _deterministic_index(1000, row_pos, pass_index, sum(lanes), int(target * 100)) / 1000.0

    return (
        center_excess * 3.2
        + outer_shortage * 1.55
        + lane_spread * 0.08
        + middle_pair_penalty
        + max(0, center_count - outer_count) * 0.35
        - neighbor_overlap * 1.65
        - self_overlap * 1.15
        + deterministic_tie * 0.02
    )


def _jack_stack_patterns(size: int) -> List[List[int]]:
    if size <= 1:
        return [[0], [1], [2], [3]]
    if size == 2:
        return [[0, 1], [1, 2], [2, 3], [0, 3], [0, 2], [1, 3]]
    if size == 3:
        return [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
    return [[0, 1, 2, 3]]

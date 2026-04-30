import bisect
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..core.difficulty_estimator import DifficultyEstimator
from ..core.models import DifficultyConfig, NoteObject
from ..core.style_rules import clamp_max_chord_size
from ..core.validator import Validator

LANES = (0, 1, 2, 3)


@dataclass(frozen=True)
class TechPoint:
    time_ms: int
    attack: float
    kick: float
    texture: float
    sustain: float
    release: float
    beat: float
    accent: float
    support: float
    dump_risk: float
    must_hit: bool
    can_hit: bool
    silent: bool

    @property
    def score(self) -> float:
        return (
            self.accent * 1.10
            + self.attack * 0.80
            + self.kick * 0.90
            + self.texture * 0.48
            + self.release * 0.22
            - self.dump_risk * 0.50
        )


def generate_tech_to_target_sr(
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    tolerance: float = 0.15,
    max_attempts: int = 24,
) -> Tuple[List[NoteObject], float, bool, int]:
    target = float(config.target_star or 5.0)
    snap_points = sorted({int(t) for t in snap_points})
    if not snap_points:
        return [], 0.0, False, 0

    context = _build_context(analysis, snap_points, accent_snap_points)
    beat = 60000.0 / max(1.0, float(analysis.get("bpm", 120.0)))
    best_notes: List[NoteObject] = []
    best_sr = 0.0
    best_quality = float("-inf")
    best_diff = float("inf")
    best_attempt = 0

    for attempt, variant in enumerate(_variants(config, target, max_attempts), start=1):
        rows = _select_rows(analysis, snap_points, context, beat, target, variant)
        if not rows:
            continue
        pattern_target = _ln_base_target(target, config) if config.chart_type == "ln" else target
        lanes_by_row = _assign_lanes(rows, context, beat, pattern_target, config, variant)
        lanes_by_row = _calibrate_lanes(lanes_by_row, context, pattern_target, tolerance, config, analysis)
        lanes_by_row = _repair_active_holes(lanes_by_row, snap_points, context, beat, target)
        notes = _notes_from_rows(lanes_by_row)
        if config.chart_type == "ln":
            notes = _apply_lns(notes, context, snap_points, config, analysis, beat)
        notes = Validator.validate_and_fix(notes, config, analysis.get("silent_regions", []), snap_points)
        sr = DifficultyEstimator.estimate_sr(notes, int(analysis.get("duration_ms", 0)))
        notes, sr = _trim_over_target(notes, context, target, tolerance, config, analysis, snap_points)

        diff = abs(sr - target)
        quality = _quality(notes, context, beat)
        if diff < best_diff or (diff <= best_diff + 0.04 and quality > best_quality):
            best_notes = notes
            best_sr = sr
            best_quality = quality
            best_diff = diff
            best_attempt = attempt
        if diff <= tolerance:
            return notes, sr, True, attempt

    return best_notes, best_sr, best_diff <= tolerance, best_attempt


def _ln_base_target(target: float, config: DifficultyConfig) -> float:
    ratio = max(0.0, min(1.0, config.ln_ratio))
    margin = 0.26 + _target_norm(target) * 0.34
    return max(1.0, target - margin * min(1.0, ratio / 0.45))


def _variants(config: DifficultyConfig, target: float, max_attempts: int) -> List[Dict[str, float]]:
    allowed = _allowed_divisors(config)
    preferred = 4 if target < 4.2 else 6 if target < 5.4 else 8
    ordered = sorted(allowed, key=lambda divisor: (abs(divisor - preferred), divisor))
    variants: List[Dict[str, float]] = []
    for divisor in ordered:
        for support_shift, chord_shift, continuity_shift in [
            (0.00, 0.00, 0.00),
            (-0.05, 0.03, -0.05),
            (0.04, -0.03, 0.05),
            (-0.09, 0.05, -0.08),
        ]:
            variants.append({
                "divisor": float(divisor),
                "support_shift": support_shift,
                "chord_shift": chord_shift,
                "continuity_shift": continuity_shift,
            })
    return variants[:max(1, max_attempts)]


def _allowed_divisors(config: DifficultyConfig) -> List[int]:
    divisors: List[int] = []
    for value in config.allowed_subdivisions or []:
        if isinstance(value, str) and "/" in value:
            try:
                denominator = int(value.split("/", 1)[1])
            except ValueError:
                continue
            if denominator in {4, 6, 8}:
                divisors.append(denominator)
    return sorted(set(divisors)) or [4, 6, 8]


def _select_rows(
    analysis: Dict[str, Any],
    snap_points: List[int],
    context: Dict[int, TechPoint],
    beat: float,
    target: float,
    variant: Dict[str, float],
) -> List[int]:
    divisor = int(variant["divisor"])
    support_gate = max(0.12, 0.30 - _target_norm(target) * 0.10 + float(variant["support_shift"]))
    min_gap = _min_row_gap(beat, target, divisor)
    desired_gap = _desired_gap(beat, target, float(variant["continuity_shift"]))
    rows: List[int] = []

    for time_ms in _variable_tech_grid(analysis, snap_points, beat, target, divisor):
        point = context.get(time_ms)
        if point is None or point.silent:
            continue
        keep = False
        if point.must_hit:
            keep = True
        elif point.can_hit and point.support >= support_gate:
            keep = True
        elif point.can_hit and point.texture >= 0.36 and point.dump_risk <= 0.64 and point.beat >= 0.20:
            keep = True
        if keep:
            rows = _insert_row(rows, time_ms, context, min_gap)

    must_rows = [time_ms for time_ms, point in context.items() if point.must_hit]
    for time_ms in sorted(must_rows, key=lambda t: (-context[t].score, t)):
        rows = _insert_row(rows, time_ms, context, min_gap)

    rows = _fill_supported_gaps(rows, snap_points, context, min_gap, desired_gap, support_gate)
    return rows


def _variable_tech_grid(
    analysis: Dict[str, Any],
    snap_points: List[int],
    beat: float,
    target: float,
    base_divisor: int,
) -> List[int]:
    duration = int(analysis.get("duration_ms", snap_points[-1] if snap_points else 0))
    offset = float(analysis.get("offset_ms", snap_points[0] if snap_points else 0))
    if target < 4.2:
        pool = [4, 4, 6]
        section_beats = 4
    elif target < 5.6:
        pool = [4, 6, 6, 8]
        section_beats = 4
    elif target < 6.6:
        pool = [6, 8, 6, 8, 4]
        section_beats = 3
    else:
        pool = [8, 6, 8, 8, 6, 4]
        section_beats = 2
    if base_divisor in {4, 6, 8}:
        pool.insert(0, base_divisor)

    rows: List[int] = []
    section = max(beat, beat * section_beats)
    section_start = offset
    section_index = 0
    while section_start <= duration:
        divisor = pool[_deterministic_index(len(pool), int(section_start), section_index, int(target * 100))]
        step = beat / divisor
        phase_roll = _deterministic_index(6, section_index, int(target * 100), divisor)
        phase = 0.0
        if phase_roll == 1:
            phase = step * 0.5
        elif phase_roll == 2 and target >= 5.0:
            phase = step * 0.25
        t = section_start + phase
        section_end = min(duration + beat, section_start + section)
        while t < section_end:
            snap = _nearest_snap(snap_points, int(round(t)))
            if snap is not None:
                rows.append(snap)
            t += step
        section_start += section
        section_index += 1
    return sorted(dict.fromkeys(rows))


def _fill_supported_gaps(
    rows: List[int],
    snap_points: List[int],
    context: Dict[int, TechPoint],
    min_gap: int,
    desired_gap: int,
    support_gate: float,
) -> List[int]:
    rows = sorted(dict.fromkeys(rows))
    if len(rows) < 2:
        return rows
    existing = set(rows)
    additions: List[int] = []
    for left, right in zip(rows, rows[1:]):
        current = left
        guard = 0
        while right - current > desired_gap and guard < 16:
            guard += 1
            start = current + min_gap
            end = min(right - min_gap, current + desired_gap + max(12, min_gap // 2))
            if end < start:
                break
            lo = bisect.bisect_left(snap_points, start)
            hi = bisect.bisect_right(snap_points, end)
            candidates = [
                time_ms for time_ms in snap_points[lo:hi]
                if time_ms not in existing
                and time_ms not in additions
                and time_ms in context
                and _can_fill_single(context[time_ms], support_gate)
            ]
            if not candidates:
                fallback_hi = bisect.bisect_right(snap_points, right - min_gap)
                candidates = [
                    time_ms for time_ms in snap_points[lo:fallback_hi]
                    if time_ms not in existing
                    and time_ms not in additions
                    and time_ms in context
                    and _can_fill_single(context[time_ms], support_gate)
                ]
                if not candidates:
                    break
            ideal = current + desired_gap
            chosen = max(
                candidates,
                key=lambda t: (context[t].support, context[t].texture, context[t].score, -abs(t - ideal)),
            )
            additions.append(chosen)
            current = chosen
    if additions:
        rows.extend(additions)
    return sorted(dict.fromkeys(rows))


def _can_fill_single(point: TechPoint, support_gate: float) -> bool:
    if point.silent or not point.can_hit:
        return False
    if point.dump_risk > 0.84:
        return False
    return (
        point.must_hit
        or point.support >= max(0.10, support_gate - 0.14)
        or (point.texture >= 0.24 and point.dump_risk <= 0.74)
    )


def _assign_lanes(
    rows: List[int],
    context: Dict[int, TechPoint],
    beat: float,
    target: float,
    config: DifficultyConfig,
    variant: Dict[str, float],
) -> Dict[int, Set[int]]:
    max_chord = min(4, clamp_max_chord_size(config, "tech"))
    chord_rows = _pick_chord_rows(rows, context, target, max_chord, float(variant["chord_shift"]))
    modules = _modules_for_rows(rows, target)
    pressure = {lane: 0 for lane in LANES}
    last_lane_time = {lane: -999999 for lane in LANES}
    last_chord: Set[int] = set()
    lanes_by_row: Dict[int, Set[int]] = {}

    for index, time_ms in enumerate(rows):
        module = modules.get(time_ms, "bullets")
        if time_ms in chord_rows:
            lanes = _chord_lanes(time_ms, index, context[time_ms], target, max_chord, last_chord, pressure, last_lane_time)
            last_chord = set(lanes)
        else:
            lane = _single_lane(time_ms, index, module, beat, target, pressure, last_lane_time, last_chord)
            lanes = {lane}
        lanes_by_row[time_ms] = lanes
        for lane in lanes:
            pressure[lane] += 1
            last_lane_time[lane] = time_ms
    return lanes_by_row


def _modules_for_rows(rows: List[int], target: float) -> Dict[int, str]:
    if not rows:
        return {}
    result: Dict[int, str] = {}
    index = 0
    module_tables = {
        "low": [("bullets", 40), ("cut", 28), ("awkward", 22), ("stacklet", 10)],
        "mid": [("bullets", 27), ("cut", 30), ("awkward", 24), ("stacklet", 19)],
        "high": [("bullets", 20), ("cut", 32), ("awkward", 25), ("stacklet", 23)],
    }
    table = module_tables["low" if target < 4.5 else "mid" if target < 6.2 else "high"]
    while index < len(rows):
        length = 8 + _deterministic_index(13, rows[index], index, int(target * 100))
        roll = _deterministic_index(100, rows[index], length, int(target * 100))
        acc = 0
        module = table[-1][0]
        for name, weight in table:
            acc += weight
            if roll < acc:
                module = name
                break
        for time_ms in rows[index:index + length]:
            result[time_ms] = module
        index += length
    return result


def _pick_chord_rows(
    rows: List[int],
    context: Dict[int, TechPoint],
    target: float,
    max_chord: int,
    chord_shift: float,
) -> Set[int]:
    if max_chord <= 1 or not rows:
        return set()
    norm = _target_norm(target)
    ratio = max(0.08, min(0.48, 0.14 + norm * 0.22 + chord_shift))
    desired = int(round(len(rows) * ratio))
    max_run = _max_chord_run(target)
    candidates = [
        time_ms for time_ms in rows
        if context[time_ms].can_hit
        and context[time_ms].dump_risk <= 0.70
        and (
            context[time_ms].must_hit
            or context[time_ms].accent >= 0.28
            or context[time_ms].texture >= 0.44
            or context[time_ms].attack >= 0.22
        )
    ]
    ranked = sorted(
        candidates,
        key=lambda t: (context[t].accent, context[t].kick, context[t].attack, context[t].texture, context[t].score),
        reverse=True,
    )
    index_by_time = {time_ms: index for index, time_ms in enumerate(rows)}
    chosen: Set[int] = set()
    for time_ms in ranked:
        if len(chosen) >= desired:
            break
        idx = index_by_time[time_ms]
        if _would_exceed_chord_run(idx, chosen, index_by_time, max_run):
            continue
        chosen.add(time_ms)
    return chosen


def _would_exceed_chord_run(idx: int, chosen: Set[int], index_by_time: Dict[int, int], max_run: int) -> bool:
    chosen_indices = {index_by_time[t] for t in chosen}
    run = 1
    pos = idx - 1
    while pos in chosen_indices:
        run += 1
        pos -= 1
    pos = idx + 1
    while pos in chosen_indices:
        run += 1
        pos += 1
    return run > max_run


def _single_lane(
    time_ms: int,
    index: int,
    module: str,
    beat: float,
    target: float,
    pressure: Dict[int, int],
    last_lane_time: Dict[int, int],
    last_chord: Set[int],
) -> int:
    motifs = {
        "bullets": [0, 2, 1, 3, 2, 0, 3, 1],
        "cut": [0, 3, 1, 2, 3, 0, 2, 1],
        "awkward": [1, 0, 2, 3, 1, 3, 0, 2],
        "stacklet": [0, 3, 1, 2, 0, 2, 3, 1],
    }
    motif = motifs.get(module, motifs["bullets"])
    preferred = motif[index % len(motif)]
    hard_gap = _same_lane_gap(beat, target)

    def lane_cost(lane: int) -> Tuple[float, int, int]:
        gap = time_ms - last_lane_time[lane]
        bad_same = 1000 if gap < hard_gap else 0
        module_bias = 0.0
        if module == "stacklet" and last_chord:
            module_bias = -0.25 if lane not in last_chord else 0.22
        elif module == "cut":
            module_bias = -0.12 if lane in {0, 3} else 0.0
        elif module == "awkward":
            module_bias = -0.10 if lane in {1, 2} else 0.04
        random_bias = _deterministic_index(100, time_ms, lane, index) / 1000.0
        return (
            bad_same + (0 if lane == preferred else 1) + module_bias + random_bias,
            pressure[lane],
            lane,
        )

    return min(LANES, key=lane_cost)


def _chord_lanes(
    time_ms: int,
    index: int,
    point: TechPoint,
    target: float,
    max_chord: int,
    last_chord: Set[int],
    pressure: Dict[int, int],
    last_lane_time: Dict[int, int],
) -> Set[int]:
    norm = _target_norm(target)
    allow_triple = (
        max_chord >= 3
        and target >= 5.7
        and (point.accent >= 0.55 or point.kick >= 0.40 or point.attack >= 0.50 or point.texture >= 0.74)
        and _deterministic_index(100, time_ms, index, int(target * 100)) < int(4 + norm * 16)
    )
    allow_quad = (
        max_chord >= 4
        and target >= 6.8
        and point.accent >= 0.82
        and point.dump_risk <= 0.28
        and _deterministic_index(100, time_ms, index) < 2
    )
    pools = [
        [{0, 1}, {2, 3}, {0, 3}, {1, 2}, {0, 2}, {1, 3}],
        [{0, 1, 3}, {0, 2, 3}, {0, 1, 2}, {1, 2, 3}],
        [{0, 1, 2, 3}],
    ]
    pool = pools[2] if allow_quad else pools[1] if allow_triple else pools[0]
    hard_gap = 0 if target >= 6.5 else 24
    ranked = sorted(
        pool,
        key=lambda lanes: (
            sum(1 for lane in lanes if time_ms - last_lane_time[lane] < hard_gap),
            lanes == last_chord,
            sum(pressure[lane] for lane in lanes),
            _deterministic_index(1000, time_ms, index, sum(lanes)),
        ),
    )
    lanes = set(ranked[0])
    while len(lanes) > max_chord:
        lanes.remove(max(lanes, key=lambda lane: pressure[lane]))
    return lanes


def _calibrate_lanes(
    rows: Dict[int, Set[int]],
    context: Dict[int, TechPoint],
    target: float,
    tolerance: float,
    config: DifficultyConfig,
    analysis: Dict[str, Any],
) -> Dict[int, Set[int]]:
    duration = int(analysis.get("duration_ms", 0))
    max_chord = clamp_max_chord_size(config, "tech")
    max_run = _max_chord_run(target)
    rows = {time_ms: set(lanes) for time_ms, lanes in rows.items()}

    sr = DifficultyEstimator.estimate_sr(_notes_from_rows(rows), duration)
    if sr < target - tolerance:
        rows, sr = _raise_sr(rows, context, target, tolerance, duration, max_chord, max_run)
    if sr > target + tolerance:
        rows, sr = _lower_sr(rows, context, target, tolerance, duration)
    return rows


def _repair_active_holes(
    rows: Dict[int, Set[int]],
    snap_points: List[int],
    context: Dict[int, TechPoint],
    beat: float,
    target: float,
) -> Dict[int, Set[int]]:
    if len(rows) < 2:
        return rows
    rows = {time_ms: set(lanes) for time_ms, lanes in rows.items()}
    pressure = {lane: 0 for lane in LANES}
    last_lane_time = {lane: -999999 for lane in LANES}
    for time_ms in sorted(rows):
        for lane in rows[time_ms]:
            pressure[lane] += 1
            last_lane_time[lane] = time_ms

    min_gap = 53
    times = sorted(rows)
    additions: Dict[int, Set[int]] = {}
    for left, right in zip(times, times[1:]):
        if right - left <= 260:
            continue
        lo = bisect.bisect_left(snap_points, left + min_gap)
        hi = bisect.bisect_right(snap_points, right - min_gap)
        candidates = [
            time_ms for time_ms in snap_points[lo:hi]
            if time_ms not in rows
            and time_ms in context
            and context[time_ms].can_hit
            and context[time_ms].support >= 0.18
        ]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda t: (context[t].support, context[t].score, -abs(t - (left + right) / 2.0)))
        lane = _single_lane(chosen, len(rows) + len(additions), "bullets", beat, target, pressure, last_lane_time, set())
        additions[chosen] = {lane}
        pressure[lane] += 1
        last_lane_time[lane] = chosen
    rows.update(additions)
    return rows


def _raise_sr(
    rows: Dict[int, Set[int]],
    context: Dict[int, TechPoint],
    target: float,
    tolerance: float,
    duration: int,
    max_chord: int,
    max_run: int,
) -> Tuple[Dict[int, Set[int]], float]:
    sr = DifficultyEstimator.estimate_sr(_notes_from_rows(rows), duration)
    if max_chord <= 1:
        return rows, sr
    pressure = {lane: 0 for lane in LANES}
    for lanes in rows.values():
        for lane in lanes:
            pressure[lane] += 1

    times = sorted(rows)
    index_by_time = {time_ms: index for index, time_ms in enumerate(times)}
    target_chord_ratio = min(0.58, 0.23 + _target_norm(target) * 0.27)
    max_chord_rows = int(round(len(times) * target_chord_ratio))
    candidate_order = sorted(
        times,
        key=lambda t: (context[t].score, context[t].accent, context[t].texture, context[t].attack),
        reverse=True,
    )

    for pass_index in range(3):
        for time_ms in candidate_order:
            if sr >= target - tolerance:
                return rows, sr
            if sum(1 for lanes in rows.values() if len(lanes) > 1) >= max_chord_rows and len(rows[time_ms]) == 1:
                continue
            if len(rows[time_ms]) >= max_chord:
                continue
            point = context[time_ms]
            if not point.can_hit or point.dump_risk > 0.76:
                continue
            if pass_index == 0 and point.accent < 0.34 and point.attack < 0.24 and point.kick < 0.16:
                continue
            if pass_index == 1 and point.texture < 0.42 and point.support < 0.28:
                continue
            if _would_exceed_chord_run(index_by_time[time_ms], {t for t, lanes in rows.items() if len(lanes) > 1 and t != time_ms}, index_by_time, max_run):
                continue
            if len(rows[time_ms]) >= 2 and target < 5.7:
                continue
            if len(rows[time_ms]) >= 3 and target < 6.8:
                continue
            lane = _extra_lane(rows[time_ms], pressure)
            if lane is None:
                continue
            rows[time_ms].add(lane)
            pressure[lane] += 1
            if _deterministic_index(8, time_ms, pass_index) == 0:
                sr = DifficultyEstimator.estimate_sr(_notes_from_rows(rows), duration)
        sr = DifficultyEstimator.estimate_sr(_notes_from_rows(rows), duration)
    return rows, sr


def _lower_sr(
    rows: Dict[int, Set[int]],
    context: Dict[int, TechPoint],
    target: float,
    tolerance: float,
    duration: int,
) -> Tuple[Dict[int, Set[int]], float]:
    sr = DifficultyEstimator.estimate_sr(_notes_from_rows(rows), duration)
    if sr <= target + tolerance:
        return rows, sr
    chord_times = sorted(
        [time_ms for time_ms, lanes in rows.items() if len(lanes) > 1],
        key=lambda t: (context[t].accent, context[t].attack, context[t].texture),
    )
    for time_ms in chord_times:
        if sr <= target + tolerance:
            break
        rows[time_ms] = _best_single(rows[time_ms], time_ms)
        sr = DifficultyEstimator.estimate_sr(_notes_from_rows(rows), duration)
    if sr > target + tolerance:
        removable = sorted(
            [time_ms for time_ms, lanes in rows.items() if len(lanes) == 1 and not context[time_ms].must_hit],
            key=lambda t: (context[t].score, context[t].support),
        )
        max_gap = 230 if target < 4.0 else 250 if target < 5.5 else 210
        supported_times = sorted(
            time_ms for time_ms, point in context.items()
            if point.can_hit and point.support >= 0.18
        )
        for time_ms in removable:
            if sr <= target + tolerance:
                break
            times = sorted(rows)
            idx = bisect.bisect_left(times, time_ms)
            if idx <= 0 or idx >= len(times) - 1:
                continue
            left = times[idx - 1]
            right = times[idx + 1]
            if right - left > max_gap:
                continue
            probe_index = bisect.bisect_right(supported_times, left)
            creates_active_hole = False
            while probe_index < len(supported_times) and supported_times[probe_index] < right:
                probe = supported_times[probe_index]
                if probe != time_ms and probe - left >= 53 and right - probe >= 53:
                    creates_active_hole = True
                    break
                probe_index += 1
            if creates_active_hole:
                continue
            candidate = dict(rows)
            candidate.pop(time_ms, None)
            rows = candidate
            sr = DifficultyEstimator.estimate_sr(_notes_from_rows(rows), duration)
    return rows, sr


def _trim_over_target(
    notes: List[NoteObject],
    context: Dict[int, TechPoint],
    target: float,
    tolerance: float,
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
) -> Tuple[List[NoteObject], float]:
    duration = int(analysis.get("duration_ms", 0))
    sr = DifficultyEstimator.estimate_sr(notes, duration)
    if sr <= target + tolerance:
        return notes, sr
    if any(note.is_ln for note in notes):
        working = [NoteObject(note.time_ms, note.lane, note.end_time_ms) for note in notes]
        ln_indices = sorted(
            [index for index, note in enumerate(working) if note.is_ln],
            key=lambda index: (
                context.get(working[index].time_ms, TechPoint(working[index].time_ms, 0, 0, 0, 0, 0, 0, 0, 0, 1, False, False, True)).sustain,
                context.get(working[index].time_ms, TechPoint(working[index].time_ms, 0, 0, 0, 0, 0, 0, 0, 0, 1, False, False, True)).release,
            ),
        )
        for index in ln_indices:
            if sr <= target + tolerance:
                break
            note = working[index]
            working[index] = NoteObject(note.time_ms, note.lane)
            sr = DifficultyEstimator.estimate_sr(working, duration)
        notes = working
        if sr <= target + tolerance:
            fixed = Validator.validate_and_fix(notes, config, analysis.get("silent_regions", []), snap_points)
            return fixed, DifficultyEstimator.estimate_sr(fixed, duration)
    rows = _rows_from_notes(notes)
    original_by_slot = {(note.time_ms, note.lane): note for note in notes}
    rows, sr = _lower_sr(rows, context, target, tolerance, duration)
    fixed = Validator.validate_and_fix(_notes_from_rows(rows, original_by_slot), config, analysis.get("silent_regions", []), snap_points)
    return fixed, DifficultyEstimator.estimate_sr(fixed, duration)


def _apply_lns(
    notes: List[NoteObject],
    context: Dict[int, TechPoint],
    snap_points: List[int],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    beat: float,
) -> List[NoteObject]:
    if config.chart_type != "ln" or config.ln_ratio <= 0.02 or not notes:
        return notes
    ratio = max(0.0, min(0.85, config.ln_ratio))
    interval = max(2, int(round(1.0 / max(0.08, ratio))))
    sorted_notes = sorted(notes, key=lambda note: (note.time_ms, note.lane))
    next_same = _next_same_lane(sorted_notes)
    result: List[NoteObject] = []
    lane_tail = {lane: -999999 for lane in LANES}
    for index, note in enumerate(sorted_notes):
        point = context.get(note.time_ms)
        copied = NoteObject(note.time_ms, note.lane)
        if point is None:
            result.append(copied)
            continue
        should_ln = (
            point.sustain >= 0.44
            and point.attack <= 0.76
            and point.dump_risk <= 0.62
            and index % interval == (note.lane + index // 4) % interval
            and note.time_ms > lane_tail[note.lane] + 30
        )
        if should_ln:
            tail = _choose_ln_tail(note.time_ms, next_same.get(index), context, snap_points, beat, config)
            if tail is not None:
                copied.end_time_ms = tail
                lane_tail[note.lane] = tail
        result.append(copied)
    return result


def _choose_ln_tail(
    head_time: int,
    next_same_time: Optional[int],
    context: Dict[int, TechPoint],
    snap_points: List[int],
    beat: float,
    config: DifficultyConfig,
) -> Optional[int]:
    min_len = max(30, min(int(config.min_ln_ms), int(beat / 12)))
    max_len = min(int(config.max_ln_ms), int(beat * 1.25))
    if next_same_time is not None:
        max_len = min(max_len, next_same_time - head_time - 30)
    if max_len < min_len:
        return None
    lo = bisect.bisect_left(snap_points, head_time + min_len)
    hi = bisect.bisect_right(snap_points, head_time + max_len)
    candidates = [time_ms for time_ms in snap_points[lo:hi] if time_ms in context]
    if not candidates:
        return None
    head = context.get(head_time)
    if head is None:
        return None
    def score(time_ms: int) -> float:
        tail = context[time_ms]
        sustain_drop = max(0.0, head.sustain - tail.sustain)
        length_pref = 1.0 - min(1.0, abs((time_ms - head_time) - beat * 0.50) / max(1.0, beat))
        return tail.release * 0.55 + sustain_drop * 0.24 + length_pref * 0.16 - tail.attack * 0.18
    chosen = max(candidates, key=score)
    return chosen if score(chosen) >= 0.10 else None


def _build_context(
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> Dict[int, TechPoint]:
    duration = max(1, int(analysis.get("duration_ms", 1)))
    beat = 60000.0 / max(1.0, float(analysis.get("bpm", 120.0)))
    max_dist = max(28, min(72, int(beat * 0.16)))
    energy = _normalise(analysis.get("energy_curve", []))
    sustain = _normalise(analysis.get("sustain_curve", []))
    vocal = _normalise(analysis.get("vocal_sustain_curve", []))
    release = _normalise(analysis.get("release_curve", []))
    onset_by_snap, bass_by_snap, stack_by_snap = _snap_onsets(analysis, snap_points, max_dist)
    beat_times = sorted(int(t) for t in analysis.get("beat_times_ms", []))
    context: Dict[int, TechPoint] = {}

    for time_ms in snap_points:
        silent = _in_regions(time_ms, analysis.get("silent_regions", []))
        e = _series_at(energy, time_ms, duration)
        s = max(_series_at(sustain, time_ms, duration), _series_at(vocal, time_ms, duration) * 1.06)
        r = _series_at(release, time_ms, duration)
        prev_e = _series_at(energy, max(0, time_ms - max_dist), duration)
        energy_attack = max(0.0, min(1.0, (e - prev_e) * 1.75))
        onset = onset_by_snap.get(time_ms, 0.0)
        bass = bass_by_snap.get(time_ms, 0.0)
        stack = min(1.0, stack_by_snap.get(time_ms, 0) / 2.0)
        beat_score = _near_score(beat_times, time_ms, max_dist)
        attack = max(onset, energy_attack, r * 0.30 + onset * 0.42)
        texture = max(e, s * 0.90)
        accent = max(attack * 0.96, bass * 1.16, stack * 0.42 + onset * 0.70, energy_attack * 0.84)
        support = max(attack, bass * 0.92, texture * 0.76, s * 0.58, beat_score * max(texture, attack) * 0.42)
        dump_risk = 1.0 - max(attack * 1.04, bass * 0.94, texture * 0.84, s * 0.62)
        dump_risk = max(0.0, min(1.0, dump_risk))
        must_hit = (
            not silent
            and dump_risk <= 0.62
            and (
                attack >= 0.46
                or bass >= 0.40
                or accent >= 0.58
                or (time_ms in accent_snap_points and support >= 0.34)
                or (r >= 0.70 and support >= 0.34)
            )
        )
        can_hit = not silent and dump_risk <= 0.82 and support >= 0.14
        if silent:
            attack = bass = texture = s = r = beat_score = accent = support = 0.0
            dump_risk = 1.0
            must_hit = False
            can_hit = False
        context[time_ms] = TechPoint(
            time_ms=time_ms,
            attack=max(0.0, min(1.0, attack)),
            kick=max(0.0, min(1.0, bass)),
            texture=max(0.0, min(1.0, texture)),
            sustain=max(0.0, min(1.0, s)),
            release=max(0.0, min(1.0, r)),
            beat=max(0.0, min(1.0, beat_score)),
            accent=max(0.0, min(1.0, accent)),
            support=max(0.0, min(1.0, support)),
            dump_risk=dump_risk,
            must_hit=must_hit,
            can_hit=can_hit,
            silent=silent,
        )
    return context


def _snap_onsets(
    analysis: Dict[str, Any],
    snap_points: List[int],
    max_dist: int,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, int]]:
    onset_by_snap: Dict[int, float] = {}
    bass_by_snap: Dict[int, float] = {}
    stack_by_snap: Dict[int, int] = {}
    peaks = analysis.get("onset_peaks") or [{"time_ms": int(t), "strength": 0.6, "bass": 0.0} for t in analysis.get("onset_times_ms", [])]
    for peak in peaks:
        if isinstance(peak, dict):
            raw = int(peak.get("time_ms", 0))
            strength = float(peak.get("strength", 0.0))
            bass = float(peak.get("bass", 0.0))
        else:
            raw = int(peak[0])
            strength = float(peak[1]) if len(peak) > 1 else 0.6
            bass = float(peak[2]) if len(peak) > 2 else 0.0
        snap = _nearest_snap(snap_points, raw)
        if snap is None:
            continue
        dist = abs(snap - raw)
        if dist > max_dist:
            continue
        weight = 1.0 - dist / max(1.0, max_dist)
        onset_by_snap[snap] = max(onset_by_snap.get(snap, 0.0), max(0.0, min(1.0, strength)) * weight)
        bass_by_snap[snap] = max(bass_by_snap.get(snap, 0.0), max(0.0, min(1.0, bass)) * weight)
        stack_by_snap[snap] = stack_by_snap.get(snap, 0) + 1
    return onset_by_snap, bass_by_snap, stack_by_snap


def _insert_row(rows: List[int], time_ms: int, context: Dict[int, TechPoint], min_gap: int) -> List[int]:
    rows = sorted(rows)
    idx = bisect.bisect_left(rows, time_ms)
    neighbours = []
    if idx > 0 and time_ms - rows[idx - 1] < min_gap:
        neighbours.append(rows[idx - 1])
    if idx < len(rows) and rows[idx] - time_ms < min_gap:
        neighbours.append(rows[idx])
    if not neighbours:
        rows.insert(idx, time_ms)
        return rows
    if all(context[time_ms].score > context[old].score + 0.03 for old in neighbours):
        rows = [old for old in rows if old not in neighbours]
        bisect.insort(rows, time_ms)
    return rows


def _notes_from_rows(
    rows: Dict[int, Set[int]],
    original_by_slot: Optional[Dict[Tuple[int, int], NoteObject]] = None,
) -> List[NoteObject]:
    result: List[NoteObject] = []
    for time_ms in sorted(rows):
        for lane in sorted(rows[time_ms]):
            original = original_by_slot.get((time_ms, lane)) if original_by_slot else None
            if original is not None:
                result.append(NoteObject(time_ms=time_ms, lane=lane, end_time_ms=original.end_time_ms))
            else:
                result.append(NoteObject(time_ms=time_ms, lane=lane))
    return result


def _rows_from_notes(notes: List[NoteObject]) -> Dict[int, Set[int]]:
    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)
    return rows


def _next_same_lane(notes: List[NoteObject]) -> Dict[int, int]:
    by_lane: Dict[int, List[Tuple[int, int]]] = {lane: [] for lane in LANES}
    for index, note in enumerate(notes):
        by_lane[note.lane].append((index, note.time_ms))
    result: Dict[int, int] = {}
    for items in by_lane.values():
        for local_index, (index, _) in enumerate(items[:-1]):
            result[index] = items[local_index + 1][1]
    return result


def _extra_lane(existing: Set[int], pressure: Dict[int, int]) -> Optional[int]:
    choices = [lane for lane in LANES if lane not in existing]
    if not choices:
        return None
    return min(choices, key=lambda lane: (pressure[lane], -min(abs(lane - old) for old in existing), lane))


def _best_single(lanes: Set[int], time_ms: int) -> Set[int]:
    if not lanes:
        return set()
    return {min(lanes, key=lambda lane: (_deterministic_index(1000, time_ms, lane), lane))}


def _quality(notes: List[NoteObject], context: Dict[int, TechPoint], beat: float) -> float:
    rows = _rows_from_notes(notes)
    if not rows:
        return float("-inf")
    times = sorted(rows)
    unsupported = sum(1 for time_ms in times if context.get(time_ms) is None or context[time_ms].dump_risk > 0.82)
    supported_times = sorted(t for t, point in context.items() if point.can_hit and point.support >= 0.22)
    bad_gaps = 0
    for left, right in zip(times, times[1:]):
        if right - left <= max(beat * 0.75, 260):
            continue
        idx = bisect.bisect_right(supported_times, left)
        if idx < len(supported_times) and supported_times[idx] < right:
            bad_gaps += 1
    music_score = sum(context[t].score for t in times if t in context) / max(1, len(times))
    return music_score - unsupported * 0.08 - bad_gaps * 0.25


def _min_row_gap(beat: float, target: float, divisor: int) -> int:
    step = beat / max(1, divisor)
    if target >= 6.8:
        return max(24, int(step * 0.48))
    if target >= 5.8:
        return max(34, int(step * 0.56))
    return max(44, int(step * 0.68))


def _desired_gap(beat: float, target: float, shift: float = 0.0) -> int:
    if target < 4.2:
        base = max(beat / 4.0, 105.0)
    elif target < 5.4:
        base = max(beat / 6.0, 78.0)
    elif target < 6.6:
        base = max(beat / 8.0, 56.0)
    else:
        base = max(beat / 8.0, 48.0)
    return int(round(base * (1.0 + shift)))


def _same_lane_gap(beat: float, target: float) -> int:
    if target >= 6.4:
        return max(48, int(beat * 0.15))
    if target >= 5.2:
        return max(58, int(beat * 0.20))
    return max(70, int(beat * 0.25))


def _max_chord_run(target: float) -> int:
    if target < 5.4:
        return 2
    if target < 6.4:
        return 3
    return 4


def _normalise(values: Iterable[float]) -> List[float]:
    series = [float(v) for v in values]
    if not series:
        return []
    ordered = sorted(series)
    low = ordered[min(len(ordered) - 1, int(len(ordered) * 0.12))]
    high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.92))]
    if high <= low + 1e-9:
        high = max(series)
        low = min(series)
    if high <= low + 1e-9:
        return [0.0 for _ in series]
    return [max(0.0, min(1.0, (value - low) / (high - low))) for value in series]


def _series_at(series: List[float], time_ms: int, duration_ms: int) -> float:
    if not series:
        return 0.0
    index = int(round(max(0, min(duration_ms, time_ms)) / max(1, duration_ms) * (len(series) - 1)))
    return float(series[max(0, min(len(series) - 1, index))])


def _near_score(times: List[int], time_ms: int, max_dist: int) -> float:
    if not times:
        return 0.0
    idx = bisect.bisect_left(times, time_ms)
    best = 0.0
    for probe in (idx - 1, idx, idx + 1):
        if 0 <= probe < len(times):
            dist = abs(times[probe] - time_ms)
            if dist <= max_dist:
                best = max(best, 1.0 - dist / max(1.0, max_dist))
    return best


def _nearest_snap(snap_points: List[int], time_ms: int) -> Optional[int]:
    if not snap_points:
        return None
    idx = bisect.bisect_left(snap_points, time_ms)
    candidates = []
    if idx < len(snap_points):
        candidates.append(snap_points[idx])
    if idx > 0:
        candidates.append(snap_points[idx - 1])
    return min(candidates, key=lambda snap: abs(snap - time_ms)) if candidates else None


def _in_regions(time_ms: int, regions: List[Tuple[int, int]]) -> bool:
    return any(int(start) <= time_ms <= int(end) for start, end in regions)


def _target_norm(target: float) -> float:
    return max(0.0, min(1.0, (target - 3.0) / 4.0))


def _deterministic_index(modulus: int, *values: int) -> int:
    if modulus <= 0:
        return 0
    seed = 2166136261
    for value in values:
        seed ^= int(value) & 0xFFFFFFFF
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed % modulus

import bisect
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.difficulty_estimator import DifficultyEstimator
from ..core.models import DifficultyConfig, NoteObject
from ..core.style_rules import clamp_max_chord_size
from ..core.validator import Validator

LANES = (0, 1, 2, 3)


@dataclass(frozen=True)
class SpeedPoint:
    time_ms: int
    attack: float
    kick: float
    energy: float
    beat: float
    accent: float
    support: float
    silent: bool


def generate_speed_to_target_sr(
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
    best_notes: List[NoteObject] = []
    best_sr = 0.0
    best_diff = float("inf")
    best_quality = float("-inf")
    best_attempt = 0

    for attempt, variant in enumerate(_variants(config, target, max_attempts), start=1):
        rows = _select_speed_rows(analysis, snap_points, context, target, variant)
        if not rows:
            continue
        chord_sizes = _speed_chord_sizes(rows, context, target, config, variant)
        notes = _assign_speed_lanes(rows, chord_sizes, context, target, variant)
        notes = _repair_speed_shape(notes, target)
        if config.chart_type == "ln":
            notes = _apply_speed_lns(notes, rows, context, config, analysis)
        notes = Validator.validate_and_fix(notes, config, analysis.get("silent_regions", []), snap_points, min_interval_ms=32)
        notes = _repair_speed_shape(notes, target)
        notes = Validator.validate_and_fix(notes, config, analysis.get("silent_regions", []), snap_points, min_interval_ms=32)
        sr = DifficultyEstimator.estimate_sr(notes, int(analysis.get("duration_ms", 0)))
        notes, sr = _trim_or_boost(notes, rows, context, target, tolerance, config, analysis, snap_points)

        diff = abs(sr - target)
        quality = _speed_quality(notes, context)
        if diff < best_diff or (diff <= best_diff + 0.04 and quality > best_quality):
            best_notes = notes
            best_sr = sr
            best_diff = diff
            best_quality = quality
            best_attempt = attempt
        if diff <= tolerance:
            return notes, sr, True, attempt

    return best_notes, best_sr, best_diff <= tolerance, best_attempt


def _variants(config: DifficultyConfig, target: float, max_attempts: int) -> List[Dict[str, float]]:
    allowed = _allowed_divisors(config)
    preferred = 6 if target < 4.5 else 8 if target < 5.7 else 12 if target < 6.8 else 16
    divisors = sorted(allowed, key=lambda divisor: (abs(divisor - preferred), -divisor))
    variants: List[Dict[str, float]] = []
    for divisor in divisors:
        for density_shift, chord_shift, seed in [
            (0.00, 0.00, 0),
            (0.05, -0.02, 1),
            (-0.04, 0.03, 2),
            (0.08, 0.02, 3),
            (-0.08, 0.05, 4),
        ]:
            variants.append({
                "divisor": float(divisor),
                "density_shift": density_shift,
                "chord_shift": chord_shift,
                "seed": float(seed),
            })
    return variants[: max(1, max_attempts)]


def _allowed_divisors(config: DifficultyConfig) -> List[int]:
    divisors: List[int] = []
    for value in config.allowed_subdivisions or []:
        try:
            if "/" in value:
                divisors.append(int(value.split("/", 1)[1]))
        except (TypeError, ValueError):
            continue
    divisors = sorted({d for d in divisors if d >= 2})
    return divisors or [4, 6, 8, 10, 12]


def _select_speed_rows(
    analysis: Dict[str, Any],
    snap_points: List[int],
    context: Dict[int, SpeedPoint],
    target: float,
    variant: Dict[str, float],
) -> List[int]:
    divisor = max(1, int(variant["divisor"]))
    beat = 60000.0 / max(1.0, float(analysis.get("bpm", 120.0)))
    step = beat / divisor
    snap_set = set(snap_points)
    rows: List[int] = []
    t = float(analysis.get("offset_ms", 0))
    while t < float(analysis.get("duration_ms", 0)):
        rounded = int(round(t))
        if rounded in snap_set and not _in_silent(rounded, analysis.get("silent_regions", [])):
            rows.append(rounded)
        t += step
    if not rows:
        return []

    density = _target_density(target) + float(variant.get("density_shift", 0.0))
    density = max(0.50, min(0.86, density))
    masks = _speed_masks(target)
    shaped: List[int] = []
    phrase = 0
    index = 0
    while index < len(rows):
        mask = masks[_dindex(len(masks), phrase, int(target * 100), int(variant.get("seed", 0)))]
        phrase_rows = rows[index : index + len(mask)]
        for local, time_ms in enumerate(phrase_rows):
            point = context.get(time_ms)
            force = point is not None and (
                point.accent >= 0.60
                or point.kick >= 0.54
                or point.attack >= 0.58
            )
            keep = mask[local % len(mask)] == 1
            if force or keep:
                shaped.append(time_ms)
        index += len(mask)
        phrase += 1

    desired_count = int(round(len(rows) * density))
    if len(shaped) < desired_count:
        shaped_set = set(shaped)
        candidates = sorted(
            [time_ms for time_ms in rows if time_ms not in shaped_set],
            key=lambda time_ms: (
                -context.get(time_ms, SpeedPoint(time_ms, 0, 0, 0, 0, 0, 0, False)).support,
                _dindex(997, time_ms, int(target * 100)),
            ),
        )
        for time_ms in candidates[: desired_count - len(shaped)]:
            shaped.append(time_ms)
        shaped.sort()
    elif len(shaped) > desired_count:
        removable = sorted(
            shaped,
            key=lambda time_ms: (
                context.get(time_ms, SpeedPoint(time_ms, 0, 0, 0, 0, 0, 0, False)).accent,
                _dindex(991, time_ms, int(target * 100)),
            ),
        )
        remove = set(removable[: len(shaped) - desired_count])
        shaped = [time_ms for time_ms in shaped if time_ms not in remove]

    return shaped


def _target_density(target: float) -> float:
    if target < 4.0:
        return 0.56
    if target < 5.0:
        return 0.61
    if target < 6.0:
        return 0.66
    if target < 7.0:
        return 0.80
    return 0.80


def _speed_masks(target: float) -> List[List[int]]:
    if target < 5.0:
        return [
            [1, 0, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 0],
        ]
    return [
        [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    ]


def _speed_chord_sizes(
    rows: List[int],
    context: Dict[int, SpeedPoint],
    target: float,
    config: DifficultyConfig,
    variant: Dict[str, float],
) -> List[int]:
    max_size = min(3, clamp_max_chord_size(config, "speed"))
    sizes = [1 for _ in rows]
    if max_size <= 1:
        return sizes

    ratio = _target_chord_ratio(target) + float(variant.get("chord_shift", 0.0))
    ratio = max(0.02, min(0.18, ratio))
    target_chords = int(round(len(rows) * ratio))
    candidates = sorted(
        range(len(rows)),
        key=lambda i: (
            -context.get(rows[i], SpeedPoint(rows[i], 0, 0, 0, 0, 0, 0, False)).accent,
            _dindex(977, rows[i], i, int(target * 100)),
        ),
    )
    chosen: Set[int] = set()
    min_spacing = _speed_chord_spacing_rows(target)
    for index in candidates:
        if len(chosen) >= target_chords:
            break
        if _too_far_from_music(rows[index], context):
            continue
        if _too_close_to_chosen_chord(index, chosen, min_spacing):
            continue
        chosen.add(index)

    for index in chosen:
        sizes[index] = 2

    if max_size >= 3 and target >= 5.9:
        triple_budget = int(round(len(rows) * (0.008 if target < 6.5 else 0.012)))
        triple_candidates = sorted(
            [i for i in chosen if context.get(rows[i]) and context[rows[i]].accent >= 0.68],
            key=lambda i: (-context[rows[i]].accent, i),
        )
        for index in triple_candidates[:triple_budget]:
            sizes[index] = 3

    _shape_chord_bursts(sizes, rows, context, target, max_size)
    _limit_speed_chord_runs(sizes, rows, context, target)
    return sizes


def _target_chord_ratio(target: float) -> float:
    if target < 4.5:
        return 0.04
    if target < 5.5:
        return 0.07
    if target < 6.5:
        return 0.10
    return 0.12


def _speed_chord_spacing_rows(target: float) -> int:
    if target < 5.0:
        return 8
    if target < 6.2:
        return 6
    return 5


def _too_close_to_chosen_chord(index: int, chosen: Set[int], min_spacing: int) -> bool:
    return any(abs(index - other) < min_spacing for other in chosen)


def _too_far_from_music(time_ms: int, context: Dict[int, SpeedPoint]) -> bool:
    point = context.get(time_ms)
    if point is None:
        return False
    return point.support < 0.18 and point.accent < 0.24


def _shape_chord_bursts(
    sizes: List[int],
    rows: List[int],
    context: Dict[int, SpeedPoint],
    target: float,
    max_size: int,
) -> None:
    if max_size < 3 or target < 6.2 or len(rows) < 32:
        return
    block = 96
    min_spacing = _speed_chord_spacing_rows(target)
    for start in range(0, len(rows), block):
        end = min(len(rows), start + block)
        local_chords = [i for i in range(start, end) if sizes[i] >= 2]
        if not local_chords:
            continue
        best = max(
            local_chords,
            key=lambda i: (
                context.get(rows[i], SpeedPoint(rows[i], 0, 0, 0, 0, 0, 0, False)).accent,
                -_dindex(997, rows[i], i),
            ),
        )
        point = context.get(rows[best])
        if point and point.accent >= 0.76 and not _too_close_to_chosen_chord(
            best,
            {i for i in local_chords if i != best and sizes[i] >= 3},
            min_spacing * 2,
        ):
            sizes[best] = 3


def _limit_speed_chord_runs(
    sizes: List[int],
    rows: List[int],
    context: Dict[int, SpeedPoint],
    target: float,
) -> None:
    max_run = 1
    start: Optional[int] = None
    for index, size in enumerate(sizes + [1]):
        if size >= 2:
            if start is None:
                start = index
            continue
        if start is None:
            continue
        end = index
        while end - start > max_run:
            window = list(range(start, end))
            downgrade = sorted(
                window,
                key=lambda i: (
                    context.get(rows[i], SpeedPoint(rows[i], 0, 0, 0, 0, 0, 0, False)).accent,
                    -abs(i - (start + end) // 2),
                    _dindex(389, rows[i], i, int(target * 100)),
                ),
            )[0]
            sizes[downgrade] = 1
            if downgrade == start:
                start += 1
            else:
                end = downgrade
        start = None


def _assign_speed_lanes(
    rows: List[int],
    sizes: List[int],
    context: Dict[int, SpeedPoint],
    target: float,
    variant: Dict[str, float],
) -> List[NoteObject]:
    notes: List[NoteObject] = []
    recent: List[List[int]] = []
    module = "random"
    module_left = 0
    state = {
        "stair_dir": 1,
        "stair_lane": _dindex(4, int(variant.get("seed", 0)), int(target * 100)),
    }

    for row_index, (time_ms, size) in enumerate(zip(rows, sizes)):
        if module_left <= 0:
            module = _choose_module(row_index, time_ms, context, target, variant)
            module_left = _module_length(module, row_index, target)
            if module in ["stair", "reverse"]:
                state["stair_dir"] = 1 if _dindex(2, row_index, time_ms) == 0 else -1
                state["stair_lane"] = _dindex(4, row_index, time_ms, int(target * 100))
        lanes = _next_lanes(size, recent, module, state, row_index, time_ms, target)
        notes.extend(NoteObject(time_ms=time_ms, lane=lane) for lane in lanes)
        recent.append(lanes)
        if len(recent) > 32:
            recent.pop(0)
        module_left -= 1

    return notes


def _choose_module(
    row_index: int,
    time_ms: int,
    context: Dict[int, SpeedPoint],
    target: float,
    variant: Dict[str, float],
) -> str:
    point = context.get(time_ms)
    accent = point.accent if point else 0.0
    modules = ["stair", "reverse", "wide", "random", "cross"]
    if accent >= 0.62:
        modules = ["wide", "cross", "random", "stair"]
    return modules[_dindex(len(modules), row_index, time_ms, int(target * 100), int(variant.get("seed", 0)))]


def _module_length(module: str, row_index: int, target: float) -> int:
    base = {
        "stair": 6,
        "reverse": 6,
        "wide": 5,
        "random": 4,
        "cross": 5,
    }.get(module, 4)
    return base + _dindex(4, row_index, int(target * 100))


def _next_lanes(
    size: int,
    recent: List[List[int]],
    module: str,
    state: Dict[str, int],
    row_index: int,
    time_ms: int,
    target: float,
) -> List[int]:
    patterns = _patterns(size)
    previous = set(recent[-1]) if recent else set()
    previous_single = recent[-1][0] if recent and len(recent[-1]) == 1 else None

    preferred: List[int] = []
    if module in ["stair", "reverse"] and size == 1:
        lane = int(state.get("stair_lane", 0))
        direction = int(state.get("stair_dir", 1))
        preferred = [[lane]]
        next_lane = lane + direction
        if next_lane < 0 or next_lane > 3:
            direction *= -1
            next_lane = lane + direction
        state["stair_dir"] = direction
        state["stair_lane"] = max(0, min(3, next_lane))
    elif module == "wide":
        preferred = [[0, 3], [0], [3], [1, 3], [0, 2]]
    elif module == "cross":
        preferred = [[0, 2], [1, 3], [0, 3], [1], [2]]

    candidates = preferred + patterns
    unique: List[List[int]] = []
    seen: Set[Tuple[int, ...]] = set()
    for lanes in candidates:
        lanes = sorted(lanes[:size] if len(lanes) > size else lanes)
        if len(lanes) != size or len(set(lanes)) != size:
            continue
        key = tuple(lanes)
        if key not in seen:
            unique.append(lanes)
            seen.add(key)
    candidates = unique or patterns

    counts = Counter(lane for lanes in recent[-16:] for lane in lanes)
    recent_patterns = Counter(tuple(lanes) for lanes in recent[-10:])

    def score(lanes: List[int]) -> Tuple[int, int, int, int, int, int]:
        lane_set = set(lanes)
        direct_repeat = 1 if size == 1 and previous_single is not None and lanes[0] == previous_single else 0
        if size == 1 and _would_make_long_abab(recent, lanes[0]):
            direct_repeat += 2
        cooldown = _speed_lane_cooldown_penalty(lanes, recent, size, target)
        overlap = len(lane_set & previous)
        if size == 1:
            overlap *= 8 if len(previous) >= 2 else 4
        elif len(previous) >= 2:
            overlap *= 4
        else:
            overlap *= 2
        pressure = sum(counts.get(lane, 0) for lane in lanes)
        return (
            direct_repeat,
            cooldown,
            overlap,
            recent_patterns.get(tuple(lanes), 0),
            pressure,
            _dindex(997, row_index, time_ms, sum(lanes), int(target * 100)),
        )

    return list(sorted(candidates, key=score)[0])


def _speed_lane_cooldown_penalty(
    lanes: List[int],
    recent: List[List[int]],
    size: int,
    target: float,
) -> int:
    min_distance = _speed_single_lane_min_row_distance(target)
    penalty = 0
    for lane in lanes:
        distance = _recent_lane_distance(recent, lane)
        if distance <= min_distance:
            weight = 18 if size == 1 else 6
            penalty += (min_distance - distance + 1) * weight
    return penalty


def _speed_single_lane_min_row_distance(target: float) -> int:
    if target < 6.8:
        return 2
    return 3


def _recent_lane_distance(recent: List[List[int]], lane: int) -> int:
    for distance, lanes in enumerate(reversed(recent), start=1):
        if lane in lanes:
            return distance
    return 999


def _would_make_long_abab(recent: List[List[int]], lane: int) -> bool:
    singles: List[int] = []
    for lanes in reversed(recent):
        if len(lanes) != 1:
            break
        singles.append(lanes[0])
        if len(singles) >= 5:
            break
    if len(singles) < 3:
        return False
    return lane == singles[1] and singles[0] == singles[2]


def _patterns(size: int) -> List[List[int]]:
    if size <= 1:
        return [[0], [1], [2], [3]]
    if size == 2:
        return [[0, 3], [1, 2], [0, 2], [1, 3], [0, 1], [2, 3]]
    return [[0, 1, 3], [0, 2, 3], [0, 1, 2], [1, 2, 3]]


def _repair_speed_shape(notes: List[NoteObject], target: float) -> List[NoteObject]:
    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    repaired: List[NoteObject] = []
    previous_lanes: Set[int] = set()
    previous_single: Optional[int] = None
    last_lane_time = {lane: -999999 for lane in LANES}
    min_single_gap_ms = _speed_single_lane_min_gap_ms(target)
    for time_ms in sorted(rows):
        row = [
            NoteObject(note.time_ms, note.lane, note.end_time_ms)
            for note in sorted(rows[time_ms], key=lambda n: (n.lane, n.end_time_ms or -1))
        ]
        if len(row) == 1:
            if previous_single is not None and row[0].lane == previous_single:
                row[0].lane = _replacement_lane(row[0].lane, rows, time_ms)
            if len(previous_lanes) >= 2 and row[0].lane in previous_lanes:
                row[0].lane = _replacement_lane_avoiding(row[0].lane, previous_lanes, rows, time_ms)
            crowded_lanes = {
                lane
                for lane, last_time in last_lane_time.items()
                if time_ms - last_time < min_single_gap_ms
            }
            if row[0].lane in crowded_lanes:
                safe_lanes = [lane for lane in LANES if lane not in crowded_lanes]
                if not safe_lanes:
                    previous_lanes = set()
                    previous_single = None
                    continue
                row[0].lane = sorted(
                    safe_lanes,
                    key=lambda lane: (
                        Counter(note.lane for values in rows.values() for note in values).get(lane, 0),
                        _dindex(461, time_ms, lane),
                    ),
                )[0]
            previous_single = row[0].lane
        else:
            crowded_lanes = {
                lane
                for lane, last_time in last_lane_time.items()
                if time_ms - last_time < min_single_gap_ms
            }
            if any(note.lane in crowded_lanes for note in row):
                row = _repair_crowded_speed_chord(row, crowded_lanes, rows, time_ms)
            if not row:
                previous_lanes = set()
                previous_single = None
                continue
            if len(previous_lanes) >= 2:
                row = _repair_chord_overlap(row, previous_lanes, rows, time_ms)
            previous_single = None
        previous_lanes = {note.lane for note in row}
        for note in row:
            last_lane_time[note.lane] = time_ms
        repaired.extend(row)
    return sorted(repaired, key=lambda n: (n.time_ms, n.lane, n.end_time_ms or -1))


def _speed_single_lane_min_gap_ms(target: float) -> int:
    if target < 5.0:
        return 110
    if target < 6.8:
        return 100
    return 92


def _repair_crowded_speed_chord(
    row: List[NoteObject],
    crowded_lanes: Set[int],
    rows: Dict[int, List[NoteObject]],
    time_ms: int,
) -> List[NoteObject]:
    counts = Counter(note.lane for values in rows.values() for note in values)
    safe_lanes = [lane for lane in LANES if lane not in crowded_lanes]
    if len(safe_lanes) >= len(row):
        candidates = [
            pattern
            for pattern in _patterns(len(row))
            if not (set(pattern) & crowded_lanes)
        ]
        if candidates:
            best = sorted(
                candidates,
                key=lambda cand: (
                    sum(counts.get(lane, 0) for lane in cand),
                    _dindex(443, time_ms, sum(cand)),
                ),
            )[0]
            return [
                NoteObject(row[0].time_ms, lane, row[i].end_time_ms if i < len(row) else None)
                for i, lane in enumerate(best)
            ]
    if safe_lanes:
        lane = sorted(safe_lanes, key=lambda l: (counts.get(l, 0), _dindex(457, time_ms, l)))[0]
        return [NoteObject(row[0].time_ms, lane, row[0].end_time_ms)]
    return []


def _replacement_lane(current: int, rows: Dict[int, List[NoteObject]], time_ms: int) -> int:
    counts = Counter(note.lane for row in rows.values() for note in row)
    candidates = [lane for lane in LANES if lane != current]
    return sorted(candidates, key=lambda lane: (counts.get(lane, 0), _dindex(97, time_ms, current, lane)))[0]


def _replacement_lane_avoiding(
    current: int,
    avoid: Set[int],
    rows: Dict[int, List[NoteObject]],
    time_ms: int,
) -> int:
    counts = Counter(note.lane for row in rows.values() for note in row)
    candidates = [lane for lane in LANES if lane != current and lane not in avoid]
    if not candidates:
        candidates = [lane for lane in LANES if lane != current]
    return sorted(candidates, key=lambda lane: (counts.get(lane, 0), _dindex(193, time_ms, current, lane)))[0]


def _repair_chord_overlap(
    row: List[NoteObject],
    previous_lanes: Set[int],
    rows: Dict[int, List[NoteObject]],
    time_ms: int,
) -> List[NoteObject]:
    lanes = {note.lane for note in row}
    if not lanes & previous_lanes:
        return row
    counts = Counter(note.lane for values in rows.values() for note in values)
    candidates = _patterns(len(row))
    best = sorted(
        candidates,
        key=lambda cand: (
            len(set(cand) & previous_lanes),
            sum(counts.get(lane, 0) for lane in cand),
            _dindex(271, time_ms, sum(cand)),
        ),
    )[0]
    return [
        NoteObject(row[0].time_ms, lane, row[i].end_time_ms if i < len(row) else None)
        for i, lane in enumerate(best)
    ]


def _apply_speed_lns(
    notes: List[NoteObject],
    rows: List[int],
    context: Dict[int, SpeedPoint],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
) -> List[NoteObject]:
    if config.chart_type != "ln" or not notes:
        return notes
    row_index_by_time = {time_ms: i for i, time_ms in enumerate(rows)}
    ln_ratio = max(0.0, min(0.75, config.ln_ratio))
    budget = int(round(len(notes) * ln_ratio * 0.42))
    if budget <= 0:
        return notes
    converted: List[NoteObject] = []
    used_lane_until = {lane: -1 for lane in LANES}
    made = 0
    for note in notes:
        point = context.get(note.time_ms)
        idx = row_index_by_time.get(note.time_ms)
        can_ln = (
            made < budget
            and point is not None
            and idx is not None
            and point.support >= 0.36
            and note.time_ms > used_lane_until[note.lane] + 35
            and _dindex(100, note.time_ms, note.lane) < int(ln_ratio * 45)
        )
        if not can_ln:
            converted.append(note)
            continue
        length_rows = 2 + _dindex(3, note.time_ms, note.lane, int(config.target_star or 5))
        end_idx = min(len(rows) - 1, idx + length_rows)
        end_time = rows[end_idx]
        if end_time <= note.time_ms:
            converted.append(note)
            continue
        used_lane_until[note.lane] = end_time
        converted.append(NoteObject(note.time_ms, note.lane, end_time))
        made += 1
    return converted


def _trim_or_boost(
    notes: List[NoteObject],
    rows: List[int],
    context: Dict[int, SpeedPoint],
    target: float,
    tolerance: float,
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
) -> Tuple[List[NoteObject], float]:
    sr = DifficultyEstimator.estimate_sr(notes, int(analysis.get("duration_ms", 0)))
    if abs(sr - target) <= tolerance:
        return notes, sr
    if sr > target + tolerance:
        notes = _trim_weak_notes(notes, context, target, analysis, snap_points)
    elif sr < target - tolerance:
        notes = _boost_strong_chords(notes, context, config, target)
    notes = Validator.validate_and_fix(notes, config, analysis.get("silent_regions", []), snap_points, min_interval_ms=32)
    notes = _repair_speed_shape(notes, target)
    notes = Validator.validate_and_fix(notes, config, analysis.get("silent_regions", []), snap_points, min_interval_ms=32)
    sr = DifficultyEstimator.estimate_sr(notes, int(analysis.get("duration_ms", 0)))
    return notes, sr


def _trim_weak_notes(
    notes: List[NoteObject],
    context: Dict[int, SpeedPoint],
    target: float,
    analysis: Dict[str, Any],
    snap_points: List[int],
) -> List[NoteObject]:
    rows: Dict[int, List[NoteObject]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
    row_times = sorted(rows)
    remove_count = max(1, int(round(len(row_times) * 0.025)))
    candidates = sorted(
        [
            time_ms
            for time_ms in row_times
            if len(rows[time_ms]) == 1
            and context.get(time_ms, SpeedPoint(time_ms, 0, 0, 0, 0, 0, 0, False)).accent < 0.42
        ],
        key=lambda time_ms: (
            context.get(time_ms, SpeedPoint(time_ms, 0, 0, 0, 0, 0, 0, False)).support,
            _dindex(811, time_ms, int(target * 100)),
        ),
    )
    remove = set(candidates[:remove_count])
    return [note for note in notes if note.time_ms not in remove]


def _boost_strong_chords(
    notes: List[NoteObject],
    context: Dict[int, SpeedPoint],
    config: DifficultyConfig,
    target: float,
) -> List[NoteObject]:
    max_size = min(2 if target < 5.8 else 3, clamp_max_chord_size(config, "speed"))
    if max_size <= 1:
        return notes
    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)
    row_times = sorted(rows)
    row_index_by_time = {time_ms: index for index, time_ms in enumerate(row_times)}
    chord_indices = {
        row_index_by_time[time_ms]
        for time_ms, lanes in rows.items()
        if len(lanes) >= 2
    }
    additions: List[NoteObject] = []
    candidates = sorted(
        [time_ms for time_ms, lanes in rows.items() if len(lanes) < max_size],
        key=lambda time_ms: (
            -context.get(time_ms, SpeedPoint(time_ms, 0, 0, 0, 0, 0, 0, False)).accent,
            time_ms,
        ),
    )
    for time_ms in candidates[: max(1, int(len(rows) * 0.035))]:
        lanes = rows[time_ms]
        index = row_index_by_time.get(time_ms)
        if index is None or _too_close_to_chosen_chord(index, chord_indices, _speed_chord_spacing_rows(target)):
            continue
        point = context.get(time_ms)
        if point is None or point.accent < 0.42:
            continue
        lane = sorted([lane for lane in LANES if lane not in lanes], key=lambda lane: _dindex(73, time_ms, lane))[0]
        lanes.add(lane)
        chord_indices.add(index)
        additions.append(NoteObject(time_ms, lane))
    return sorted(notes + additions, key=lambda n: (n.time_ms, n.lane, n.end_time_ms or -1))


def _speed_quality(notes: List[NoteObject], context: Dict[int, SpeedPoint]) -> float:
    if not notes:
        return -999.0
    rows: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, set()).add(note.lane)
    chord_ratio = sum(1 for lanes in rows.values() if len(lanes) > 1) / max(1, len(rows))
    avg_support = sum(context.get(t, SpeedPoint(t, 0, 0, 0, 0, 0, 0, False)).support for t in rows) / max(1, len(rows))
    chord_score = -abs(chord_ratio - 0.20)
    return avg_support + chord_score


def _build_context(
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
) -> Dict[int, SpeedPoint]:
    beat = 60000.0 / max(1.0, float(analysis.get("bpm", 120.0)))
    radius = max(28, min(58, int(beat * 0.11)))
    peaks = _indexed_peaks(analysis)
    peak_times = [item[0] for item in peaks]
    beat_times = [int(t) for t in analysis.get("beat_times_ms", [])]
    energy_curve = [float(v) for v in analysis.get("energy_curve", [])]
    duration = max(1, int(analysis.get("duration_ms", 1)))

    def energy_at(time_ms: int) -> float:
        if not energy_curve:
            return 0.45
        idx = max(0, min(len(energy_curve) - 1, int((time_ms / duration) * (len(energy_curve) - 1))))
        return max(0.0, min(1.0, energy_curve[idx]))

    context: Dict[int, SpeedPoint] = {}
    for time_ms in snap_points:
        silent = _in_silent(time_ms, analysis.get("silent_regions", []))
        attack = 0.0
        kick = 0.0
        left = bisect.bisect_left(peak_times, time_ms - radius)
        right = bisect.bisect_right(peak_times, time_ms + radius)
        for peak_time, strength, bass in peaks[left:right]:
            weight = max(0.0, 1.0 - abs(peak_time - time_ms) / max(1.0, radius))
            attack = max(attack, strength * weight)
            kick = max(kick, bass * weight)
        nearest_beat = _nearest_distance(beat_times, time_ms)
        beat_score = max(0.0, 1.0 - nearest_beat / max(1.0, radius)) if nearest_beat is not None and nearest_beat <= radius else 0.0
        energy = energy_at(time_ms)
        accent = max(
            attack * 0.70 + kick * 0.45,
            kick * 0.86,
            (0.35 + energy * 0.35) if time_ms in accent_snap_points else 0.0,
            beat_score * 0.28 + energy * 0.22,
        )
        support = max(
            accent,
            attack * 0.55 + energy * 0.22,
            beat_score * 0.28 + energy * 0.20,
        )
        if silent:
            attack = kick = energy = beat_score = accent = support = 0.0
        context[time_ms] = SpeedPoint(time_ms, attack, kick, energy, beat_score, min(1.0, accent), min(1.0, support), silent)
    return context


def _indexed_peaks(analysis: Dict[str, Any]) -> List[Tuple[int, float, float]]:
    raw = analysis.get("onset_peaks") or [
        {"time_ms": int(t), "strength": 0.65, "bass": 0.0}
        for t in analysis.get("onset_times_ms", [])
    ]
    peaks: List[Tuple[int, float, float]] = []
    for peak in raw:
        if isinstance(peak, dict):
            peaks.append((int(peak.get("time_ms", 0)), float(peak.get("strength", 0.0)), float(peak.get("bass", 0.0))))
        else:
            peaks.append((int(peak[0]), float(peak[1]) if len(peak) > 1 else 0.0, float(peak[2]) if len(peak) > 2 else 0.0))
    return sorted(peaks, key=lambda item: item[0])


def _nearest_distance(values: List[int], target: int) -> Optional[int]:
    if not values:
        return None
    index = bisect.bisect_left(values, target)
    candidates: List[int] = []
    if index < len(values):
        candidates.append(abs(values[index] - target))
    if index > 0:
        candidates.append(abs(values[index - 1] - target))
    return min(candidates) if candidates else None


def _in_silent(time_ms: int, regions: List[Tuple[int, int]]) -> bool:
    for start, end in regions:
        if start <= time_ms <= end:
            return True
    return False


def _dindex(modulo: int, *values: int) -> int:
    if modulo <= 0:
        return 0
    state = 0x9E3779B1
    for value in values:
        state ^= int(value) + 0x9E3779B9 + ((state << 6) & 0xFFFFFFFF) + (state >> 2)
        state &= 0xFFFFFFFF
    return state % modulo

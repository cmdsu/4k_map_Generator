import bisect
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.models import DifficultyConfig, NoteObject
from ..core.style_rules import clamp_max_chord_size
from ..core.calibration_utils import (
    _analysis_curve_average,
    _analysis_curve_value,
    _deterministic_index,
    _energy_score_at,
    _flatten_note_rows,
    _nearest_distance,
    _nearest_snap_point,
    _time_in_regions,
)


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
    active_style = config.key_style or "tech"
    if active_style == "jack":
        gap_threshold = int(beat_length * 0.45)
    elif active_style == "stream":
        gap_threshold = int(beat_length * 1.55)
    elif active_style == "speed":
        gap_threshold = int(beat_length * 1.75)
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
    adjusted = _rebalance_chord_weight_to_music(adjusted, config, analysis, snap_points, accent_snap_points, active_style, influence)
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
        return 0.052
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


def _rebalance_chord_weight_to_music(
    notes: List[NoteObject],
    config: DifficultyConfig,
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    active_style: Optional[str],
    influence: float,
) -> List[NoteObject]:
    max_chord_size = clamp_max_chord_size(config, active_style)
    if max_chord_size <= 1 or influence <= 0.05 or not notes:
        return notes

    rows: Dict[int, List[NoteObject]] = {}
    row_lanes: Dict[int, Set[int]] = {}
    for note in notes:
        rows.setdefault(note.time_ms, []).append(note)
        row_lanes.setdefault(note.time_ms, set()).add(note.lane)

    row_times = sorted(row_lanes)
    if len(row_times) < 2:
        return notes

    target = config.target_star or 4.0
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    strength_by_time = {
        time_ms: _chord_music_strength_at(analysis, snap_points, accent_snap_points, time_ms)
        for time_ms in row_times
    }
    desired_by_time = {
        time_ms: _music_aligned_chord_size(
            _music_entry(analysis, snap_points, accent_snap_points, time_ms),
            max_chord_size,
            active_style,
            target,
        )
        for time_ms in row_times
    }

    receiver_threshold = _chord_rebalance_receiver_threshold(active_style, target)
    min_delta = 0.11 - min(1.0, influence) * 0.05
    max_moves = max(1, int(round(len(row_times) * _chord_rebalance_ratio(active_style) * influence)))

    donor_capacity: Dict[int, int] = {}
    for time_ms in row_times:
        row_size = len(row_lanes[time_ms])
        if row_size <= 1:
            continue
        strength = strength_by_time[time_ms]
        desired_size = desired_by_time[time_ms]
        if strength >= receiver_threshold:
            continue
        keep_size = max(1, min(row_size, desired_size))
        if strength < 0.18:
            keep_size = 1
        capacity = row_size - keep_size
        if capacity > 0:
            donor_capacity[time_ms] = capacity

    receivers = sorted(
        (
            time_ms
            for time_ms in row_times
            if strength_by_time[time_ms] >= receiver_threshold
            and len(row_lanes[time_ms]) < desired_by_time[time_ms]
        ),
        key=lambda time_ms: (
            -strength_by_time[time_ms],
            len(row_lanes[time_ms]),
            time_ms,
        ),
    )
    donors = sorted(
        donor_capacity,
        key=lambda time_ms: (
            strength_by_time[time_ms],
            -donor_capacity[time_ms],
            time_ms,
        ),
    )
    removed: Set[int] = set()
    additions: List[NoteObject] = []
    donor_cursor = 0
    moves = 0

    if donors and receivers:
        for receiver_time in receivers:
            if moves >= max_moves:
                break
            receiver_strength = strength_by_time[receiver_time]
            while len(row_lanes[receiver_time]) < desired_by_time[receiver_time] and moves < max_moves:
                donor_time = _next_chord_weight_donor(
                    donors,
                    donor_cursor,
                    donor_capacity,
                    strength_by_time,
                    receiver_strength,
                    min_delta,
                    receiver_time,
                )
                if donor_time is None:
                    break
                donor_cursor = max(donor_cursor, donors.index(donor_time))

                note_to_move = _pick_chord_weight_donor_note(donor_time, rows[donor_time], row_lanes, row_times, removed, active_style)
                if note_to_move is None:
                    donor_capacity[donor_time] = 0
                    donor_cursor += 1
                    continue

                receiver_entry = _music_entry(analysis, snap_points, accent_snap_points, receiver_time)
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
                donor_capacity[donor_time] = max(0, donor_capacity[donor_time] - 1)
                row_lanes[receiver_time].add(lane)
                additions.append(NoteObject(time_ms=receiver_time, lane=lane))
                moves += 1

    cleanup_limit = max(0, int(round(len(row_times) * _chord_cleanup_ratio(active_style) * influence)))
    cleanup_threshold = _chord_cleanup_threshold(active_style)
    cleanup_moves = 0
    cleanup_candidates = sorted(
        (
            time_ms
            for time_ms in row_times
            if len(row_lanes.get(time_ms, set())) > _weak_chord_keep_size(
                strength_by_time.get(time_ms, 0.0),
                active_style,
                target,
            )
        ),
        key=lambda time_ms: (
            strength_by_time.get(time_ms, 0.0),
            -len(row_lanes.get(time_ms, set())),
            time_ms,
        ),
    )
    for donor_time in cleanup_candidates:
        if cleanup_moves >= cleanup_limit:
            break
        if strength_by_time.get(donor_time, 0.0) >= cleanup_threshold:
            continue
        keep_size = _weak_chord_keep_size(strength_by_time.get(donor_time, 0.0), active_style, target)
        while len(row_lanes.get(donor_time, set())) > keep_size and cleanup_moves < cleanup_limit:
            note_to_remove = _pick_chord_weight_donor_note(donor_time, rows[donor_time], row_lanes, row_times, removed, active_style)
            if note_to_remove is None:
                break
            removed.add(id(note_to_remove))
            row_lanes[donor_time].discard(note_to_remove.lane)
            cleanup_moves += 1

    if moves <= 0 and cleanup_moves <= 0:
        return notes
    kept = [note for note in notes if id(note) not in removed]
    return sorted(kept + additions, key=lambda note: (note.time_ms, note.lane, note.end_time_ms or -1))


def _chord_music_strength(entry: Dict[str, float]) -> float:
    hit_strength = max(
        entry.get("kick", 0.0) * 1.12,
        entry.get("onset", 0.0) * 0.98,
        entry.get("stack", 0.0) * 0.90,
        entry.get("protected", 0.0) * 0.72,
    )
    bed_strength = max(
        entry.get("score", 0.0) * 0.62,
        entry.get("accent", 0.0) * 0.54,
        entry.get("energy", 0.0) * 0.45,
    )
    if max(entry.get("kick", 0.0), entry.get("onset", 0.0), entry.get("stack", 0.0)) < 0.08:
        bed_strength *= 0.42
    return max(
        hit_strength,
        bed_strength,
    )


def _chord_music_strength_at(
    analysis: Dict[str, Any],
    snap_points: List[int],
    accent_snap_points: Set[int],
    time_ms: int,
) -> float:
    entry = _music_entry(analysis, snap_points, accent_snap_points, time_ms)
    return max(
        _chord_music_strength(entry),
        _nearby_onset_hit_strength(analysis, time_ms),
    )


def _nearby_onset_hit_strength(analysis: Dict[str, Any], time_ms: int) -> float:
    peaks = analysis.get("onset_peaks") or []
    if not peaks:
        return 0.0

    beat_length = 60000.0 / max(1.0, analysis.get("bpm", 120.0))
    radius = max(30, min(46, int(round(beat_length * 0.105))))
    cache_key = (int(time_ms), int(radius))
    cache = analysis.setdefault("_nearby_onset_hit_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    indexed = analysis.get("_indexed_onset_peaks")
    if indexed is None:
        indexed = []
        for peak in peaks:
            if isinstance(peak, dict):
                peak_time = int(peak.get("time_ms", 0))
                strength = float(peak.get("strength", 0.0))
                bass = float(peak.get("bass", 0.0))
            else:
                peak_time = int(peak[0])
                strength = float(peak[1]) if len(peak) > 1 else 0.0
                bass = float(peak[2]) if len(peak) > 2 else 0.0
            indexed.append((peak_time, strength, bass))
        indexed.sort(key=lambda item: item[0])
        analysis["_indexed_onset_peaks"] = indexed
        analysis["_indexed_onset_times"] = [item[0] for item in indexed]

    peak_times = analysis.get("_indexed_onset_times", [])
    best = 0.0
    left = bisect.bisect_left(peak_times, time_ms - radius)
    right = bisect.bisect_right(peak_times, time_ms + radius)
    for peak_time, strength, bass in indexed[left:right]:
        distance = abs(peak_time - time_ms)
        distance_weight = max(0.0, 1.0 - (distance / max(1.0, radius)) * 0.55)
        hit = max(
            bass * 1.12,
            strength * 0.96,
            bass * 0.72 + strength * 0.42,
        ) * distance_weight
        best = max(best, min(1.0, hit))
    cache[cache_key] = best
    return best


def _music_aligned_chord_size(
    entry: Dict[str, float],
    max_chord_size: int,
    active_style: Optional[str],
    target: float,
) -> int:
    strength = _chord_music_strength(entry)
    desired = 1
    if strength >= 0.44 or entry.get("kick", 0.0) >= 0.38 or entry.get("onset", 0.0) >= 0.36:
        desired = 2
    if max_chord_size >= 3 and (strength >= 0.58 or entry.get("kick", 0.0) >= 0.55 or entry.get("stack", 0.0) >= 0.64):
        desired = 3
    if (
        max_chord_size >= 4
        and active_style in ["jack", "tech", None]
        and target >= 5.0
        and (strength >= 0.70 or entry.get("kick", 0.0) >= 0.72 or entry.get("stack", 0.0) >= 0.82)
    ):
        desired = 4
    if active_style == "stream":
        desired = min(desired, 3)
    if active_style == "speed" and strength < 0.72:
        desired = min(desired, 2)
    return max(1, min(max_chord_size, desired))


def _chord_rebalance_receiver_threshold(active_style: Optional[str], target: float) -> float:
    if active_style == "jack":
        return 0.42 if target < 6.0 else 0.39
    if active_style == "stream":
        return 0.46
    if active_style == "tech":
        return 0.45
    if active_style == "speed":
        return 0.54
    return 0.44


def _chord_rebalance_ratio(active_style: Optional[str]) -> float:
    if active_style == "jack":
        return 0.20
    if active_style == "stream":
        return 0.10
    if active_style == "tech":
        return 0.12
    if active_style == "speed":
        return 0.090
    return 0.10


def _chord_cleanup_ratio(active_style: Optional[str]) -> float:
    if active_style == "jack":
        return 0.90
    if active_style == "stream":
        return 0.45
    if active_style == "tech":
        return 0.25
    if active_style == "speed":
        return 0.08
    return 0.25


def _chord_cleanup_threshold(active_style: Optional[str]) -> float:
    if active_style == "jack":
        return 0.34
    if active_style == "stream":
        return 0.28
    if active_style == "tech":
        return 0.22
    if active_style == "speed":
        return 0.18
    return 0.22


def _weak_chord_keep_size(strength: float, active_style: Optional[str], target: float) -> int:
    if active_style == "jack":
        # Jack still needs stacked texture, but large stacks should belong to clear accents.
        return 2 if target >= 4.75 else 1
    if active_style == "stream":
        # Medium-strength jumpstream cells should survive cleanup; a later cut
        # integrity pass removes/rewrites only the cells that would cause fast
        # same-lane overlap.
        if target >= 5.25 and strength >= 0.24:
            return 3
        if target >= 5.25 and strength >= 0.14:
            return 2
        return 2 if target >= 6.25 and strength >= 0.10 else 1
    if active_style == "tech":
        return 2 if target >= 6.0 and strength >= 0.18 else 1
    if active_style == "speed":
        return 1
    return 2 if target >= 6.0 and strength >= 0.18 else 1


def _next_chord_weight_donor(
    donors: List[int],
    start_index: int,
    donor_capacity: Dict[int, int],
    strength_by_time: Dict[int, float],
    receiver_strength: float,
    min_delta: float,
    receiver_time: int,
) -> Optional[int]:
    for index in range(max(0, start_index), len(donors)):
        time_ms = donors[index]
        if time_ms == receiver_time or donor_capacity.get(time_ms, 0) <= 0:
            continue
        if receiver_strength - strength_by_time.get(time_ms, 0.0) < min_delta:
            continue
        return time_ms
    return None


def _pick_chord_weight_donor_note(
    time_ms: int,
    row_notes: List[NoteObject],
    row_lanes: Dict[int, Set[int]],
    row_times: List[int],
    removed: Set[int],
    active_style: Optional[str],
) -> Optional[NoteObject]:
    if len(row_lanes.get(time_ms, set())) <= 1:
        return None

    candidates = [note for note in row_notes if id(note) not in removed and not note.is_ln]
    if not candidates:
        return None

    idx = bisect.bisect_left(row_times, time_ms)
    prev_lanes = row_lanes.get(row_times[idx - 1], set()) if idx > 0 else set()
    next_lanes = row_lanes.get(row_times[idx + 1], set()) if idx + 1 < len(row_times) else set()
    structural_lanes = prev_lanes | next_lanes

    def score(note: NoteObject) -> Tuple[int, float, int, int]:
        if active_style == "jack":
            structural_penalty = 1 if note.lane in structural_lanes else 0
        elif active_style in ["stream", "tech"]:
            structural_penalty = 1 if note.lane not in structural_lanes else 0
        else:
            structural_penalty = 0
        return (
            structural_penalty,
            -abs(note.lane - 1.5),
            _deterministic_index(193, time_ms, note.lane),
            note.lane,
        )

    return sorted(candidates, key=score)[0]


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

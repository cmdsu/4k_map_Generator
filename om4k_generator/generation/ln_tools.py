import bisect
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.difficulty_estimator import DifficultyEstimator
from ..core.models import DifficultyConfig, NoteObject
from ..core.style_rules import clamp_max_chord_size
from ..core.validator import Validator
from ..core.calibration_utils import (
    _analysis_curve_average,
    _deterministic_index,
    _nearest_snap_point,
)
from .music_alignment import _music_context, _music_entry, _music_influence
from ..audio.snap_utils import build_accent_snap_points


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
    if config.chart_type != "ln" or not notes or not any(note.is_ln for note in notes):
        return notes

    context = _music_context(analysis, snap_points, accent_snap_points)
    target = config.target_star or 4.0
    max_chord_size = clamp_max_chord_size(config)
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    tail_gap = max(34, int(min(58, beat_length / 12.0)))
    min_ln_ms = max(30, int(config.min_ln_ms))
    if config.key_style in ["stream", "speed"]:
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
    if config.chart_type != "ln" or current_sr <= upper_bound or not any(note.is_ln for note in notes):
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
    context = _music_context(analysis, snap_points, accent_snap_points)
    beat_length = 60000.0 / max(1.0, analysis["bpm"])
    tail_gap = max(34, int(min(58, beat_length / 12.0)))
    min_ln_ms = max(30, int(config.min_ln_ms))
    if config.key_style in ["stream", "speed"]:
        min_ln_ms = max(30, min(min_ln_ms, int(max(45, beat_length / 8.0))))
    max_chord_size = clamp_max_chord_size(config)

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
        candidate = _release_ln_blocks_for_music_hits(
            candidate,
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
        candidate = _restore_music_hits_after_ln_refine(
            candidate,
            config,
            analysis,
            snap_points,
            accent_snap_points,
            context,
            target,
            max_chord_size,
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
            max_chord_size,
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
    if config.key_style == "speed":
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
        if config.key_style in ["stream", "speed"]:
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

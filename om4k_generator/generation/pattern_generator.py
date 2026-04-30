import random
from typing import List, Optional, Set, Tuple

from ..core.models import DifficultyConfig, NoteObject
from ..core.style_rules import clamp_max_chord_size


class PatternGenerator:
    def __init__(self, config: DifficultyConfig):
        self.config = config
        self.lanes = [0, 1, 2, 3]
        self.stream_order = [0, 2, 1, 3]
        self.stream_index = 0
        self.tech_patterns = [[0, 2, 1, 3], [0, 1, 3, 2], [1, 3, 0, 2], [2, 0, 3, 1]]
        self.tech_index = 0
        self.jack_stack_lanes: List[int] = []
        self.jack_stack_remaining = 0
        self.jack_quad_streak = 0

    def _energy_peak_threshold(self, energy_curve: List[float]) -> float:
        values = sorted(float(v) for v in energy_curve if v is not None)
        if not values:
            return 0.0

        percentile_index = min(len(values) - 1, int(len(values) * 0.85))
        average = sum(values) / len(values)
        return max(values[percentile_index], average * 1.25)

    def _is_energy_peak(self, t: int, energy_curve: List[float], max_time: int, threshold: float) -> bool:
        if not energy_curve or max_time <= 0 or threshold <= 0:
            return False

        idx = int((t / max_time) * (len(energy_curve) - 1))
        idx = max(0, min(len(energy_curve) - 1, idx))
        return float(energy_curve[idx]) >= threshold

    def _select_style(self) -> Optional[str]:
        return self.config.key_style or "tech"

    def _timing_limits(self, style: Optional[str], density_multiplier: float, median_interval: float) -> tuple[float, float]:
        if style == "jack":
            interval = max(55, min(120, median_interval * 0.90))
            return interval, interval
        if style == "stream":
            interval = max(45, min(90, median_interval * 0.80))
            return 0, interval
        if style == "speed":
            return 45, 35
        return 50, 35

    def _get_chord_size(self, style: Optional[str], is_peak: bool, is_accent: bool) -> int:
        if style == "speed" and not is_peak:
            return 1
        if style == "jack" and not is_accent:
            if self.jack_stack_remaining > 0 and len(self.jack_stack_lanes) > 1 and not self._jack_should_collapse_to_single():
                if len(self.jack_stack_lanes) == 4 and self.jack_quad_streak >= self._max_jack_quad_streak():
                    return 3
                return len(self.jack_stack_lanes)
            if self.config.target_star is not None and self.config.target_star < 4.5:
                return 1

        max_chord_size = clamp_max_chord_size(self.config, style)
        if max_chord_size <= 1:
            return 1

        chord_enabled = self.config.chord_enabled or style in ["jack", "stream"]
        if not chord_enabled:
            return 1

        base_probability = self.config.chord_probability
        if style == "jack":
            if self.config.target_star is None:
                floor = 0.94
            elif self.config.target_star >= 5.5:
                floor = 0.88
            elif self.config.target_star >= 4.5:
                floor = 0.62
            elif self.config.target_star >= 3.5:
                floor = 0.28
            else:
                floor = 0.08
            base_probability = max(base_probability, floor)
        elif style == "stream":
            base_probability = max(base_probability, 0.45)
        elif style == "tech":
            base_probability = max(base_probability, 0.25)

        if random.random() > min(1.0, base_probability + (0.25 if is_peak else 0.0)):
            return 1

        chord_size = self._weighted_chord_size(style, max_chord_size)
        if style == "jack" and chord_size == 4 and self.jack_quad_streak >= self._max_jack_quad_streak():
            return 3
        return chord_size

    def _weighted_chord_size(self, style: Optional[str], max_chord_size: int) -> int:
        if max_chord_size <= 2:
            return 2

        if style == "jack":
            if max_chord_size >= 4:
                if self.config.target_star is not None and self.config.target_star < 6.0:
                    return random.choices([2, 3, 4], weights=[0.46, 0.44, 0.10], k=1)[0]
                return random.choices([2, 3, 4], weights=[0.38, 0.44, 0.18], k=1)[0]
            return random.choices([2, 3], weights=[0.52, 0.48], k=1)[0]

        if style == "stream":
            return random.choices([2, 3], weights=[0.82, 0.18], k=1)[0]

        return random.randint(2, max_chord_size)

    def _max_jack_quad_streak(self) -> int:
        if self.config.target_star is not None and self.config.target_star < 6.5:
            return 1
        return 2

    def _jack_should_collapse_to_single(self) -> bool:
        return self.config.target_star is not None and self.config.target_star < 4.5

    def _jack_min_keep_count(self, chord_size: int, keep_limit: int) -> int:
        if keep_limit <= 0:
            return 0
        if self.config.target_star is not None and self.config.target_star >= 6.5:
            return min(2, keep_limit)
        if self.config.target_star is not None and self.config.target_star >= 4.5:
            return 1
        return 1

    def generate(
        self,
        snap_points: List[int],
        energy_curve: List[float],
        silent_regions: List[Tuple[int, int]],
        density_multiplier: float = 1.0,
        accent_times_ms: Optional[Set[int]] = None,
        bpm: float = 120.0,
    ) -> List[NoteObject]:
        notes = []
        last_lanes: List[int] = []
        recent_times = []
        last_lane_times = {0: -9999, 1: -9999, 2: -9999, 3: -9999}
        jack_streaks = {0: 0, 1: 0, 2: 0, 3: 0}
        last_note_time: Optional[int] = None
        max_time = max(snap_points) if snap_points else 0
        energy_peak_threshold = self._energy_peak_threshold(energy_curve)
        median_interval = self._median_interval(snap_points)
        beat_length = 60000.0 / max(1.0, bpm)

        max_nps = max(3.0, 20.0 * density_multiplier)
        silent_regions = sorted(silent_regions)
        silent_index = 0
        jack_density_meter = max(0.0, 1.0 - min(1.0, density_multiplier))

        for t in snap_points:
            while silent_index < len(silent_regions) and silent_regions[silent_index][1] < t:
                silent_index += 1
            if silent_index < len(silent_regions) and silent_regions[silent_index][0] <= t <= silent_regions[silent_index][1]:
                continue

            style = self._select_style()
            is_accent = accent_times_ms is None or t in accent_times_ms
            min_lane_interval, global_min_interval = self._timing_limits(style, density_multiplier, median_interval)
            is_reasonable_rest = self._is_reasonable_rest(t, energy_curve, max_time, silent_regions)
            must_place = last_note_time is None or (
                last_note_time is not None
                and (t - last_note_time) >= self._continuity_limit_ms(style, beat_length, density_multiplier)
                and not is_reasonable_rest
            )

            while recent_times and t - recent_times[0] > 1000:
                recent_times.pop(0)
            if not must_place and len(recent_times) >= max_nps:
                continue
            if not must_place and recent_times and (t - recent_times[-1]) < global_min_interval:
                continue
            if density_multiplier < 1.0 and not must_place:
                if style == "jack":
                    if not is_accent:
                        jack_density_meter += density_multiplier
                        if jack_density_meter < 1.0:
                            continue
                        jack_density_meter -= 1.0
                elif random.random() > density_multiplier:
                    continue

            is_peak = self._is_energy_peak(t, energy_curve, max_time, energy_peak_threshold)
            chord_size = self._get_chord_size(style, is_peak, is_accent)
            available_lanes = [lane for lane in self.lanes if t - last_lane_times[lane] >= min_lane_interval]
            if not available_lanes:
                continue

            chord_size = min(chord_size, len(available_lanes), clamp_max_chord_size(self.config, style))
            chosen = self._choose_lanes(style, chord_size, available_lanes, last_lanes, jack_streaks, last_lane_times)
            if not chosen:
                continue

            for lane in chosen:
                notes.append(NoteObject(time_ms=t, lane=lane))
                last_lane_times[lane] = t

            for lane in self.lanes:
                jack_streaks[lane] = jack_streaks[lane] + 1 if lane in chosen else 0
            if style == "jack":
                self.jack_quad_streak = self.jack_quad_streak + 1 if len(chosen) == 4 else 0

            recent_times.append(t)
            last_lanes = chosen
            last_note_time = t

        if self.config.chart_type == "ln":
            notes = self._convert_to_lns(notes)

        return notes

    def _choose_lanes(
        self,
        style: Optional[str],
        chord_size: int,
        available_lanes: List[int],
        last_lanes: List[int],
        jack_streaks: dict[int, int],
        last_lane_times: dict[int, int],
    ) -> List[int]:
        if style == "jack":
            return self._choose_jack_lanes(chord_size, available_lanes, last_lanes, jack_streaks)
        if style == "stream":
            return self._choose_stream_lanes(chord_size, available_lanes, last_lanes)
        if style == "tech":
            return self._choose_tech_lanes(chord_size, available_lanes)
        return self._choose_speed_lanes(chord_size, available_lanes, last_lane_times)

    def _choose_jack_lanes(
        self,
        chord_size: int,
        available_lanes: List[int],
        last_lanes: List[int],
        jack_streaks: dict[int, int],
    ) -> List[int]:
        if self.jack_stack_remaining <= 0 or len(self.jack_stack_lanes) != chord_size:
            self.jack_stack_lanes = self._start_jack_stack(chord_size, available_lanes, last_lanes)
            self.jack_stack_remaining = random.randint(4, 8)
        elif chord_size > 1:
            self.jack_stack_lanes = self._advance_jack_stack(chord_size, available_lanes, self.jack_stack_lanes)

        chosen = [lane for lane in self.jack_stack_lanes if lane in available_lanes]
        if len(chosen) < chord_size:
            remaining = [lane for lane in available_lanes if lane not in chosen]
            chosen.extend(remaining[: chord_size - len(chosen)])

        self.jack_stack_remaining -= 1
        return chosen[:chord_size]

    def _start_jack_stack(self, chord_size: int, available_lanes: List[int], last_lanes: List[int]) -> List[int]:
        if chord_size <= 0 or not available_lanes:
            return []

        previous = [lane for lane in last_lanes if lane in available_lanes]
        if chord_size == 1:
            if previous:
                return [random.choice(previous)]
            return [random.choice(available_lanes)]

        chosen: List[int] = []
        if previous:
            keep_count = min(len(previous), chord_size - 1)
            keep_min = self._jack_min_keep_count(chord_size, keep_count)
            keep_count = random.randint(max(1, keep_min), max(1, keep_count))
            chosen.extend(random.sample(previous, keep_count))

        remaining = [lane for lane in available_lanes if lane not in chosen]
        random.shuffle(remaining)
        while remaining and len(chosen) < chord_size:
            chosen.append(remaining.pop())

        if len(chosen) < chord_size:
            fallback = [lane for lane in available_lanes if lane not in chosen]
            chosen.extend(fallback[: chord_size - len(chosen)])

        return sorted(chosen[:chord_size])

    def _advance_jack_stack(
        self,
        chord_size: int,
        available_lanes: List[int],
        previous_lanes: List[int],
    ) -> List[int]:
        previous = [lane for lane in previous_lanes if lane in available_lanes]
        if chord_size <= 1 or not previous:
            return self._start_jack_stack(chord_size, available_lanes, previous_lanes)

        keep_limit = min(len(previous), chord_size - 1)
        keep_min = self._jack_min_keep_count(chord_size, keep_limit)
        keep_count = random.randint(max(1, keep_min), max(1, keep_limit))
        chosen = random.sample(previous, keep_count)

        remaining = [lane for lane in available_lanes if lane not in chosen]
        random.shuffle(remaining)
        while remaining and len(chosen) < chord_size:
            chosen.append(remaining.pop())

        if len(chosen) < chord_size:
            fallback = [lane for lane in available_lanes if lane not in chosen]
            chosen.extend(fallback[: chord_size - len(chosen)])

        if set(chosen) == set(previous_lanes) and len(available_lanes) > chord_size:
            outside = [lane for lane in available_lanes if lane not in chosen]
            replaceable = [lane for lane in chosen if lane not in previous[:1]] or chosen[1:]
            if outside and replaceable:
                chosen.remove(random.choice(replaceable))
                chosen.append(random.choice(outside))

        return sorted(chosen[:chord_size])

    def _choose_stream_lanes(self, chord_size: int, available_lanes: List[int], last_lanes: List[int]) -> List[int]:
        chosen = []
        non_repeating = [lane for lane in available_lanes if lane not in last_lanes]
        target_size = min(chord_size, len(non_repeating)) if non_repeating else min(chord_size, len(available_lanes))

        for _ in range(8):
            lane = self.stream_order[self.stream_index % len(self.stream_order)]
            self.stream_index += 1
            if lane in available_lanes and lane not in last_lanes and lane not in chosen:
                chosen.append(lane)
            if len(chosen) >= target_size:
                return chosen

        fallback = [lane for lane in available_lanes if lane not in chosen and lane not in last_lanes]
        if len(chosen) + len(fallback) < target_size:
            fallback = [lane for lane in available_lanes if lane not in chosen]
        chosen.extend(fallback[: max(0, target_size - len(chosen))])
        return chosen

    def _choose_tech_lanes(self, chord_size: int, available_lanes: List[int]) -> List[int]:
        pattern = self.tech_patterns[self.tech_index % len(self.tech_patterns)]
        self.tech_index += 1
        chosen = [lane for lane in pattern if lane in available_lanes][:chord_size]
        if len(chosen) < chord_size:
            remaining = [lane for lane in available_lanes if lane not in chosen]
            chosen.extend(remaining[: chord_size - len(chosen)])
        return chosen

    def _choose_speed_lanes(
        self,
        chord_size: int,
        available_lanes: List[int],
        last_lane_times: dict[int, int],
    ) -> List[int]:
        ordered = sorted(available_lanes, key=lambda lane: last_lane_times[lane])
        return ordered[:chord_size]

    @staticmethod
    def _median_interval(snap_points: List[int]) -> float:
        intervals = [b - a for a, b in zip(snap_points, snap_points[1:]) if b > a]
        if not intervals:
            return 80.0
        intervals.sort()
        return float(intervals[len(intervals) // 2])

    @staticmethod
    def _continuity_limit_ms(style: Optional[str], beat_length: float, density_multiplier: float) -> float:
        if style == "stream":
            return beat_length * (1.25 if density_multiplier >= 0.8 else 1.5)
        if style == "jack":
            return beat_length * (1.00 if density_multiplier >= 0.8 else 1.10)
        if style == "speed":
            return beat_length * (1.1 if density_multiplier >= 0.8 else 1.35)
        return beat_length * (1.5 if density_multiplier >= 0.8 else 1.75)

    @staticmethod
    def _normalized_energy_at(t: int, energy_curve: List[float], max_time: int) -> float:
        if not energy_curve or max_time <= 0:
            return 1.0

        values = sorted(float(v) for v in energy_curve if v is not None)
        if not values:
            return 1.0

        low = values[min(len(values) - 1, int(len(values) * 0.25))]
        high = values[min(len(values) - 1, int(len(values) * 0.90))]
        if high <= low:
            return 0.5

        idx = int((t / max_time) * (len(energy_curve) - 1))
        idx = max(0, min(len(energy_curve) - 1, idx))
        return max(0.0, min(1.0, (float(energy_curve[idx]) - low) / (high - low)))

    def _is_reasonable_rest(
        self,
        t: int,
        energy_curve: List[float],
        max_time: int,
        silent_regions: List[Tuple[int, int]],
    ) -> bool:
        for start, end in silent_regions:
            if start <= t <= end:
                return True
        return self._normalized_energy_at(t, energy_curve, max_time) < 0.35

    def _convert_to_lns(self, notes: List[NoteObject]) -> List[NoteObject]:
        result = []
        skip_to = {0: 0, 1: 0, 2: 0, 3: 0}
        next_same_lane_time = self._next_same_lane_times(notes) if self.config.key_style == "jack" else {}

        for index, note in enumerate(notes):
            if note.time_ms < skip_to[note.lane]:
                continue

            if random.random() < self.config.ln_ratio:
                max_ln_ms = self.config.max_ln_ms
                if self.config.key_style == "jack":
                    next_same = next_same_lane_time.get(index)
                    if next_same is not None:
                        max_ln_ms = min(max_ln_ms, max(0, next_same - note.time_ms - 35))

                if max_ln_ms >= self.config.min_ln_ms:
                    end_time = note.time_ms + random.randint(self.config.min_ln_ms, max_ln_ms)
                    note.end_time_ms = end_time
                    skip_to[note.lane] = end_time

            result.append(note)
        return result

    @staticmethod
    def _next_same_lane_times(notes: List[NoteObject]) -> dict[int, int]:
        next_times: dict[int, int] = {}
        last_seen: dict[int, int] = {}
        for index in range(len(notes) - 1, -1, -1):
            note = notes[index]
            if note.lane in last_seen:
                next_times[index] = last_seen[note.lane]
            last_seen[note.lane] = note.time_ms
        return next_times

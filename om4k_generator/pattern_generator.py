import random
from typing import List, Optional, Set, Tuple

from .models import DifficultyConfig, NoteObject
from .style_rules import clamp_max_chord_size, normalize_hybrid_weights


class PatternGenerator:
    def __init__(self, config: DifficultyConfig):
        self.config = config
        self.lanes = [0, 1, 2, 3]
        self.hybrid_weights = normalize_hybrid_weights(config.hybrid_weights)
        self.stream_order = [0, 2, 1, 3]
        self.stream_index = 0
        self.tech_patterns = [[0, 2, 1, 3], [0, 1, 3, 2], [1, 3, 0, 2], [2, 0, 3, 1]]
        self.tech_index = 0
        self.vibro_lanes = list(config.vibro_options.get("lanes", [1, 2])) if config.vibro_options else [1, 2]
        self.vibro_lanes = [lane for lane in self.vibro_lanes if lane in self.lanes] or [1, 2]
        self.vibro_index = 0

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
        if self.config.chart_type == "vibro":
            return "vibro"
        if self.config.chart_type == "hybrid":
            roll = random.random()
            cumulative = 0.0
            for style, weight in self.hybrid_weights.items():
                cumulative += weight
                if roll <= cumulative:
                    return style
            return "tech"
        return self.config.key_style or "tech"

    def _timing_limits(self, style: Optional[str], density_multiplier: float) -> tuple[float, float]:
        if style == "vibro":
            return 35, 20
        if style == "jack":
            interval = max(60, 160 - (density_multiplier * 50))
            return interval, interval
        if style == "stream":
            return 55, 35
        if style == "speed":
            return 45, 35
        return 50, 35

    def _get_chord_size(self, style: Optional[str], is_peak: bool, is_accent: bool) -> int:
        if style == "speed" and not is_peak:
            return 1
        if style == "jack" and not is_accent:
            return 1

        max_chord_size = clamp_max_chord_size(self.config, style)
        if max_chord_size <= 1:
            return 1

        chord_enabled = self.config.chord_enabled or style in ["jack", "stream"]
        if not chord_enabled:
            return 1

        base_probability = self.config.chord_probability
        if style == "jack":
            base_probability = max(base_probability, 0.55)
        elif style == "stream":
            base_probability = max(base_probability, 0.45)
        elif style == "vibro":
            base_probability = max(base_probability, 0.60)
        elif style == "tech":
            base_probability = max(base_probability, 0.25)

        if random.random() > min(1.0, base_probability + (0.25 if is_peak else 0.0)):
            return 1

        return random.randint(2, max_chord_size)

    def generate(
        self,
        snap_points: List[int],
        energy_curve: List[float],
        silent_regions: List[Tuple[int, int]],
        density_multiplier: float = 1.0,
        accent_times_ms: Optional[Set[int]] = None,
    ) -> List[NoteObject]:
        notes = []
        last_lanes: List[int] = []
        recent_times = []
        last_lane_times = {0: -9999, 1: -9999, 2: -9999, 3: -9999}
        jack_streaks = {0: 0, 1: 0, 2: 0, 3: 0}
        max_time = max(snap_points) if snap_points else 0
        energy_peak_threshold = self._energy_peak_threshold(energy_curve)

        max_nps = max(3.0, 20.0 * density_multiplier)
        silent_regions = sorted(silent_regions)
        silent_index = 0

        for t in snap_points:
            while silent_index < len(silent_regions) and silent_regions[silent_index][1] < t:
                silent_index += 1
            if silent_index < len(silent_regions) and silent_regions[silent_index][0] <= t <= silent_regions[silent_index][1]:
                continue

            style = self._select_style()
            min_lane_interval, global_min_interval = self._timing_limits(style, density_multiplier)

            while recent_times and t - recent_times[0] > 1000:
                recent_times.pop(0)
            if len(recent_times) >= max_nps:
                continue
            if recent_times and (t - recent_times[-1]) < global_min_interval:
                continue
            if density_multiplier < 1.0 and random.random() > density_multiplier:
                continue

            is_peak = self._is_energy_peak(t, energy_curve, max_time, energy_peak_threshold)
            is_accent = accent_times_ms is None or t in accent_times_ms
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

            recent_times.append(t)
            last_lanes = chosen

        if self.config.chart_type in ["ln", "hybrid"]:
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
        if style == "vibro":
            return self._choose_vibro_lanes(chord_size, available_lanes, last_lanes)
        if style == "jack":
            return self._choose_jack_lanes(chord_size, available_lanes, last_lanes, jack_streaks)
        if style == "stream":
            return self._choose_stream_lanes(chord_size, available_lanes, last_lanes)
        if style == "tech":
            return self._choose_tech_lanes(chord_size, available_lanes)
        return self._choose_speed_lanes(chord_size, available_lanes, last_lane_times)

    def _choose_vibro_lanes(self, chord_size: int, available_lanes: List[int], last_lanes: List[int]) -> List[int]:
        chosen = []
        for _ in range(len(self.vibro_lanes)):
            lane = self.vibro_lanes[self.vibro_index % len(self.vibro_lanes)]
            self.vibro_index += 1
            if lane in available_lanes:
                chosen.append(lane)
                break

        if not chosen:
            chosen.append(random.choice(available_lanes))

        anchors = [lane for lane in last_lanes if lane in available_lanes and lane not in chosen]
        while anchors and len(chosen) < chord_size:
            lane = anchors.pop(0)
            chosen.append(lane)

        remaining = [lane for lane in available_lanes if lane not in chosen]
        if remaining and len(chosen) < chord_size:
            chosen.extend(random.sample(remaining, min(chord_size - len(chosen), len(remaining))))

        return chosen

    def _choose_jack_lanes(
        self,
        chord_size: int,
        available_lanes: List[int],
        last_lanes: List[int],
        jack_streaks: dict[int, int],
    ) -> List[int]:
        valid_anchors = [
            lane for lane in last_lanes
            if lane in available_lanes and jack_streaks[lane] < self.config.max_jack_length
        ]
        anchor_count = min(chord_size, len(valid_anchors), 1 if chord_size < 3 else 2)
        chosen = random.sample(valid_anchors, anchor_count) if anchor_count > 0 else []

        remaining = [lane for lane in available_lanes if lane not in chosen]
        if remaining and len(chosen) < chord_size:
            chosen.extend(random.sample(remaining, min(chord_size - len(chosen), len(remaining))))
        return chosen

    def _choose_stream_lanes(self, chord_size: int, available_lanes: List[int], last_lanes: List[int]) -> List[int]:
        chosen = []
        for _ in range(4):
            lane = self.stream_order[self.stream_index % len(self.stream_order)]
            self.stream_index += 1
            if lane in available_lanes and lane not in last_lanes and lane not in chosen:
                chosen.append(lane)
            if len(chosen) >= chord_size:
                return chosen

        fallback = [lane for lane in available_lanes if lane not in chosen and lane not in last_lanes]
        if len(chosen) + len(fallback) < chord_size:
            fallback = [lane for lane in available_lanes if lane not in chosen]
        chosen.extend(fallback[: max(0, chord_size - len(chosen))])
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

    def _convert_to_lns(self, notes: List[NoteObject]) -> List[NoteObject]:
        result = []
        skip_to = {0: 0, 1: 0, 2: 0, 3: 0}

        for note in notes:
            if note.time_ms < skip_to[note.lane]:
                continue

            if random.random() < self.config.ln_ratio:
                end_time = note.time_ms + random.randint(self.config.min_ln_ms, self.config.max_ln_ms)
                note.end_time_ms = end_time
                skip_to[note.lane] = end_time

            result.append(note)
        return result

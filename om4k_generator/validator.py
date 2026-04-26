from .models import DifficultyConfig, NoteObject
from .style_rules import clamp_max_chord_size
from typing import Dict, List, Optional, Set, Tuple

class Validator:
    @staticmethod
    def validate_and_fix(
        notes: List[NoteObject],
        config: Optional[DifficultyConfig] = None,
        silent_regions: Optional[List[Tuple[int, int]]] = None,
        snap_points: Optional[List[int]] = None,
        min_interval_ms: Optional[int] = None,
    ) -> List[NoteObject]:
        fixed = []
        silent_regions = silent_regions or []
        snap_set: Optional[Set[int]] = set(snap_points) if snap_points is not None else None

        max_chord_size = clamp_max_chord_size(config) if config else 4
        max_jack_length = config.max_jack_length if config and config.max_jack_length > 0 else 9999
        if config and config.key_style == "jack":
            max_jack_length = 9999
        allow_ln = config is None or config.chart_type in ["ln", "hybrid"]
        ln_tail_gap_ms = 30

        if min_interval_ms is None:
            min_interval_ms = 60 if config and config.key_style == "jack" else 40

        rows: Dict[int, List[NoteObject]] = {}
        for n in notes:
            if n.lane not in [0, 1, 2, 3]:
                continue
            rows.setdefault(n.time_ms, []).append(n)

        lane_block_until = {0: -1, 1: -1, 2: -1, 3: -1}
        last_lane_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
        last_ln_tail_time = {0: -999999, 1: -999999, 2: -999999, 3: -999999}
        jack_streaks = {0: 0, 1: 0, 2: 0, 3: 0}

        sorted_silent_regions = sorted(silent_regions)
        silent_index = 0

        for time_ms in sorted(rows):
            if snap_set is not None and time_ms not in snap_set:
                continue

            while silent_index < len(sorted_silent_regions) and sorted_silent_regions[silent_index][1] < time_ms:
                silent_index += 1
            if silent_index < len(sorted_silent_regions) and sorted_silent_regions[silent_index][0] <= time_ms <= sorted_silent_regions[silent_index][1]:
                continue

            accepted_row = []
            accepted_lanes = set()

            for n in sorted(rows[time_ms], key=lambda item: item.lane):
                if len(accepted_row) >= max_chord_size:
                    break
                if n.lane in accepted_lanes:
                    continue
                if time_ms <= lane_block_until[n.lane]:
                    continue
                if time_ms - last_ln_tail_time[n.lane] <= ln_tail_gap_ms:
                    continue
                if time_ms - last_lane_time[n.lane] < min_interval_ms:
                    continue
                if jack_streaks[n.lane] >= max_jack_length:
                    continue

                end_time_ms = n.end_time_ms if allow_ln else None
                if end_time_ms is not None:
                    if end_time_ms <= time_ms:
                        continue
                    if config:
                        min_end = time_ms + config.min_ln_ms
                        max_end = time_ms + config.max_ln_ms
                        end_time_ms = max(min_end, min(end_time_ms, max_end))

                accepted = NoteObject(time_ms=time_ms, lane=n.lane, end_time_ms=end_time_ms)
                accepted_row.append(accepted)
                accepted_lanes.add(n.lane)

            for lane in [0, 1, 2, 3]:
                if lane in accepted_lanes:
                    jack_streaks[lane] += 1
                else:
                    jack_streaks[lane] = 0

            for n in accepted_row:
                fixed.append(n)
                last_lane_time[n.lane] = time_ms
                lane_block_until[n.lane] = n.end_time_ms if n.is_ln else time_ms
                if n.is_ln and n.end_time_ms is not None:
                    last_ln_tail_time[n.lane] = n.end_time_ms

        return fixed

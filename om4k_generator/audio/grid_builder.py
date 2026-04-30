class GridBuilder:
    @staticmethod
    def build(bpm: float, offset_ms: int, duration_ms: int, allowed_subdivisions: list[str]) -> list[int]:
        beat_length_ms = 60000 / bpm
        snap_points = []
        
        # Calculate snap intervals from subdivisions (e.g., '1/4' -> 1/4 beat length)
        # However, music notation like 1/4 note in osu usually means 1/1 beat duration
        # Based on osu, 1/1 is one beat, 1/2 is half beat.
        fractions = []
        for sub in allowed_subdivisions:
            try:
                num, den = map(int, sub.split("/"))
            except ValueError:
                continue
            if num == 1 and den > 0:
                fractions.append(den)
            
        if not fractions:
            fractions = [1, 2, 4]
            
        # build grid
        current_time = float(offset_ms)
        max_t = duration_ms
        
        all_points = set()
        
        while current_time < max_t:
            for divs in fractions:
                step = beat_length_ms / divs
                t = current_time
                for i in range(divs):
                    all_points.add(int(round(t)))
                    t += step
            current_time += beat_length_ms
            
        return sorted(list(all_points))

    @staticmethod
    def snap(onsets: list[int], snap_points: list[int], max_dist_ms=30) -> list[int]:
        import bisect
        valid_onsets = []
        for o in onsets:
            idx = bisect.bisect_left(snap_points, o)
            closest = None
            if idx == 0:
                closest = snap_points[0]
            elif idx == len(snap_points):
                closest = snap_points[-1]
            else:
                left = snap_points[idx-1]
                right = snap_points[idx]
                closest = left if (o - left) < (right - o) else right
                
            if abs(closest - o) <= max_dist_ms:
                valid_onsets.append(closest)
                
        return sorted(list(set(valid_onsets)))

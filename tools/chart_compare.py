import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, List, Optional


def parse_osu_hitobjects(path: Path) -> list[dict]:
    in_hitobjects = False
    notes = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("["):
            in_hitobjects = line == "[HitObjects]"
            continue
        if not in_hitobjects or line.startswith("//"):
            continue

        parts = line.split(",")
        if len(parts) < 6:
            continue

        x = int(float(parts[0]))
        time_ms = int(float(parts[2]))
        obj_type = int(parts[3])
        lane = min(3, max(0, int(x * 4 / 512)))
        end_time: Optional[int] = None
        if obj_type & 128:
            end_time = int(parts[5].split(":")[0])

        notes.append({"time": time_ms, "lane": lane, "end": end_time})

    return notes


def summarize_notes(notes: Iterable[dict]) -> dict:
    notes = sorted(notes, key=lambda n: (n["time"], n["lane"]))
    if not notes:
        return {}

    rows = defaultdict(list)
    for note in notes:
        rows[note["time"]].append(note)

    row_times = sorted(rows)
    intervals = [b - a for a, b in zip(row_times, row_times[1:]) if b > a]
    chord_counts = Counter(len(rows[t]) for t in row_times)

    jack_links = 0
    stream_switches = 0
    previous_lanes = set()
    for t in row_times:
        lanes = {note["lane"] for note in rows[t]}
        if lanes & previous_lanes:
            jack_links += 1
        if previous_lanes and not (lanes & previous_lanes):
            stream_switches += 1
        previous_lanes = lanes

    duration_s = max(1.0, (notes[-1]["time"] - notes[0]["time"]) / 1000.0)
    ln_count = sum(1 for note in notes if note["end"] is not None)
    row_count = len(row_times)

    return {
        "notes": len(notes),
        "rows": row_count,
        "duration_s": round(duration_s, 2),
        "nps": round(len(notes) / duration_s, 2),
        "rps": round(row_count / duration_s, 2),
        "median_row_interval_ms": round(median(intervals), 2) if intervals else 0,
        "max_chord": max(chord_counts),
        "chord_ratio": round(sum(count for size, count in chord_counts.items() if size > 1) / row_count, 3),
        "chord_distribution": dict(sorted(chord_counts.items())),
        "ln_ratio": round(ln_count / len(notes), 3),
        "jack_link_ratio": round(jack_links / row_count, 3),
        "stream_switch_ratio": round(stream_switches / row_count, 3),
    }


def summarize_osu_file(path: Path) -> dict:
    return summarize_notes(parse_osu_hitobjects(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize osu!mania chart structure metrics.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    results = {}
    for path in args.paths:
        if path.is_dir():
            for osu_file in path.rglob("*.osu"):
                results[str(osu_file)] = summarize_osu_file(osu_file)
        else:
            results[str(path)] = summarize_osu_file(path)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

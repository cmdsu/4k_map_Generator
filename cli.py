import argparse
import os
import sys

from om4k_generator.audio_analyzer import AudioAnalyzer
from om4k_generator.calibrator import build_snap_candidates, generate_to_target_sr
from om4k_generator.models import DifficultyConfig
from om4k_generator.osu_exporter import OsuExporter
from om4k_generator.style_rules import (
    DEFAULT_HYBRID_WEIGHTS,
    HYBRID_LN_TENDENCY_RATIO,
    HYBRID_PRESETS,
    chord_enabled_for,
    hybrid_ln_ratio_for_tendency,
    hybrid_weights_for_preset,
    max_chord_bounds_for,
    preserve_allowed_subdivisions,
    recommended_subdivisions,
)


def main():
    parser = argparse.ArgumentParser(description="osu!mania 4K CLI debugger")
    parser.add_argument("--audio", type=str, required=True, help="Audio file name relative to in/")
    parser.add_argument("--target_sr", type=float, default=3.5, help="Target official star rating. Use 0 for unconstrained.")
    parser.add_argument("--sr_tolerance", type=float, default=0.15, help="Allowed official SR tolerance around the target.")
    parser.add_argument("--chart_type", type=str, default="rice", choices=["rice", "ln", "hybrid"])
    parser.add_argument("--key_style", type=str, default="jack", choices=["jack", "stream", "tech", "speed"])
    parser.add_argument("--bpm", type=float, default=0.0, help="Manual BPM. Use 0 for auto.")
    parser.add_argument("--offset", type=int, default=None, help="Manual offset in ms.")
    parser.add_argument("--skip_audio_analysis", action="store_true", help="Use a synthetic analysis for fast local regression tests.")
    parser.add_argument("--duration_ms", type=int, default=60000, help="Duration used with --skip_audio_analysis.")
    parser.add_argument("--subdivisions", type=str, default="auto", help="Comma-separated allowed subdivisions, or auto.")
    parser.add_argument("--max_chord_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.35, help="Pattern variation temperature from 0.0 to 1.0.")
    parser.add_argument("--music_influence", type=float, default=0.65, help="Music-fit influence from 0.0 to 1.0.")
    parser.add_argument("--ln_ratio", type=float, default=0.1)
    parser.add_argument("--hybrid_preset", type=str, default="balanced_pp", choices=sorted(HYBRID_PRESETS.keys()))
    parser.add_argument("--ln_tendency", type=str, default="auto", choices=sorted(HYBRID_LN_TENDENCY_RATIO.keys()))
    parser.add_argument("--hybrid_jack", type=float, default=DEFAULT_HYBRID_WEIGHTS["jack"])
    parser.add_argument("--hybrid_stream", type=float, default=DEFAULT_HYBRID_WEIGHTS["stream"])
    parser.add_argument("--hybrid_tech", type=float, default=DEFAULT_HYBRID_WEIGHTS["tech"])
    parser.add_argument("--hybrid_speed", type=float, default=DEFAULT_HYBRID_WEIGHTS["speed"])
    parser.add_argument("--title", type=str, default="Debug Track")
    parser.add_argument("--artist", type=str, default="CLI Auto")
    parser.add_argument("--version", type=str, default="Debug Version")

    args = parser.parse_args()

    input_path = os.path.join(os.getcwd(), "in", args.audio)
    if not os.path.exists(input_path):
        print(f"Error: input audio not found: {input_path}")
        sys.exit(1)

    out_dir = os.path.join(os.getcwd(), "out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[DEBUG] Reading audio: {input_path}")
    with open(input_path, "rb") as f:
        audio_bytes = f.read()

    if args.skip_audio_analysis:
        if args.bpm <= 0:
            print("Error: --skip_audio_analysis requires --bpm.")
            sys.exit(1)
        step = 60000.0 / args.bpm
        offset = args.offset or 0
        beat_times = []
        t = float(offset)
        while t < args.duration_ms:
            beat_times.append(int(round(t)))
            t += step
        analysis = {
            "bpm": args.bpm,
            "offset_ms": offset,
            "duration_ms": args.duration_ms,
            "onset_times_ms": beat_times,
            "beat_times_ms": beat_times,
            "energy_curve": [1.0 if i % 4 == 0 else 0.35 for i in range(1000)],
            "silent_regions": [],
        }
    else:
        print("[DEBUG] Analyzing audio...")
        analyzer = AudioAnalyzer(
            audio_bytes,
            args.bpm if args.bpm > 0 else None,
            args.offset,
        )
        analysis = analyzer.analyze()
    analysis["offset_ms"] -= 20
    print(f"[DEBUG] BPM: {analysis['bpm']:.2f}, Offset: {analysis['offset_ms']}ms")

    hybrid_weights = hybrid_weights_for_preset(args.hybrid_preset)
    manual_hybrid_weights = any(
        [
            args.hybrid_jack != DEFAULT_HYBRID_WEIGHTS["jack"],
            args.hybrid_stream != DEFAULT_HYBRID_WEIGHTS["stream"],
            args.hybrid_tech != DEFAULT_HYBRID_WEIGHTS["tech"],
            args.hybrid_speed != DEFAULT_HYBRID_WEIGHTS["speed"],
        ]
    )
    if manual_hybrid_weights:
        hybrid_weights = {
            "jack": args.hybrid_jack,
            "stream": args.hybrid_stream,
            "tech": args.hybrid_tech,
            "speed": args.hybrid_speed,
        }
    ln_ratio = args.ln_ratio
    if args.chart_type == "hybrid":
        ln_ratio = hybrid_ln_ratio_for_tendency(args.ln_tendency, args.hybrid_preset)
        if args.ln_ratio != 0.1:
            ln_ratio = args.ln_ratio
    key_style = None if args.chart_type == "hybrid" else args.key_style
    chord_enabled = chord_enabled_for(args.chart_type, key_style, hybrid_weights)
    _, _, default_chord = max_chord_bounds_for(args.chart_type, key_style, hybrid_weights)
    max_chord_size = args.max_chord_size or default_chord

    config = DifficultyConfig(
        version=args.version,
        target_star=args.target_sr if args.target_sr > 0 else None,
        target_msd=None,
        chart_type=args.chart_type,
        key_style=key_style,
        allowed_subdivisions=preserve_allowed_subdivisions(
            recommended_subdivisions(analysis["bpm"], args.chart_type, key_style, args.target_sr if args.target_sr > 0 else None)
            if args.subdivisions == "auto"
            else (s.strip() for s in args.subdivisions.split(","))
        ),
        chord_enabled=chord_enabled,
        max_chord_size=max_chord_size,
        chord_probability=0.35,
        max_jack_length=4,
        max_anchor_length=4,
        hand_balance=0.5,
        ln_ratio=ln_ratio if args.chart_type in ["ln", "hybrid"] else 0.0,
        min_ln_ms=120,
        max_ln_ms=1000,
        hybrid_weights=hybrid_weights,
        pattern_temperature=max(0.0, min(1.0, args.temperature)),
        music_influence=max(0.0, min(1.0, args.music_influence)),
    )

    print("[DEBUG] Building snap candidates...")
    snapped = build_snap_candidates(analysis, config)

    rating_label = "official SR"
    print(f"[DEBUG] Generating and calibrating {rating_label}...")
    best_notes, best_est_sr, target_met, attempts = generate_to_target_sr(config, analysis, snapped, tolerance=args.sr_tolerance)

    if config.target_star is not None and not target_met:
        print(f"[ERROR] Could not strictly reach target {rating_label}. Target: {config.target_star:.2f}, actual: {best_est_sr:.2f}.")
        print(f"[ERROR] Increase subdivisions/chords/max chord size, or lower target {rating_label}.")
        sys.exit(2)

    print(f"[DEBUG] Final {rating_label}: {best_est_sr} ({attempts} attempts)")

    osu_str = OsuExporter.export(
        config,
        best_notes,
        analysis["bpm"],
        analysis["offset_ms"],
        args.audio,
        None,
        args.artist,
        args.title,
        "CLI_Debugger",
    )

    out_file = os.path.join(out_dir, f"{args.artist} - {args.title} [{args.version}].osu")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(osu_str)

    print(f"[DEBUG] Wrote: {out_file}")


if __name__ == "__main__":
    main()

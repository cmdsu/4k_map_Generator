from .models import DifficultyConfig, NoteObject
from typing import List

class OsuExporter:
    @staticmethod
    def export(
        config: DifficultyConfig, 
        notes: List[NoteObject], 
        bpm: float, 
        offset: int, 
        audio_filename: str, 
        bg_filename: str = None,
        artist: str = "Artist", 
        title: str = "Title", 
        creator: str = "Creator"
    ) -> str:
        
        lines = [
            "osu file format v14", "",
            "[General]",
            f"AudioFilename: {audio_filename}",
            "AudioLeadIn: 0",
            "PreviewTime: -1",
            "Countdown: 0",
            "SampleSet: Normal",
            "StackLeniency: 0.7",
            "Mode: 3",
            "LetterboxInBreaks: 0",
            "SpecialStyle: 0",
            f"WidescreenStoryboard: {1 if bg_filename else 0}",
            "",
            "[Editor]",
            "DistanceSpacing: 1",
            "BeatDivisor: 4",
            "GridSize: 4",
            "TimelineZoom: 1",
            "",
            "[Metadata]",
            f"Title:{title}",
            f"TitleUnicode:{title}",
            f"Artist:{artist}",
            f"ArtistUnicode:{artist}",
            f"Creator:{creator}",
            f"Version:{config.version}",
            f"Source:",
            f"Tags:4k mania auto_generated",
            "BeatmapID:0",
            "BeatmapSetID:-1",
            "",
            "[Difficulty]",
            "HPDrainRate:8",
            "CircleSize:4",
            "OverallDifficulty:8",
            "ApproachRate:5",
            "SliderMultiplier:1.4",
            "SliderTickRate:1",
            "",
            "[Events]",
            "//Background and Video events"
        ]
        
        if bg_filename:
            lines.append(f'0,0,"{bg_filename}",0,0')
            
        lines.extend([
            "//Break Periods",
            "//Storyboard Layer 0 (Background)",
            "//Storyboard Layer 1 (Fail)",
            "//Storyboard Layer 2 (Pass)",
            "//Storyboard Layer 3 (Foreground)",
            "//Storyboard Layer 4 (Overlay)",
            "//Storyboard Sound Samples",
            "",
            "[TimingPoints]"
        ])
        
        beat_len = 60000 / bpm
        lines.append(f"{offset},{beat_len},4,1,0,100,1,0")
        lines.extend(["", "[HitObjects]"])
        
        lane_coords = {0: 64, 1: 192, 2: 320, 3: 448}
        
        for n in sorted(notes, key=lambda x: (x.time_ms, x.lane)):
            x = lane_coords[n.lane]
            if n.is_ln:
                lines.append(f"{x},192,{n.time_ms},128,0,{n.end_time_ms}:0:0:0:0:")
            else:
                lines.append(f"{x},192,{n.time_ms},1,0,0:0:0:0:")
                
        return "\n".join(lines)

from typing import Dict, Iterable, Literal, Optional

from .models import DifficultyConfig

KeyStyle = Literal["jack", "stream", "tech", "speed"]

KEY_STYLES: tuple[KeyStyle, ...] = ("jack", "stream", "tech", "speed")


def chord_enabled_for(chart_type: str, key_style: Optional[str], _style_mix: Dict[str, float]) -> bool:
    return key_style in ["jack", "stream", "speed", "tech"]


def max_chord_bounds_for(chart_type: str, key_style: Optional[str], _style_mix: Dict[str, float]) -> tuple[int, int, int]:
    if key_style == "jack":
        return 2, 4, 4
    if key_style == "stream":
        return 2, 3, 3
    if key_style == "speed":
        return 1, 3, 1
    if key_style == "tech":
        return 1, 4, 3
    return 1, 3, 1


def clamp_max_chord_size(config: DifficultyConfig, style: Optional[str] = None) -> int:
    active_style = style or config.key_style
    if active_style == "jack":
        return max(2, min(4, config.max_chord_size))
    if active_style == "stream":
        return max(2, min(3, config.max_chord_size))
    if active_style == "speed":
        return max(1, min(3, config.max_chord_size))
    if active_style == "tech":
        return max(1, min(4, config.max_chord_size))
    return max(1, min(3, config.max_chord_size))


def recommended_subdivisions(bpm: float, chart_type: str, key_style: Optional[str], target_star: Optional[float]) -> list[str]:
    if key_style == "jack":
        return ["1/4"] if bpm <= 180 else ["1/2", "1/4"]
    if key_style == "stream":
        if bpm <= 320:
            return ["1/4", "1/5", "1/6", "1/7", "1/8"]
        return ["1/2", "1/3", "1/4", "1/5", "1/6"]
    if key_style == "speed":
        if target_star is not None and target_star >= 6.8:
            return ["1/4", "1/6", "1/8", "1/10", "1/12", "1/16"]
        return ["1/4", "1/5", "1/6", "1/7", "1/8", "1/10", "1/12"] if 140 <= bpm <= 210 else ["1/3", "1/4", "1/6", "1/8", "1/10", "1/12"]
    if key_style == "tech":
        return ["1/4", "1/6", "1/8"]
    if target_star is not None:
        return ["1/2", "1/4", "1/8"]
    return ["1/2", "1/4", "1/8"]


def preserve_allowed_subdivisions(selected: Iterable[str]) -> list[str]:
    values = []
    for value in selected:
        if value not in values:
            values.append(value)
    return values or ["1/2", "1/4", "1/8"]

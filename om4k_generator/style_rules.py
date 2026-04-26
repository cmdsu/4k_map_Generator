from typing import Dict, Iterable, Literal, Optional

from .models import DifficultyConfig

KeyStyle = Literal["jack", "stream", "tech", "speed"]

KEY_STYLES: tuple[KeyStyle, ...] = ("jack", "stream", "tech", "speed")
DEFAULT_HYBRID_WEIGHTS: Dict[str, float] = {
    "jack": 0.25,
    "stream": 0.25,
    "tech": 0.25,
    "speed": 0.25,
}


def normalize_hybrid_weights(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned = {style: max(0.0, float(weights.get(style, 0.0))) for style in KEY_STYLES}
    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_HYBRID_WEIGHTS.copy()
    return {style: value / total for style, value in cleaned.items()}


def chord_enabled_for(chart_type: str, key_style: Optional[str], hybrid_weights: Dict[str, float]) -> bool:
    if chart_type == "vibro":
        return True
    if chart_type == "hybrid":
        weights = normalize_hybrid_weights(hybrid_weights)
        return weights["jack"] > 0 or weights["stream"] > 0
    return key_style in ["jack", "stream"]


def max_chord_bounds_for(chart_type: str, key_style: Optional[str], hybrid_weights: Dict[str, float]) -> tuple[int, int, int]:
    if chart_type == "vibro":
        return 2, 4, 2
    if chart_type == "hybrid":
        weights = normalize_hybrid_weights(hybrid_weights)
        upper = 4 if weights["jack"] > 0 else 3 if weights["stream"] > 0 else 2
        return 2, upper, min(2, upper)
    if key_style == "jack":
        return 2, 4, 4
    if key_style == "stream":
        return 2, 3, 2
    return 1, 3, 1


def clamp_max_chord_size(config: DifficultyConfig, style: Optional[str] = None) -> int:
    active_style = style or config.key_style
    if config.chart_type == "vibro":
        return max(2, min(4, config.max_chord_size))
    if active_style == "jack":
        return max(2, min(4, config.max_chord_size))
    if active_style == "stream":
        return max(2, min(3, config.max_chord_size))
    return max(1, min(3, config.max_chord_size))


def recommended_subdivisions(bpm: float, chart_type: str, key_style: Optional[str], target_star: Optional[float]) -> list[str]:
    if chart_type == "vibro":
        return ["1/4", "1/8", "1/12", "1/16"]
    if key_style == "jack":
        return ["1/4"] if bpm <= 180 else ["1/2"]
    if target_star is not None:
        return ["1/2", "1/4", "1/8"]
    if key_style == "stream":
        return ["1/4", "1/8"] if bpm <= 320 else ["1/2", "1/4"]
    if key_style == "speed":
        return ["1/4", "1/5", "1/6", "1/7", "1/8"] if 140 <= bpm <= 210 else ["1/3", "1/4", "1/6", "1/8"]
    if key_style == "tech":
        return ["1/3", "1/4", "1/6", "1/8", "1/12"]
    if chart_type == "hybrid":
        return ["1/3", "1/4", "1/6", "1/8"]
    return ["1/2", "1/4", "1/8"]


def preserve_allowed_subdivisions(selected: Iterable[str]) -> list[str]:
    values = []
    for value in selected:
        if value not in values:
            values.append(value)
    return values or ["1/2", "1/4", "1/8"]

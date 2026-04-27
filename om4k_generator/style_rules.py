from typing import Dict, Iterable, Literal, Optional

from .models import DifficultyConfig

KeyStyle = Literal["jack", "stream", "tech", "speed"]

KEY_STYLES: tuple[KeyStyle, ...] = ("jack", "stream", "tech", "speed")
DEFAULT_HYBRID_WEIGHTS: Dict[str, float] = {
    "jack": 0.10,
    "stream": 0.32,
    "tech": 0.38,
    "speed": 0.20,
}

HYBRID_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced_pp": {"jack": 0.10, "stream": 0.32, "tech": 0.38, "speed": 0.20},
    "ln_hybrid": {"jack": 0.05, "stream": 0.34, "tech": 0.44, "speed": 0.17},
    "rice_hybrid": {"jack": 0.08, "stream": 0.42, "tech": 0.28, "speed": 0.22},
    "tech_hybrid": {"jack": 0.08, "stream": 0.24, "tech": 0.54, "speed": 0.14},
    "speed_hybrid": {"jack": 0.05, "stream": 0.30, "tech": 0.25, "speed": 0.40},
}

HYBRID_PRESET_LN_RATIO: Dict[str, float] = {
    "balanced_pp": 0.55,
    "ln_hybrid": 0.72,
    "rice_hybrid": 0.45,
    "tech_hybrid": 0.18,
    "speed_hybrid": 0.30,
}

HYBRID_LN_TENDENCY_RATIO: Dict[str, float] = {
    "auto": -1.0,
    "few": 0.18,
    "medium": 0.45,
    "many": 0.68,
}


def hybrid_weights_for_preset(preset: str) -> Dict[str, float]:
    return normalize_hybrid_weights(HYBRID_PRESETS.get(preset, DEFAULT_HYBRID_WEIGHTS))


def hybrid_ln_ratio_for_tendency(tendency: str, preset: str) -> float:
    value = HYBRID_LN_TENDENCY_RATIO.get(tendency, HYBRID_LN_TENDENCY_RATIO["auto"])
    if value >= 0:
        return value
    return HYBRID_PRESET_LN_RATIO.get(preset, HYBRID_PRESET_LN_RATIO["balanced_pp"])


def normalize_hybrid_weights(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned = {style: max(0.0, float(weights.get(style, 0.0))) for style in KEY_STYLES}
    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_HYBRID_WEIGHTS.copy()
    return {style: value / total for style, value in cleaned.items()}


def chord_enabled_for(chart_type: str, key_style: Optional[str], hybrid_weights: Dict[str, float]) -> bool:
    if chart_type == "hybrid":
        weights = normalize_hybrid_weights(hybrid_weights)
        return weights["jack"] > 0 or weights["stream"] > 0 or weights["tech"] > 0
    return key_style in ["jack", "stream", "speed", "tech"]


def max_chord_bounds_for(chart_type: str, key_style: Optional[str], hybrid_weights: Dict[str, float]) -> tuple[int, int, int]:
    if chart_type == "hybrid":
        weights = normalize_hybrid_weights(hybrid_weights)
        upper = 4 if weights["jack"] > 0 else 3 if weights["stream"] > 0 else 2
        return 2, upper, min(4, upper)
    if key_style == "jack":
        return 2, 4, 4
    if key_style == "stream":
        return 2, 3, 3
    if key_style == "speed":
        return 1, 3, 2
    if key_style == "tech":
        return 1, 4, 3
    return 1, 3, 1


def clamp_max_chord_size(config: DifficultyConfig, style: Optional[str] = None) -> int:
    active_style = style or config.key_style
    if config.chart_type == "hybrid" and active_style is None:
        return max(1, min(4, config.max_chord_size))
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
    if chart_type == "hybrid":
        return ["1/2", "1/4", "1/8"]
    if key_style == "jack":
        return ["1/4"] if bpm <= 180 else ["1/2", "1/4"]
    if key_style == "stream":
        if bpm <= 320:
            return ["1/4", "1/5", "1/6", "1/7", "1/8"]
        return ["1/2", "1/3", "1/4", "1/5", "1/6"]
    if key_style == "speed":
        return ["1/4", "1/5", "1/6", "1/7", "1/8", "1/10", "1/12"] if 140 <= bpm <= 210 else ["1/3", "1/4", "1/6", "1/8", "1/10", "1/12"]
    if key_style == "tech":
        return ["1/3", "1/4", "1/5", "1/6", "1/8", "1/12", "1/16"]
    if target_star is not None:
        return ["1/2", "1/4", "1/8"]
    return ["1/2", "1/4", "1/8"]


def preserve_allowed_subdivisions(selected: Iterable[str]) -> list[str]:
    values = []
    for value in selected:
        if value not in values:
            values.append(value)
    return values or ["1/2", "1/4", "1/8"]

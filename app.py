import html

import streamlit as st

from om4k_generator.audio_analyzer import AudioAnalyzer
from om4k_generator.calibrator import build_snap_candidates, generate_to_target_sr
from om4k_generator.models import DifficultyConfig
from om4k_generator.osu_exporter import OsuExporter
from om4k_generator.packager import Packager
from om4k_generator.style_rules import (
    DEFAULT_HYBRID_WEIGHTS,
    HYBRID_LN_TENDENCY_RATIO,
    HYBRID_PRESETS,
    chord_enabled_for,
    hybrid_ln_ratio_for_tendency,
    hybrid_weights_for_preset,
    max_chord_bounds_for,
    normalize_hybrid_weights,
    preserve_allowed_subdivisions,
    recommended_subdivisions,
)


st.set_page_config(page_title="osu!mania 4K Chart Generator", layout="wide")

def esc(value) -> str:
    return html.escape(str(value), quote=True)

TEXT = {
    "zh": {
        "lang_label": "语言 / Language",
        "app_title": "osu!mania 4K 自动练习谱生成器",
        "app_caption": "基于规则的 4K 谱面生成器：重点控制风格、SR、音乐贴合、LN 合法性和可玩性。",
        "upload_header": "输入文件",
        "upload_audio": "上传音频",
        "upload_bg": "上传背景图（可选）",
        "metadata_header": "谱面信息",
        "title": "标题",
        "artist": "艺术家",
        "creator": "制作者",
        "difficulty_name": "难度名",
        "difficulty_header": "难度控制",
        "target_sr": "目标 SR（0 表示不限制）",
        "sr_tolerance": "SR 波动范围（±）",
        "temperature": "Pattern temperature（变化程度）",
        "temperature_help": "越低越稳定；越高越会改变轨道、同押布局和短模块排列，但仍会尽量贴近目标 SR。",
        "music_influence": "音乐贴合度",
        "music_influence_help": "越低越偏纯键型训练；越高越优先把重音、kick 和明显 onset 对应到 note / 多押。",
        "chart_header": "谱面类型与键型",
        "chart_type": "谱面类型",
        "key_style": "键型风格",
        "hybrid_preset": "Hybrid 方向",
        "ln_tendency": "LN 倾向",
        "hybrid_advanced": "高级调试：手动覆盖 Hybrid 软比例",
        "use_custom_hybrid": "手动设置键型软比例",
        "use_custom_ln": "手动设置 LN 数值",
        "hybrid_caption": "Hybrid 会按音乐段落自动选择 LN 骨架、rice burst、tech/release 和 speed 压力；这里选择的是方向，不是硬比例。",
        "normalized_weights": "当前软偏好",
        "ln_ratio": "LN 数值",
        "ln_ratio_help": "普通模式建议使用 LN 倾向。这个数值只用于高级调试，算法仍会按音乐段落决定实际 LN。",
        "ln_lengths": "LN 长度限制",
        "min_ln": "最短 LN（ms）",
        "max_ln": "最长 LN（ms）",
        "timing_header": "节拍与分辨率",
        "manual_bpm": "手动 BPM（0 表示自动检测）",
        "manual_offset_toggle": "使用手动 offset",
        "manual_offset": "手动 offset（ms）",
        "offset_note": "生成时会在检测/手动 offset 基础上固定减 20ms，用于修正当前导出偏移。",
        "allowed_subdivisions": "允许分辨率",
        "subdivision_help": "这些分辨率会真实限制作图候选。Hybrid 只保留 1/2、1/4、1/8，避免 note 过细碎。",
        "chord_header": "同押控制",
        "max_chord": "最大同时按键数",
        "chord_caption": "当前风格会自动管理多押；这里仅限制同一行最多几个键。",
        "generate": "生成 .osz",
        "need_audio": "请先上传音频文件。",
        "analyzing": "正在分析音频...",
        "generating": "正在生成并校准 SR...",
        "packaging": "正在打包输出...",
        "final": "生成结果",
        "target_failed": "未能严格达到目标 SR。请提高分辨率/最大同押，或降低目标 SR。",
        "download": "下载 .osz",
        "settings_preview": "当前设置预览",
        "target_met": "目标达成",
        "target_unlimited": "未限制",
        "yes": "是",
        "no": "否",
        "style_guide": "风格说明",
        "chart_rice": "Rice",
        "chart_ln": "LN",
        "chart_hybrid": "Hybrid",
        "style_jack": "Jack / 叠键",
        "style_stream": "Stream / 切",
        "style_tech": "Tech / 综合技巧",
        "style_speed": "Speed / 单点高速",
        "preset_balanced_pp": "Balanced PP（推荐）",
        "preset_ln_hybrid": "LN Hybrid",
        "preset_rice_hybrid": "Rice Hybrid",
        "preset_tech_hybrid": "Tech Hybrid",
        "preset_speed_hybrid": "Speed Hybrid",
        "ln_auto": "Auto（按方向自动）",
        "ln_few": "少",
        "ln_medium": "中",
        "ln_many": "多",
    },
    "en": {
        "lang_label": "Language / 语言",
        "app_title": "osu!mania 4K Auto Practice Chart Generator",
        "app_caption": "A rule-based 4K generator focused on style, SR targeting, music fit, LN legality, and playability.",
        "upload_header": "Input Files",
        "upload_audio": "Upload audio",
        "upload_bg": "Upload background (optional)",
        "metadata_header": "Metadata",
        "title": "Title",
        "artist": "Artist",
        "creator": "Creator",
        "difficulty_name": "Difficulty name",
        "difficulty_header": "Difficulty",
        "target_sr": "Target SR (0 for unconstrained)",
        "sr_tolerance": "SR tolerance (+/-)",
        "temperature": "Pattern temperature",
        "temperature_help": "Lower values keep patterns steadier; higher values vary lanes, chord layouts, and short modules while targeting SR.",
        "music_influence": "Music fit",
        "music_influence_help": "Lower values favor pure style practice; higher values prioritize accents, kicks, and clear onsets for notes/chords.",
        "chart_header": "Chart Type and Style",
        "chart_type": "Chart type",
        "key_style": "Key style",
        "hybrid_preset": "Hybrid direction",
        "ln_tendency": "LN tendency",
        "hybrid_advanced": "Advanced debug: override Hybrid soft weights",
        "use_custom_hybrid": "Manually set style soft weights",
        "use_custom_ln": "Manually set LN value",
        "hybrid_caption": "Hybrid chooses LN skeletons, rice bursts, tech/release, and speed pressure by music section. This is a direction, not a hard ratio.",
        "normalized_weights": "Current soft bias",
        "ln_ratio": "LN value",
        "ln_ratio_help": "Use LN tendency in normal mode. This value is for advanced debugging; actual LN placement remains section-aware.",
        "ln_lengths": "LN length limits",
        "min_ln": "Minimum LN (ms)",
        "max_ln": "Maximum LN (ms)",
        "timing_header": "Timing and Subdivisions",
        "manual_bpm": "Manual BPM (0 for auto)",
        "manual_offset_toggle": "Use manual offset",
        "manual_offset": "Manual offset (ms)",
        "offset_note": "Export applies a fixed -20ms correction on top of detected/manual offset.",
        "allowed_subdivisions": "Allowed subdivisions",
        "subdivision_help": "These values really constrain chart candidates. Hybrid only keeps 1/2, 1/4, and 1/8 to avoid overly fragmented notes.",
        "chord_header": "Chord Control",
        "max_chord": "Max simultaneous keys",
        "chord_caption": "The active style manages chord use automatically; this only limits the largest row size.",
        "generate": "Generate .osz",
        "need_audio": "Please upload an audio file first.",
        "analyzing": "Analyzing audio...",
        "generating": "Generating and calibrating SR...",
        "packaging": "Packaging output...",
        "final": "Result",
        "target_failed": "Could not strictly reach the target SR. Increase subdivisions/max chord size or lower the target.",
        "download": "Download .osz",
        "settings_preview": "Settings Preview",
        "target_met": "Target met",
        "target_unlimited": "Unconstrained",
        "yes": "Yes",
        "no": "No",
        "style_guide": "Style Guide",
        "chart_rice": "Rice",
        "chart_ln": "LN",
        "chart_hybrid": "Hybrid",
        "style_jack": "Jack / Chordjack",
        "style_stream": "Stream",
        "style_tech": "Tech",
        "style_speed": "Speed",
        "preset_balanced_pp": "Balanced PP (Recommended)",
        "preset_ln_hybrid": "LN Hybrid",
        "preset_rice_hybrid": "Rice Hybrid",
        "preset_tech_hybrid": "Tech Hybrid",
        "preset_speed_hybrid": "Speed Hybrid",
        "ln_auto": "Auto by direction",
        "ln_few": "Few",
        "ln_medium": "Medium",
        "ln_many": "Many",
    },
}

STYLE_DESCRIPTIONS = {
    "zh": {
        "jack": "低星小叠，中星重叠，高星大叠/满叠；默认最大 4 押，重点是稳定叠键而不是切。",
        "stream": "从轻切 jumpstream 到强切 handstream；默认最大 3 押，避免 4141 式 jack 化。",
        "speed": "以高速单点和少量重音同押为主，适合细分辨率的短压力段。",
        "tech": "混合爆发、楼梯、短锚、切换与 release-tech，强调局部技巧而不是长段 speed。",
        "hybrid": "按音乐段落自动混合 LN 骨架、rice burst、tech/release 与 speed 压力，不再追求固定百分比。",
    },
    "en": {
        "jack": "Low SR uses smaller stacks, mid SR heavier stacks, high SR large/full stacks. Default max chord is 4 and the goal is stacked jack, not stream.",
        "stream": "Evolves from light jumpstream to stronger handstream. Max chord is 3 by default to avoid turning into 4141 jack.",
        "speed": "Fast single-note emphasis with sparse accent chords and short fine-snap pressure sections.",
        "tech": "Mixes bursts, stairs, short anchors, switching, and release-tech without becoming long speed streams.",
        "hybrid": "Section-aware mixing of LN skeletons, rice bursts, tech/release, and speed pressure, not fixed percentages.",
    },
}

CHART_LABELS = {"rice": "chart_rice", "ln": "chart_ln", "hybrid": "chart_hybrid"}
STYLE_LABELS = {"jack": "style_jack", "stream": "style_stream", "tech": "style_tech", "speed": "style_speed"}
PRESET_LABELS = {
    "balanced_pp": "preset_balanced_pp",
    "ln_hybrid": "preset_ln_hybrid",
    "rice_hybrid": "preset_rice_hybrid",
    "tech_hybrid": "preset_tech_hybrid",
    "speed_hybrid": "preset_speed_hybrid",
}
LN_TENDENCY_LABELS = {"auto": "ln_auto", "few": "ln_few", "medium": "ln_medium", "many": "ln_many"}
SUBDIVISION_OPTIONS = ["1/1", "1/2", "1/3", "1/4", "1/5", "1/6", "1/7", "1/8", "1/10", "1/12", "1/16"]
HYBRID_SUBDIVISION_OPTIONS = ["1/2", "1/4", "1/8"]


def tr(key: str) -> str:
    return TEXT[st.session_state.get("lang", "zh")][key]


def note_summary(notes):
    rows = {}
    ln_count = 0
    for note in notes:
        rows.setdefault(note.time_ms, 0)
        rows[note.time_ms] += 1
        if note.is_ln:
            ln_count += 1
    return {
        "notes": len(notes),
        "ln": ln_count,
        "rice": len(notes) - ln_count,
        "rows": len(rows),
        "max_chord": max(rows.values()) if rows else 0,
    }


def style_description(lang: str, chart_type: str, key_style):
    if chart_type == "hybrid":
        return STYLE_DESCRIPTIONS[lang]["hybrid"]
    if key_style:
        return STYLE_DESCRIPTIONS[lang][key_style]
    return ""


if "lang" not in st.session_state:
    st.session_state.lang = "zh"

with st.sidebar:
    language_label = TEXT[st.session_state.lang]["lang_label"]
    selected_lang_label = st.selectbox(language_label, ["中文", "English"], index=0 if st.session_state.lang == "zh" else 1)
    st.session_state.lang = "zh" if selected_lang_label == "中文" else "en"

lang = st.session_state.lang

theme_css_vars = """
        --st-bg: var(--background-color, #f8fffb);
        --st-bg-2: var(--secondary-background-color, #eef7f4);
        --st-text: var(--text-color, #102423);
        --st-primary: var(--primary-color, #16bfa7);
        --app-bg: radial-gradient(circle at 10% 8%, color-mix(in srgb, var(--st-primary) 18%, transparent), transparent 32%), radial-gradient(circle at 82% 2%, rgba(244, 166, 42, 0.16), transparent 30%), linear-gradient(135deg, var(--st-bg) 0%, var(--st-bg-2) 100%);
        --text: var(--st-text);
        --muted: color-mix(in srgb, var(--st-text) 64%, transparent);
        --panel: color-mix(in srgb, var(--st-bg-2) 78%, transparent);
        --panel-soft: color-mix(in srgb, var(--st-bg) 74%, transparent);
        --sidebar: color-mix(in srgb, var(--st-bg-2) 88%, transparent);
        --input-bg: color-mix(in srgb, var(--st-bg) 88%, transparent);
        --line: color-mix(in srgb, var(--st-primary) 28%, transparent);
        --line-strong: color-mix(in srgb, var(--st-primary) 56%, transparent);
        --grid: color-mix(in srgb, var(--st-primary) 10%, transparent);
        --cyan: var(--st-primary);
        --acid: #8fcf23;
        --amber: #f4a62a;
        --coral: #ff6b4a;
        --blue: #268bd8;
        --button-text: #061112;
        --shadow: 0 24px 72px color-mix(in srgb, var(--st-text) 16%, transparent);
        --hero-bg: linear-gradient(135deg, color-mix(in srgb, var(--st-bg) 88%, transparent), color-mix(in srgb, var(--st-bg-2) 76%, transparent)), radial-gradient(circle at 18% 18%, color-mix(in srgb, var(--st-primary) 18%, transparent), transparent 34%), radial-gradient(circle at 86% 28%, rgba(244,166,42,0.16), transparent 28%);
        --scan-opacity: 0.46;
""".strip()

st.markdown(
    """
    <style>
    :root {
__THEME_CSS_VARS__
    }

    .stApp {
        color: var(--text);
        background: var(--app-bg);
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(var(--grid) 1px, transparent 1px),
            linear-gradient(90deg, var(--grid) 1px, transparent 1px),
            linear-gradient(90deg, transparent 0 49%, rgba(255,255,255,0.12) 50%, transparent 51%);
        background-size: 46px 46px, 46px 46px, 100% 100%;
        mask-image: linear-gradient(to bottom, rgba(0,0,0,0.56), rgba(0,0,0,0.24) 68%, transparent);
    }

    .stApp::after {
        content: "";
        position: fixed;
        left: -16%;
        top: 10%;
        width: 132%;
        height: 2px;
        pointer-events: none;
        background: linear-gradient(90deg, transparent, var(--line-strong), rgba(244,166,42,0.44), transparent);
        box-shadow: 0 0 28px var(--line-strong);
        animation: deckScan 8.5s ease-in-out infinite;
        opacity: var(--scan-opacity);
    }

    @keyframes deckScan {
        0% { transform: translateY(-22vh) rotate(-5deg); opacity: 0; }
        18% { opacity: var(--scan-opacity); }
        70% { opacity: calc(var(--scan-opacity) * 0.72); }
        100% { transform: translateY(88vh) rotate(-5deg); opacity: 0; }
    }

    @keyframes gridPulse {
        0%, 100% { box-shadow: var(--shadow), inset 0 0 0 rgba(22,191,167,0); }
        50% { box-shadow: var(--shadow), 0 0 42px rgba(22,191,167,0.16), inset 0 0 28px rgba(22,191,167,0.08); }
    }

    @keyframes laneDance {
        0%, 100% { transform: scaleY(0.34); opacity: 0.50; }
        38% { transform: scaleY(0.95); opacity: 1; }
        70% { transform: scaleY(0.58); opacity: 0.76; }
    }

    @keyframes orbit {
        to { transform: rotate(360deg); }
    }

    @keyframes panelRise {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.7rem;
        padding-bottom: 4rem;
    }

    html, body, [class*="css"], .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stMultiSelect {
        font-family: "Bahnschrift", "Cascadia Code", "Microsoft YaHei UI", "Segoe UI", sans-serif;
    }

    h1, h2, h3, .studio-title, .album-title, .studio-strip-title {
        font-family: "Bahnschrift", "Microsoft YaHei UI", "Segoe UI", sans-serif !important;
        letter-spacing: -0.035em;
    }

    h2, h3, label, p, span, div {
        color: inherit;
    }

    h2, h3 {
        color: var(--text);
        text-shadow: 0 0 20px rgba(22,191,167,0.12);
    }

    section[data-testid="stSidebar"] {
        background: var(--sidebar);
        backdrop-filter: blur(22px);
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .album-hero {
        position: relative;
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(290px, 0.72fr);
        gap: 1.6rem;
        align-items: center;
        overflow: hidden;
        min-height: 300px;
        padding: 1.8rem;
        border-radius: 28px;
        background: var(--hero-bg);
        border: 1px solid var(--line);
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.24);
        isolation: isolate;
        animation: gridPulse 6s ease-in-out infinite;
    }

    .album-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(90deg, var(--grid) 1px, transparent 1px),
            linear-gradient(var(--grid) 1px, transparent 1px),
            repeating-linear-gradient(135deg, transparent 0 18px, rgba(143,207,35,0.07) 19px, transparent 20px);
        background-size: 30px 30px, 30px 30px, 100% 100%;
        z-index: -1;
    }

    .album-hero::after {
        content: "";
        position: absolute;
        width: 42%;
        height: 180%;
        right: -10%;
        top: -42%;
        background: linear-gradient(90deg, transparent, rgba(22,191,167,0.16), transparent);
        transform: rotate(18deg);
        filter: blur(1px);
    }

    .studio-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        color: var(--cyan);
        background: rgba(22,191,167,0.10);
        border: 1px solid var(--line);
        font-size: 0.76rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        box-shadow: inset 0 0 18px rgba(22,191,167,0.06);
    }

    .studio-eyebrow::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--acid);
        box-shadow: 0 0 18px var(--acid);
    }

    .studio-title {
        margin: 0.95rem 0 0.68rem;
        color: var(--text);
        font-size: clamp(2.15rem, 5.2vw, 5.0rem);
        line-height: 0.92;
        max-width: 860px;
        text-transform: uppercase;
        text-shadow: 0 0 26px rgba(22,191,167,0.12);
    }

    .studio-copy {
        max-width: 760px;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.7;
    }

    .studio-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.2rem;
    }

    .studio-pill, .album-tags span {
        padding: 0.46rem 0.64rem;
        border-radius: 10px;
        color: var(--text);
        background: rgba(22,191,167,0.09);
        border: 1px solid var(--line);
        box-shadow: inset 0 0 14px rgba(22,191,167,0.04);
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }

    .cover-stage {
        position: relative;
        min-height: 242px;
        display: grid;
        place-items: center;
    }

    .record-disc {
        position: absolute;
        width: 226px;
        height: 226px;
        right: 8%;
        border-radius: 50%;
        background:
            radial-gradient(circle at center, rgba(255,255,255,0.90) 0 4%, rgba(16,36,35,0.88) 5% 13%, transparent 14%),
            repeating-radial-gradient(circle at center, var(--line) 0 1px, transparent 2px 15px),
            conic-gradient(from 0deg, transparent 0 18%, var(--line-strong) 18% 20%, transparent 20% 50%, rgba(244,166,42,0.35) 50% 52%, transparent 52% 100%);
        border: 1px solid var(--line);
        box-shadow: 0 0 46px rgba(22,191,167,0.14), inset 0 0 38px rgba(22,191,167,0.06);
        animation: orbit 18s linear infinite;
    }

    .cover-card {
        position: relative;
        width: min(278px, 78vw);
        aspect-ratio: 1;
        padding: 1rem;
        border-radius: 22px;
        background:
            linear-gradient(135deg, var(--panel), var(--panel-soft)),
            repeating-linear-gradient(90deg, var(--grid) 0 1px, transparent 1px 22px);
        border: 1px solid var(--line-strong);
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.16);
        overflow: hidden;
    }

    .cover-card::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, transparent 0 48%, rgba(143,207,35,0.14) 49%, transparent 50% 100%);
        background-size: 100% 16px;
        opacity: 0.44;
    }

    .cover-inner {
        position: relative;
        z-index: 1;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 1.05rem;
        border-radius: 17px;
        color: var(--text);
        background: rgba(255,255,255,0.16);
        border: 1px solid var(--line);
    }

    .cover-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--acid);
    }

    .cover-title {
        font-family: "Bahnschrift", "Microsoft YaHei UI", sans-serif;
        font-size: 2rem;
        line-height: 0.95;
        letter-spacing: -0.04em;
        text-transform: uppercase;
        color: var(--text);
    }

    .wave-bars {
        display: grid;
        grid-template-columns: repeat(14, 1fr);
        align-items: end;
        gap: 5px;
        height: 66px;
    }

    .wave-bars span {
        width: 100%;
        height: 100%;
        min-width: 7px;
        border-radius: 4px 4px 0 0;
        transform-origin: bottom;
        background: linear-gradient(to top, var(--blue), var(--cyan), var(--acid));
        box-shadow: 0 0 14px rgba(22,191,167,0.20);
        animation: laneDance 1.05s ease-in-out infinite;
        animation-delay: calc(var(--i) * -0.08s);
    }

    .soft-card, div[data-testid="stExpander"], div[data-testid="stMetric"], div[data-testid="stFileUploader"] {
        border-radius: 20px !important;
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.12) !important;
        backdrop-filter: blur(18px);
    }

    .soft-card {
        padding: 1rem 1.08rem;
        color: var(--text);
    }

    .soft-card strong { color: var(--cyan); }

    .album-preview {
        position: relative;
        overflow: hidden;
        padding: 1.15rem;
        border-radius: 24px;
        background:
            linear-gradient(135deg, var(--panel), var(--panel-soft)),
            radial-gradient(circle at 18% 16%, rgba(22,191,167,0.13), transparent 34%),
            radial-gradient(circle at 88% 20%, rgba(244,166,42,0.13), transparent 30%);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
    }

    .album-preview-grid {
        display: grid;
        grid-template-columns: 112px minmax(0, 1fr);
        gap: 1rem;
        align-items: center;
    }

    .mini-cover {
        position: relative;
        overflow: hidden;
        aspect-ratio: 1;
        border-radius: 18px;
        background:
            linear-gradient(90deg, var(--grid) 1px, transparent 1px),
            linear-gradient(var(--grid) 1px, transparent 1px),
            linear-gradient(135deg, rgba(22,191,167,0.20), rgba(244,166,42,0.10));
        background-size: 22px 22px, 22px 22px, 100% 100%;
        border: 1px solid var(--line);
        box-shadow: inset 0 0 30px rgba(22,191,167,0.07), 0 18px 42px rgba(0,0,0,0.10);
    }

    .mini-cover::before {
        content: "";
        position: absolute;
        inset: 18% 16%;
        background:
            linear-gradient(to right, transparent 0 10%, var(--cyan) 10% 18%, transparent 18% 34%, var(--acid) 34% 42%, transparent 42% 58%, var(--amber) 58% 66%, transparent 66% 82%, var(--blue) 82% 90%, transparent 90%);
        opacity: 0.78;
        filter: drop-shadow(0 0 10px rgba(22,191,167,0.26));
    }

    .mini-cover::after {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 46%;
        height: 2px;
        background: var(--acid);
        box-shadow: 0 0 14px var(--acid);
    }

    .album-title {
        margin: 0;
        font-size: 1.66rem;
        line-height: 1;
        color: var(--text);
    }

    .album-meta {
        color: var(--muted);
        margin-top: 0.35rem;
        font-size: 0.92rem;
    }

    .album-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.85rem;
    }

    .studio-strip {
        min-height: 88px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }

    .studio-strip-title {
        margin: 0;
        font-size: 1.28rem;
        color: var(--text);
        text-transform: uppercase;
    }

    .studio-strip-copy {
        margin-top: 0.25rem;
        color: var(--muted);
        font-size: 0.9rem;
    }

    .studio-strip-orb {
        width: 68px;
        min-width: 68px;
        aspect-ratio: 1;
        border-radius: 16px;
        background:
            linear-gradient(90deg, var(--line-strong) 1px, transparent 1px),
            linear-gradient(var(--line-strong) 1px, transparent 1px),
            linear-gradient(135deg, rgba(22,191,167,0.16), rgba(143,207,35,0.14));
        background-size: 12px 12px, 12px 12px, 100% 100%;
        border: 1px solid var(--line);
        box-shadow: 0 0 28px rgba(22,191,167,0.14);
        animation: gridPulse 4.8s ease-in-out infinite;
    }

    .option-card-title {
        margin: 0 0 0.7rem 0;
        font-size: 0.82rem;
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: 0.13em;
    }

    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stFileUploader"]),
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stExpander"]) {
        animation: panelRise 0.65s ease both;
    }

    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(143,207,35,0.48) !important;
        border-radius: 14px !important;
        min-height: 3.12rem;
        color: var(--button-text) !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em;
        background: linear-gradient(90deg, var(--cyan), var(--acid), var(--amber)) !important;
        box-shadow: 0 18px 44px rgba(22,191,167,0.18), inset 0 1px 0 rgba(255,255,255,0.36) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        filter: saturate(1.10) brightness(1.04);
        box-shadow: 0 24px 58px rgba(143,207,35,0.20), 0 0 0 5px rgba(22,191,167,0.10) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea,
    div[data-baseweb="base-input"] {
        border-radius: 12px !important;
        background: var(--input-bg) !important;
        border-color: var(--line) !important;
        color: var(--text) !important;
        box-shadow: inset 0 0 18px rgba(22,191,167,0.04);
    }

    input, textarea, div[data-baseweb="select"] span {
        color: var(--text) !important;
    }

    div[data-testid="stMetric"] {
        padding: 0.85rem 1rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--acid);
        text-shadow: 0 0 16px rgba(143,207,35,0.16);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    @media (max-width: 980px) {
        .album-hero {
            grid-template-columns: 1fr;
            padding: 1.25rem;
        }
        .record-disc {
            right: 50%;
            transform: translateX(50%);
            opacity: 0.38;
        }
        .album-preview-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """.replace("__THEME_CSS_VARS__", theme_css_vars),
    unsafe_allow_html=True,
)

hero_secondary = (
    "把音频分析、键型约束和 SR 校准集中到一个作图控制台。"
    if lang == "zh"
    else "Centralize audio analysis, style constraints, and SR calibration in one charting deck."
)
hero_tags = ["BPM GRID", "SR TARGET", "MUSIC FIT", "OSZ EXPORT"] if lang == "en" else ["BPM 网格", "星数校准", "音乐贴合", "OSZ 导出"]
hero_tag_html = "".join(f"<span class='studio-pill'>{esc(tag)}</span>" for tag in hero_tags)
hero_bars = "".join(f"<span style='--i:{idx}'></span>" for idx in range(14))

st.markdown(
    f"""
    <section class="album-hero">
        <div>
            <div class="studio-eyebrow">RHYTHM CONTROL DECK</div>
            <div class="studio-title">{esc(tr('app_title'))}</div>
            <div class="studio-copy">{esc(tr('app_caption'))}<br>{esc(hero_secondary)}</div>
            <div class="studio-actions">{hero_tag_html}</div>
        </div>
        <div class="cover-stage">
            <div class="record-disc"></div>
            <div class="cover-card">
                <div class="cover-inner">
                    <div class="cover-kicker">LIVE GRID</div>
                    <div class="cover-title">Map<br>Engine</div>
                    <div class="wave-bars">{hero_bars}</div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

strip_title = "Control Console" if lang == "en" else "生成控制台"
strip_copy = (
    "Colors now follow Streamlit's active theme, so native widgets and the custom shell stay readable together."
    if lang == "en"
    else "界面颜色现在跟随 Streamlit 当前主题，原生控件和外层视觉会保持一致。"
)
st.markdown(
    f"""
    <div class="soft-card studio-strip">
        <div>
            <p class="studio-strip-title">{esc(strip_title)}</p>
            <div class="studio-strip-copy">{esc(strip_copy)}</div>
        </div>
        <div class="studio-strip-orb"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, style_col, tuning_col = st.columns([1.0, 1.05, 1.0], gap="large")

with input_col:
    section_label = "01 INPUT" if lang == "en" else "01 输入"
    st.markdown(f"<div class='option-card-title'>{esc(section_label)}</div>", unsafe_allow_html=True)
    st.subheader(tr("upload_header"))
    audio_file = st.file_uploader(tr("upload_audio"), type=["mp3", "wav", "ogg"])
    bg_file = st.file_uploader(tr("upload_bg"), type=["png", "jpg", "jpeg"])

    with st.expander(tr("metadata_header"), expanded=True):
        title = st.text_input(tr("title"), "生成的曲目" if lang == "zh" else "Generated Track")
        artist = st.text_input(tr("artist"), "未知艺术家" if lang == "zh" else "Unknown Artist")
        creator = st.text_input(tr("creator"), "AutoGenerator")
        difficulty_name = st.text_input(tr("difficulty_name"), "默认难度" if lang == "zh" else "Normal")

with style_col:
    section_label = "02 PATTERN" if lang == "en" else "02 键型"
    st.markdown(f"<div class='option-card-title'>{esc(section_label)}</div>", unsafe_allow_html=True)
    st.subheader(tr("chart_header"))
    chart_type = st.selectbox(
        tr("chart_type"),
        ["rice", "ln", "hybrid"],
        format_func=lambda value: tr(CHART_LABELS[value]),
        key="chart_type_selector",
    )

    hybrid_preset = "balanced_pp"
    ln_tendency = "auto"
    hybrid_weights = DEFAULT_HYBRID_WEIGHTS.copy()
    key_style = None
    custom_hybrid_weights = False
    custom_hybrid_ln = False

    if chart_type == "hybrid":
        st.caption(tr("hybrid_caption"))
        hybrid_preset = st.selectbox(
            tr("hybrid_preset"),
            list(HYBRID_PRESETS.keys()),
            index=list(HYBRID_PRESETS.keys()).index("balanced_pp"),
            format_func=lambda value: tr(PRESET_LABELS[value]),
        )
        ln_tendency = st.selectbox(
            tr("ln_tendency"),
            list(HYBRID_LN_TENDENCY_RATIO.keys()),
            index=list(HYBRID_LN_TENDENCY_RATIO.keys()).index("auto"),
            format_func=lambda value: tr(LN_TENDENCY_LABELS[value]),
        )
        hybrid_weights = hybrid_weights_for_preset(hybrid_preset)
        with st.expander(tr("hybrid_advanced"), expanded=False):
            custom_hybrid_weights = st.checkbox(tr("use_custom_hybrid"), False)
            if custom_hybrid_weights:
                hybrid_weights = {
                    "jack": st.slider(tr("style_jack"), 0.0, 1.0, hybrid_weights["jack"], 0.05),
                    "stream": st.slider(tr("style_stream"), 0.0, 1.0, hybrid_weights["stream"], 0.05),
                    "tech": st.slider(tr("style_tech"), 0.0, 1.0, hybrid_weights["tech"], 0.05),
                    "speed": st.slider(tr("style_speed"), 0.0, 1.0, hybrid_weights["speed"], 0.05),
                }
            custom_hybrid_ln = st.checkbox(tr("use_custom_ln"), False)
    else:
        key_style = st.selectbox(
            tr("key_style"),
            ["jack", "stream", "tech", "speed"],
            format_func=lambda value: tr(STYLE_LABELS[value]),
        )

    normalized = normalize_hybrid_weights(hybrid_weights)
    if chart_type == "hybrid":
        st.caption(
            f"{tr('normalized_weights')}: "
            f"Jack {normalized['jack']:.0%} / Stream {normalized['stream']:.0%} / "
            f"Tech {normalized['tech']:.0%} / Speed {normalized['speed']:.0%}"
        )

    description = style_description(lang, chart_type, key_style)
    if description:
        st.markdown(
            f"<div class='soft-card'><strong>{esc(tr('style_guide'))}</strong><br>{esc(description)}</div>",
            unsafe_allow_html=True,
        )

    st.subheader(tr("chord_header"))
    chord_enabled = chord_enabled_for(chart_type, key_style, hybrid_weights)
    min_chord, max_chord, default_chord = max_chord_bounds_for(chart_type, key_style, hybrid_weights)
    max_chord_size = st.slider(tr("max_chord"), min_chord, max_chord, default_chord)
    st.caption(tr("chord_caption"))

    ln_ratio = 0.0
    min_ln_ms = 120
    max_ln_ms = 1000
    if chart_type in ["ln", "hybrid"]:
        st.subheader("LN")
        if chart_type == "hybrid":
            ln_ratio = hybrid_ln_ratio_for_tendency(ln_tendency, hybrid_preset)
            if custom_hybrid_ln:
                ln_ratio = st.slider(tr("ln_ratio"), 0.0, 1.0, ln_ratio, 0.05, help=tr("ln_ratio_help"))
            else:
                st.caption(f"{tr('ln_ratio')}: {ln_ratio:.2f}")
        else:
            ln_ratio = st.slider(tr("ln_ratio"), 0.0, 1.0, 0.45, 0.05, help=tr("ln_ratio_help"))
        with st.expander(tr("ln_lengths"), expanded=False):
            ln_a, ln_b = st.columns(2)
            with ln_a:
                min_ln_ms = st.number_input(tr("min_ln"), 30, 500, 120, step=10)
            with ln_b:
                max_ln_ms = st.number_input(tr("max_ln"), 120, 3000, 1000, step=50)
            if max_ln_ms < min_ln_ms:
                st.warning("Max LN must be greater than or equal to Min LN." if lang == "en" else "最长 LN 必须大于等于最短 LN。")
                max_ln_ms = min_ln_ms

with tuning_col:
    section_label = "03 CALIBRATE" if lang == "en" else "03 校准"
    st.markdown(f"<div class='option-card-title'>{esc(section_label)}</div>", unsafe_allow_html=True)
    st.subheader(tr("difficulty_header"))
    target_star = st.number_input(tr("target_sr"), 0.0, 15.0, 3.5, step=0.5)
    sr_tolerance = st.number_input(tr("sr_tolerance"), 0.05, 0.5, 0.15, step=0.01)
    pattern_temperature = st.slider(tr("temperature"), 0.0, 1.0, 0.35, 0.05, help=tr("temperature_help"))
    music_influence = st.slider(tr("music_influence"), 0.0, 1.0, 0.65, 0.05, help=tr("music_influence_help"))

    st.subheader(tr("timing_header"))
    manual_bpm = st.number_input(tr("manual_bpm"), 0.0, step=1.0)
    timing_a, timing_b = st.columns([0.48, 0.52])
    with timing_a:
        use_manual_offset = st.checkbox(tr("manual_offset_toggle"), False)
    with timing_b:
        manual_offset = st.number_input(tr("manual_offset"), value=0, step=1, disabled=not use_manual_offset)
    st.caption(tr("offset_note"))

    preview_bpm = manual_bpm if manual_bpm > 0 else 220.0
    default_subdivisions = recommended_subdivisions(preview_bpm, chart_type, key_style, target_star if target_star > 0 else None)
    if chart_type == "hybrid":
        subdivision_options = HYBRID_SUBDIVISION_OPTIONS
    else:
        subdivision_options = SUBDIVISION_OPTIONS
    allowed_subdivisions = preserve_allowed_subdivisions(
        st.multiselect(
            tr("allowed_subdivisions"),
            subdivision_options,
            default=default_subdivisions,
            key=f"allowed_subdivisions_{chart_type}_{key_style or hybrid_preset}_{int(preview_bpm)}_{target_star}",
            help=tr("subdivision_help"),
        )
    )

preview_style = tr(PRESET_LABELS[hybrid_preset]) if chart_type == "hybrid" else tr(STYLE_LABELS[key_style])
preview_audio = audio_file.name if audio_file else ("等待音频输入" if lang == "zh" else "Awaiting audio input")
preview_target = f"{target_star:.1f} SR +/- {sr_tolerance:.2f}" if target_star > 0 else tr("target_unlimited")
preview_tags = [
    tr(CHART_LABELS[chart_type]),
    preview_style,
    f"Chord x{max_chord_size}",
    f"Music {music_influence:.0%}",
]
preview_tag_html = "".join(f"<span>{esc(tag)}</span>" for tag in preview_tags)
preview_heading = "当前工程快照" if lang == "zh" else "Current Project Snapshot"
st.markdown(
    f"""
    <div class="album-preview">
        <div class="album-preview-grid">
            <div class="mini-cover"></div>
            <div>
                <div class="cover-kicker">{esc(preview_heading)}</div>
                <h3 class="album-title">{esc(title)}</h3>
                <div class="album-meta">{esc(artist)} / {esc(difficulty_name)}</div>
                <div class="album-meta">{esc(preview_audio)} / {esc(preview_target)}</div>
                <div class="album-tags">{preview_tag_html}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

config = DifficultyConfig(
    version=difficulty_name,
    target_star=target_star if target_star > 0 else None,
    target_msd=None,
    chart_type=chart_type,  # type: ignore[arg-type]
    key_style=key_style,  # type: ignore[arg-type]
    allowed_subdivisions=allowed_subdivisions,
    chord_enabled=chord_enabled,
    max_chord_size=max_chord_size,
    chord_probability=0.35,
    max_jack_length=4,
    max_anchor_length=4,
    hand_balance=0.5,
    ln_ratio=ln_ratio,
    min_ln_ms=int(min_ln_ms),
    max_ln_ms=int(max_ln_ms),
    hybrid_weights=hybrid_weights,
    pattern_temperature=pattern_temperature,
    music_influence=music_influence,
)

with st.expander(tr("settings_preview"), expanded=False):
    active_style = hybrid_preset if chart_type == "hybrid" else key_style or chart_type
    target_text = f"{config.target_star:.2f} +/- {sr_tolerance:.2f}" if config.target_star is not None else tr("target_unlimited")
    st.write(
        {
            "chart_type": chart_type,
            "style": active_style,
            "target_sr": target_text,
            "temperature": pattern_temperature,
            "music_influence": music_influence,
            "max_chord_size": max_chord_size,
            "ln_ratio": ln_ratio,
            "subdivisions": allowed_subdivisions,
            "hybrid_weights": normalize_hybrid_weights(hybrid_weights) if chart_type == "hybrid" else None,
        }
    )

if st.button(tr("generate"), type="primary", use_container_width=True):
    if not audio_file:
        st.error(tr("need_audio"))
    else:
        with st.spinner(tr("analyzing")):
            audio_bytes = audio_file.read()
            bg_bytes = bg_file.read() if bg_file else None

            analyzer = AudioAnalyzer(
                audio_bytes,
                manual_bpm if manual_bpm > 0 else None,
                manual_offset if use_manual_offset else None,
            )
            analysis = analyzer.analyze()
            analysis["offset_ms"] -= 20
            snapped = build_snap_candidates(analysis, config)

        detected_cols = st.columns(4)
        detected_cols[0].metric("BPM", f"{analysis['bpm']:.2f}")
        detected_cols[1].metric("Offset", f"{analysis['offset_ms']}ms")
        detected_cols[2].metric("Snaps", f"{len(snapped)}")
        detected_cols[3].metric("Subdivisions", ", ".join(allowed_subdivisions))

        with st.spinner(tr("generating")):
            best_notes, best_est_sr, target_met, attempts = generate_to_target_sr(config, analysis, snapped, tolerance=sr_tolerance)

            if config.target_star is not None and not target_met:
                st.error(f"{tr('target_failed')} Target: {config.target_star:.2f}, Actual: {best_est_sr:.2f}.")
                st.stop()

        summary = note_summary(best_notes)
        st.subheader(tr("final"))
        result_cols = st.columns(6)
        result_cols[0].metric("SR", f"{best_est_sr:.2f}")
        result_cols[1].metric(tr("target_met"), tr("yes") if target_met or config.target_star is None else tr("no"))
        result_cols[2].metric("Notes", f"{summary['notes']}")
        result_cols[3].metric("LN", f"{summary['ln']}")
        result_cols[4].metric("Rows", f"{summary['rows']}")
        result_cols[5].metric("Max Chord", f"{summary['max_chord']}")
        st.caption(f"Attempts: {attempts}")

        final_style = tr(PRESET_LABELS[hybrid_preset]) if chart_type == "hybrid" else tr(STYLE_LABELS[key_style])
        final_target = f"{config.target_star:.2f} +/- {sr_tolerance:.2f}" if config.target_star is not None else tr("target_unlimited")
        final_tags = [
            f"SR {best_est_sr:.2f}",
            final_style,
            f"{summary['rows']} rows",
            f"{summary['ln']} LN",
            f"{attempts} attempts",
        ]
        final_tag_html = "".join(f"<span>{esc(tag)}</span>" for tag in final_tags)
        final_heading = "EXPORT READY" if lang == "en" else "导出就绪"
        st.markdown(
            f"""
            <div class="album-preview">
                <div class="album-preview-grid">
                    <div class="mini-cover"></div>
                    <div>
                        <div class="cover-kicker">{esc(final_heading)}</div>
                        <h3 class="album-title">{esc(title)}</h3>
                        <div class="album-meta">{esc(artist)} / {esc(config.version)}</div>
                        <div class="album-meta">Target: {esc(final_target)} / BPM {analysis['bpm']:.2f} / Offset {analysis['offset_ms']}ms</div>
                        <div class="album-tags">{final_tag_html}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner(tr("packaging")):
            osu_str = OsuExporter.export(
                config,
                best_notes,
                analysis["bpm"],
                analysis["offset_ms"],
                audio_file.name,
                bg_file.name if bg_file else None,
                artist,
                title,
                creator,
            )
            osu_filename = f"{artist} - {title} ({creator}) [{config.version}].osu"
            osz_bytes = Packager.package(
                audio_bytes,
                audio_file.name,
                bg_bytes,
                bg_file.name if bg_file else None,
                {osu_filename: osu_str},
            )

            st.download_button(
                label=tr("download"),
                data=osz_bytes,
                file_name=f"{artist} - {title}.osz",
                mime="application/zip",
                use_container_width=True,
            )


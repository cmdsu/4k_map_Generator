import streamlit as st

from om4k_generator.audio_analyzer import AudioAnalyzer
from om4k_generator.calibrator import build_snap_candidates, generate_to_target_sr
from om4k_generator.models import DifficultyConfig
from om4k_generator.osu_exporter import OsuExporter
from om4k_generator.packager import Packager
from om4k_generator.style_rules import (
    DEFAULT_HYBRID_WEIGHTS,
    chord_enabled_for,
    max_chord_bounds_for,
    preserve_allowed_subdivisions,
)


if "lang" not in st.session_state:
    st.session_state["lang"] = "中文"

lang_opts = ["中文", "English"]
selected_lang = st.sidebar.radio("Language / 语言", lang_opts, index=lang_opts.index(st.session_state["lang"]))
st.session_state["lang"] = selected_lang


def _t(zh: str, en: str) -> str:
    return zh if st.session_state["lang"] == "中文" else en


st.title(_t("osu!mania 4K 自动练习谱生成器", "osu!mania 4K Chart Generator"))
st.markdown(_t("规则生成 MVP：上传音频，配置目标，导出可导入 osu! 的 .osz。", "Rule-based MVP: upload audio, configure the target, and export an osu!-importable .osz."))

audio_file = st.file_uploader(_t("上传音频", "Upload Audio"), type=["mp3", "wav", "ogg"])
bg_file = st.file_uploader(_t("上传背景图（可选）", "Upload Background (Optional)"), type=["png", "jpg", "jpeg"])

with st.expander(_t("元数据", "Metadata")):
    title = st.text_input(_t("标题", "Title"), _t("生成的曲目", "Generated Track"))
    artist = st.text_input(_t("艺术家", "Artist"), _t("未知艺术家", "Unknown Artist"))
    creator = st.text_input(_t("谱师", "Creator"), "AutoGenerator")

st.header(_t("难度与谱面设置", "Difficulty & Chart Settings"))

target_star = st.number_input(_t("目标官方 SR（0 表示不限制）", "Target Official SR (0 for unconstrained)"), 0.0, 15.0, 3.5, step=0.5)
chart_type = st.selectbox(_t("谱面类型", "Chart Type"), ["rice", "ln", "hybrid", "vibro"])

hybrid_weights = DEFAULT_HYBRID_WEIGHTS.copy()
key_style = None

if chart_type == "hybrid":
    st.subheader(_t("Hybrid 键型比例", "Hybrid Style Weights"))
    hybrid_weights = {
        "jack": st.slider("Jack", 0.0, 1.0, DEFAULT_HYBRID_WEIGHTS["jack"], 0.05),
        "stream": st.slider("Stream", 0.0, 1.0, DEFAULT_HYBRID_WEIGHTS["stream"], 0.05),
        "tech": st.slider("Tech", 0.0, 1.0, DEFAULT_HYBRID_WEIGHTS["tech"], 0.05),
        "speed": st.slider("Speed", 0.0, 1.0, DEFAULT_HYBRID_WEIGHTS["speed"], 0.05),
    }
elif chart_type == "vibro":
    st.caption(_t("Vibro 本身就是独立风格，不需要额外选择键型。", "Vibro is its own style, so no extra key style is needed."))
else:
    key_style = st.selectbox(_t("键型风格", "Key Style"), ["jack", "stream", "tech", "speed"])

ln_ratio = 0.0
if chart_type in ["ln", "hybrid"]:
    ln_ratio = st.slider(_t("LN（长条）比例", "LN Ratio"), 0.0, 1.0, 0.1)

subdivision_options = ["1/1", "1/2", "1/3", "1/4", "1/5", "1/6", "1/7", "1/8", "1/12", "1/16"]
allowed_subdivisions = preserve_allowed_subdivisions(
    st.multiselect(
        _t("允许分辨率", "Allowed Subdivisions"),
        subdivision_options,
        default=["1/2", "1/4", "1/8"],
    )
)
if target_star == 0:
    st.caption(_t("目标 SR 为 0 时，会根据 BPM 与键型自动偏向合适分辨率。", "When target SR is 0, subdivisions are biased by BPM and style."))

chord_enabled = chord_enabled_for(chart_type, key_style, hybrid_weights)
min_chord, max_chord, default_chord = max_chord_bounds_for(chart_type, key_style, hybrid_weights)

if chart_type == "vibro":
    max_chord_size = st.slider(_t("最大同时按键数", "Max Chord Size"), min_chord, max_chord, default_chord)
elif key_style in ["jack", "stream"] or chart_type == "hybrid":
    if key_style in ["jack", "stream"]:
        st.caption(_t("Jack / Stream 默认包含双押或多押，因此无需额外开关。", "Jack / Stream include chords by default, so no extra toggle is needed."))
    max_chord_size = st.slider(_t("最大同时按键数", "Max Chord Size"), min_chord, max_chord, default_chord)
else:
    chord_enabled = st.checkbox(_t("允许双押/多押（Chord）", "Enable Chords"), False)
    max_chord_size = st.slider(_t("最大同时按键数", "Max Chord Size"), min_chord, max_chord, default_chord)

config = DifficultyConfig(
    version=_t("默认难度", "Normal"),
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
    min_ln_ms=120,
    max_ln_ms=1000,
    hybrid_weights=hybrid_weights,
    vibro_options={"lanes": [1, 2]},
)

config.version = st.text_input(_t("难度名称", "Difficulty Name"), _t("默认难度", "Normal"))

manual_bpm = st.number_input(_t("手动设定 BPM（0 为自动检测）", "Manual BPM (0 for Auto)"), 0.0, step=1.0)
use_manual_offset = st.checkbox(_t("使用手动 Offset", "Use Manual Offset"), False)
manual_offset = st.number_input(_t("手动设定 Offset 偏移（ms）", "Manual Offset (ms)"), 0, step=1)

if st.button(_t("生成 .osz 练习谱", "Generate .osz")):
    if not audio_file:
        st.error(_t("请先上传一个音频文件。", "Please upload an audio file first."))
    else:
        with st.spinner(_t("正在分析音频...", "Analyzing audio...")):
            audio_bytes = audio_file.read()
            bg_bytes = bg_file.read() if bg_file else None

            analyzer = AudioAnalyzer(
                audio_bytes,
                manual_bpm if manual_bpm > 0 else None,
                manual_offset if use_manual_offset else None,
            )
            analysis = analyzer.analyze()

            st.success(_t(
                f"检测到 BPM: {analysis['bpm']:.2f}, Offset: {analysis['offset_ms']}ms",
                f"Detected BPM: {analysis['bpm']:.2f}, Offset: {analysis['offset_ms']}ms",
            ))
            snapped = build_snap_candidates(analysis, config)

        with st.spinner(_t("正在生成并校准官方 SR...", "Generating and calibrating official SR...")):
            best_notes, best_est_sr, target_met, attempts = generate_to_target_sr(config, analysis, snapped)

            if config.target_star is not None and not target_met:
                st.error(_t(
                    f"未能严格达到目标 SR。目标: {config.target_star:.2f} 星，实际: {best_est_sr:.2f} 星。请增加分辨率/Chord/最大同押或降低目标。",
                    f"Could not strictly reach target SR. Target: {config.target_star:.2f} stars, actual: {best_est_sr:.2f} stars. Increase subdivisions/chords/max chord size or lower the target.",
                ))
                st.stop()

            st.info(_t(f"最终官方 SR: {best_est_sr} 星（尝试 {attempts} 次）", f"Final Official SR: {best_est_sr} stars ({attempts} attempts)"))

        with st.spinner(_t("正在打包输出...", "Packaging...")):
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
                label=_t("下载 .osz 打包文件", "Download .osz"),
                data=osz_bytes,
                file_name=f"{artist} - {title}.osz",
                mime="application/zip",
            )

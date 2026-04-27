# osu!mania 4K 自动练习谱生成器

> [English](./docs/README_EN.md) 

一个基于规则的 osu!mania 4K 练习谱生成器。项目目标是根据音频、目标 SR、键型风格和 LN 设置，生成可导入 osu! 的 `.osu` / `.osz` 谱面。

## 功能概览

- 支持 Web UI：通过 Streamlit 上传音频、调整参数、生成并下载 `.osz`。
- 支持 CLI：用于批量测试、调试谱面生成结果。
- 支持谱面类型：`rice`、`ln`、`hybrid`。
- 支持键型风格：`jack`、`stream`、`tech`、`speed`。
- 支持目标 SR 校准、SR 容差、pattern temperature、音乐贴合度、BPM / offset 手动覆盖。
- 导出 osu!mania 4K `.osu`，并可打包为 `.osz`。
- 生成后会进行合法性过滤，减少重复 note、LN 冲突和不合理的同轨尾接头。

## 目录说明

```text
.
├── app.py                         # Streamlit Web UI
├── cli.py                         # 命令行调试入口
├── requirements.txt               # Python 依赖
├── om4k_generator/                # 核心生成模块
│   ├── audio_analyzer.py          # 音频分析：BPM、offset、onset、能量、静音区
│   ├── calibrator.py              # SR 校准与主要键型生成逻辑
│   ├── difficulty_estimator.py    # 谱面难度估计
│   ├── grid_builder.py            # 节拍网格 / snap 构造
│   ├── models.py                  # 配置与 Note 数据结构
│   ├── osu_exporter.py            # .osu 文本导出
│   ├── packager.py                # .osz 打包
│   ├── pattern_generator.py       # 通用 pattern 辅助逻辑
│   ├── style_rules.py             # 风格默认值、同押限制、分辨率推荐
│   └── validator.py               # 谱面合法性过滤
├── docs/                          # 项目需求与说明文档
├── in/                            # 本地输入音频目录，不提交
├── out/                           # 本地 CLI 输出目录，不提交
├── std/                           # 本地标准谱面参考目录，不提交
└── logs/                          # 本地运行日志目录，不提交
```

`in/`、`out/`、`std/`、`logs/` 已加入 `.gitignore`。这些目录用于本地开发和测试，不会提交到 GitHub。

## 环境要求

推荐：

- Python 3.10+
- pip 或 conda
- 可访问本地浏览器的运行环境

安装依赖：

```powershell
pip install -r requirements.txt
```

如果使用 conda：

```powershell
conda create -n om4k python=3.10
conda activate om4k
pip install -r requirements.txt
```

## 本地部署与使用

### 1. 克隆项目

```powershell
git clone https://github.com/cmdsu/4k_map_Generator.git

cd 4k_map_Generator
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 启动 Web UI

普通 Python 环境：

```powershell
streamlit run app.py
```

如果你使用本机 Anaconda Streamlit，可使用：

```powershell
C:/python/anaconda/Scripts/streamlit.exe run app.py
```

启动后浏览器会打开 Streamlit 页面。一般地址是：

```text
http://localhost:8501
```

### 4. Web UI 使用流程

1. 上传音频文件，支持 `mp3`、`wav`、`ogg`。
2. 填写标题、艺术家、制作者、难度名。
3. 在 `02 键型` 区域选择谱面类型与键型风格。
4. 在 `03 校准` 区域设置目标 SR、SR 容差、分辨率、BPM / offset 等参数。
5. 点击 `生成 .osz`。
6. 下载生成的 `.osz`，导入 osu! 测试。

## CLI 调试用法

CLI 适合快速生成 `.osu` 文件并检查谱面文本。

先把音频放到本地 `in/` 目录，例如：

```text
in/audio.mp3
```

然后运行：

```powershell
python cli.py --audio audio.mp3 --chart_type rice --key_style jack --target_sr 4.0
```

输出文件会写入 `out/` 目录。

常用参数：

```text
--audio              in/ 下的音频文件名，必填
--target_sr          目标 SR，0 表示不限制
--sr_tolerance       SR 允许波动范围，默认 0.15
--chart_type         rice / ln / hybrid
--key_style          jack / stream / tech / speed，hybrid 时不使用固定 key_style
--bpm                手动 BPM，0 表示自动检测
--offset             手动 offset，单位 ms
--subdivisions       分辨率，例如 1/2,1/4,1/8；auto 表示自动推荐
--max_chord_size     最大同押数
--temperature        pattern 变化程度，0.0 到 1.0
--music_influence    音乐贴合度，0.0 到 1.0
--ln_ratio           LN 比例 / 数值
--hybrid_preset      hybrid 方向预设
--ln_tendency        hybrid LN 倾向
--title              导出标题
--artist             导出艺术家
--version            导出难度名
```

示例：

```powershell
python cli.py --audio audio.mp3 --chart_type rice --key_style stream --target_sr 5.5 --sr_tolerance 0.15 --subdivisions 1/4,1/6,1/8
```

```powershell
python cli.py --audio audio.mp3 --chart_type ln --key_style speed --target_sr 4.5 --ln_ratio 0.45
```

```powershell
python cli.py --audio audio.mp3 --chart_type hybrid --target_sr 6.0 --hybrid_preset balanced_pp --ln_tendency auto
```

## 服务器部署

### 方式一：直接运行 Streamlit

适合个人服务器、内网服务器或测试环境。

```bash
git clone https://github.com/cmdsu/4k_map_Generator.git
cd 4k_map_Generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p in out std logs
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

访问：

```text
http://服务器IP:8501
```

### 方式二：后台运行

```bash
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > logs/streamlit.out.log 2> logs/streamlit.err.log &
```

查看日志：

```bash
tail -f logs/streamlit.out.log
tail -f logs/streamlit.err.log
```

### 方式三：systemd 服务

创建服务文件：

```bash
sudo nano /etc/systemd/system/om4k-generator.service
```

示例内容：

```ini
[Unit]
Description=osu!mania 4K Chart Generator
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/4k_map_Generator
ExecStart=/path/to/4k_map_Generator/.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable om4k-generator
sudo systemctl start om4k-generator
sudo systemctl status om4k-generator
```

### 反向代理建议

如果要绑定域名，建议使用 Nginx / Caddy 做反向代理，并为域名配置 HTTPS。

Nginx 示例：

```nginx
server {
    listen 80;
    server_name your-domain.example;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 核心组件说明

### `app.py`

Streamlit 前端入口。负责：

- 上传音频和背景图
- 参数配置
- 调用音频分析、生成、校准、打包流程
- 下载 `.osz`

### `cli.py`

命令行调试入口。负责：

- 从 `in/` 读取音频
- 用命令行参数控制生成配置
- 输出 `.osu` 到 `out/`

### `audio_analyzer.py`

音频分析模块。负责：

- 自动检测 BPM
- 估计 offset
- 提取 onset
- 计算能量曲线
- 识别静音或低能量区域

### `calibrator.py`

当前最核心的生成与校准模块。负责：

- 构建 snap 候选
- 根据目标 SR 多轮尝试生成谱面
- 根据风格生成 jack / stream / tech / speed / hybrid 的 pattern
- 控制音乐贴合度、变化程度、同押倾向和密度

### `difficulty_estimator.py`

难度估计模块。负责给生成结果估计 SR，用于校准目标难度。

### `grid_builder.py`

节拍网格辅助模块。负责把 BPM 和 offset 转换为可用 snap。

### `models.py`

数据模型。主要包含：

- `DifficultyConfig`
- `NoteObject`
- `AudioAnalysisResult`

### `osu_exporter.py`

`.osu` 导出模块。负责生成 osu!mania 4K 可读取的 v14 `.osu` 文本。

### `packager.py`

`.osz` 打包模块。负责把 `.osu`、音频和可选背景图打包成 ZIP 格式的 `.osz`。

### `pattern_generator.py`

通用 pattern 辅助模块。保留部分基础 pattern 生成、lane 分配与规则辅助逻辑。

### `style_rules.py`

风格规则配置模块。负责：

- 不同键型默认分辨率
- 最大同押范围
- Hybrid 权重预设
- LN 倾向

### `validator.py`

谱面合法性过滤模块。负责：

- 删除同时间同轨重复 note
- 避免 LN 同轨冲突
- 控制 LN 尾部附近接 note / LN 头的问题
- 限制过于不合理的物件

## 开发与测试建议

语法检查：

```powershell
python -m py_compile app.py cli.py om4k_generator/*.py
```

CLI smoke test：

```powershell
python cli.py --audio audio.mp3 --chart_type rice --key_style jack --target_sr 4.0 --sr_tolerance 0.15
```

生成质量测试建议：

- 使用 `in/audio.mp3` 做固定音频回归。
- 生成不同目标 SR，例如 3.0 到 7.0 每 0.5 星一档。
- 直接打开生成的 `.osu` 检查前几百行 HitObjects。
- 与本地 `std/` 中的标准谱面做人工对照。

## 注意事项

- 本项目是规则生成器，不使用训练模型。
- 生成结果需要人工测试手感，尤其是高星、LN、hybrid 和 tech。

# osu!mania 4K Auto Chart Generator

> [中文文档](./README_CN.md)

A rule-based osu!mania 4K practice chart generator. It analyzes audio, applies style-specific pattern rules, calibrates target SR, validates note legality, and exports osu!-importable `.osu` / `.osz` files.

## Features

- Web UI powered by Streamlit for uploading audio, tuning parameters, generating charts, and downloading `.osz` packages.
- CLI debugger for batch generation and local regression checks.
- Supported chart types: `rice`, `ln`, `hybrid`.
- Supported key styles: `jack`, `stream`, `tech`, `speed`.
- Target SR calibration with configurable tolerance.
- Pattern temperature and music-fit controls.
- Manual BPM / offset override.
- osu!mania 4K `.osu` export and `.osz` packaging.
- Validation pass for duplicate notes, LN conflicts, unsafe LN tail/head placements, and unreasonable objects.

## Project Structure

```text
.
|-- app.py                         # Streamlit Web UI
|-- cli.py                         # CLI debugger
|-- requirements.txt               # Python dependencies
|-- om4k_generator/                # Core generator package
|   |-- audio_analyzer.py          # Audio analysis: BPM, offset, onset, energy, silence
|   |-- calibrator.py              # SR calibration and main style generation logic
|   |-- difficulty_estimator.py    # Difficulty estimation
|   |-- grid_builder.py            # Beat grid / snap helpers
|   |-- models.py                  # Config and note data structures
|   |-- osu_exporter.py            # .osu text export
|   |-- packager.py                # .osz packaging
|   |-- pattern_generator.py       # Shared pattern helpers
|   |-- style_rules.py             # Style defaults, chord limits, subdivision recommendations
|   `-- validator.py               # Chart legality cleanup
|-- docs/                          # Requirement and design notes
|-- in/                            # Local input audio folder, ignored by Git
|-- out/                           # Local CLI output folder, ignored by Git
|-- std/                           # Local reference chart folder, ignored by Git
`-- logs/                          # Local runtime logs, ignored by Git
```

`in/`, `out/`, `std/`, and `logs/` are intentionally ignored by Git. They are for local development, testing, reference charts, and runtime logs.

## Requirements

Recommended environment:

- Python 3.10+
- pip or conda
- A local browser for Streamlit

Install dependencies:

```bash
pip install -r requirements.txt
```

With conda:

```bash
conda create -n om4k python=3.10
conda activate om4k
pip install -r requirements.txt
```

## Local Deployment and Usage

### 1. Clone the repository

```bash
git clone https://github.com/cmdsu/4k_map_Generator.git
cd 4k_map_Generator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Web UI

Common Python environment:

```bash
streamlit run app.py
```

If you use the local Anaconda Streamlit executable on Windows:

```powershell
C:/python/anaconda/Scripts/streamlit.exe run app.py
```

The default local address is usually:

```text
http://localhost:8501
```

### 4. Web UI workflow

1. Upload an audio file. Supported formats: `mp3`, `wav`, `ogg`.
2. Fill in title, artist, creator, and difficulty name.
3. Choose chart type and key style in the `02 Pattern` area.
4. Tune target SR, tolerance, subdivisions, BPM / offset, temperature, and music-fit controls in the `03 Calibrate` area.
5. Click `Generate .osz`.
6. Download the generated `.osz` and import it into osu! for testing.

## CLI Usage

The CLI is useful for quick `.osu` generation and direct text inspection.

Place audio files in the local `in/` folder first, for example:

```text
in/audio.mp3
```

Run:

```bash
python cli.py --audio audio.mp3 --chart_type rice --key_style jack --target_sr 4.0
```

The generated `.osu` file is written to `out/`.

Common options:

```text
--audio              Audio file name under in/, required
--target_sr          Target SR, use 0 for unconstrained
--sr_tolerance       Allowed SR tolerance, default 0.15
--chart_type         rice / ln / hybrid
--key_style          jack / stream / tech / speed; not used as a fixed style for hybrid
--bpm                Manual BPM, use 0 for auto detection
--offset             Manual offset in ms
--subdivisions       Subdivisions, such as 1/2,1/4,1/8; use auto for recommended values
--max_chord_size     Maximum simultaneous keys
--temperature        Pattern variation, from 0.0 to 1.0
--music_influence    Music-fit influence, from 0.0 to 1.0
--ln_ratio           LN ratio / value
--hybrid_preset      Hybrid direction preset
--ln_tendency        Hybrid LN tendency
--title              Export title
--artist             Export artist
--version            Export difficulty name
```

Examples:

```bash
python cli.py --audio audio.mp3 --chart_type rice --key_style stream --target_sr 5.5 --sr_tolerance 0.15 --subdivisions 1/4,1/6,1/8
```

```bash
python cli.py --audio audio.mp3 --chart_type ln --key_style speed --target_sr 4.5 --ln_ratio 0.45
```

```bash
python cli.py --audio audio.mp3 --chart_type hybrid --target_sr 6.0 --hybrid_preset balanced_pp --ln_tendency auto
```

## Server Deployment

### Option 1: Run Streamlit directly

Suitable for personal servers, LAN servers, or test environments.

```bash
git clone https://github.com/cmdsu/4k_map_Generator.git
cd 4k_map_Generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p in out std logs
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Visit:

```text
http://your-server-ip:8501
```

### Option 2: Run in the background

```bash
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > logs/streamlit.out.log 2> logs/streamlit.err.log &
```

View logs:

```bash
tail -f logs/streamlit.out.log
tail -f logs/streamlit.err.log
```

### Option 3: systemd service

Create a service file:

```bash
sudo nano /etc/systemd/system/om4k-generator.service
```

Example:

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

Start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable om4k-generator
sudo systemctl start om4k-generator
sudo systemctl status om4k-generator
```

### Reverse proxy recommendation

For domain-based access, run Streamlit behind Nginx or Caddy and enable HTTPS.

Nginx example:

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

## Core Components

### `app.py`

Streamlit Web UI. Handles audio/background upload, parameter configuration, generation, packaging, and `.osz` download.

### `cli.py`

Command-line debugger. Reads audio from `in/`, generates charts from CLI options, and writes `.osu` files to `out/`.

### `audio_analyzer.py`

Audio analysis module. Detects BPM, estimates offset, extracts onsets, computes energy curves, and identifies low-energy or silent regions.

### `calibrator.py`

The main generation and calibration module. It builds snap candidates, generates charts through multiple attempts, applies jack / stream / tech / speed / hybrid style logic, and controls music fit, variation, chord tendency, and density.

### `difficulty_estimator.py`

Estimates SR for generated charts and supports target difficulty calibration.

### `grid_builder.py`

Timing grid helper. Converts BPM and offset into usable snap positions.

### `models.py`

Data models, including `DifficultyConfig`, `NoteObject`, and `AudioAnalysisResult`.

### `osu_exporter.py`

Exports osu!mania 4K compatible v14 `.osu` text.

### `packager.py`

Packages `.osu`, audio, and optional background image into ZIP-format `.osz` files.

### `pattern_generator.py`

Shared pattern helper module for basic pattern generation, lane allocation, and rule utilities.

### `style_rules.py`

Style rule configuration, including default subdivisions, max chord ranges, hybrid presets, and LN tendencies.

### `validator.py`

Chart legality cleanup. Removes duplicate same-time same-lane notes, avoids same-lane LN conflicts, controls unsafe LN tail/head placements, and limits unreasonable objects.

## Development and Testing

Syntax check:

```bash
python -m py_compile app.py cli.py om4k_generator/*.py
```

CLI smoke test:

```bash
python cli.py --audio audio.mp3 --chart_type rice --key_style jack --target_sr 4.0 --sr_tolerance 0.15
```

Generation quality testing suggestions:

- Use `in/audio.mp3` as a fixed regression audio file.
- Test target SR from 3.0 to 7.0 in 0.5 increments.
- Open the generated `.osu` directly and inspect the first few hundred HitObjects.
- Compare manually with local reference charts under `std/`.

## Notes

- This project is a rule-based generator and does not use trained models.
- Generated charts still require human playtesting, especially for high-star, LN, hybrid, and tech outputs.
- `std/` is a local reference chart folder and is not committed to the remote repository.
- `in/`, `out/`, and `logs/` are local runtime folders and are not committed to the remote repository.
- If you need to remove large files from Git history, use a dedicated Git history cleanup workflow. `.gitignore` only affects future commits.

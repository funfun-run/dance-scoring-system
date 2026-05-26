# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Dance scoring system v2.0 — 基于嵌入式边缘计算的舞蹈分段跟练与姿态纠错系统. Target hardware: Intel DK-2500 (Core Ultra 5 225U + NPU), deployment OS: Ubuntu 22.04 + OpenVINO. Compares a user's dance video against a reference video using MediaPipe pose landmark detection, joint-angle analysis, DTW alignment, fastdtw, and segment-based scoring. Outputs a terminal score report, generates slow-motion practice clips, and pinpoints weak body parts for corrective feedback.

## Commands

```bash
# Offline scoring
python scripts/score.py -r <reference.mp4> -u <user.mp4>

# Custom score threshold (default 50)
python scripts/score.py -r <reference.mp4> -u <user.mp4> -t 60

# 8-beat video segmentation
python scripts/split.py -r <reference.mp4>

# Custom BPM
python scripts/split.py -r <reference.mp4> -b 100

# Launch GUI
python src/dance_scoring/gui/app.py

# Live camera practice (placeholder)
python scripts/run_live.py
```

## Dependencies

The project uses the `.venv` virtual environment. Install dependencies before running:

```bash
pip install -r requirements.txt
```

The MediaPipe pose landmarker model is auto-downloaded on first run (~5.6 MB) to `~/.cache/dance_scoring/`. No manual model setup is needed.

Note: The `.vscode/settings.json` references a `dance_env` interpreter path which doesn't exist — use `.venv` instead.

## Architecture

Target hardware: Intel DK-2500 (Core Ultra 5 225U), Ubuntu 22.04, OpenVINO NPU, HDMI external display.

Package layout under `src/dance_scoring/`:

| Layer | Directory | Modules | Status |
|-------|-----------|---------|--------|
| AI reasoning | `core/` | `config.py`, `frame.py`, `extractor.py`, `dtw.py`, `alignment.py`, `scorer.py`, `segments.py`, `inference.py`, `correction.py` | Active / Placeholder |
| Data processing | `video/` | `info.py`, `beat_detector.py`, `splitter.py`, `merger.py` | Active |
| Perception | `camera/` | `base.py`, `usb.py`, `stream.py` | Active |
| Interaction | `gui/` | `app.py`, `components.py`, `worker.py` | Active |
| Hardware (DK-2500) | `platform/` | `npu.py`, `gpio.py` | Placeholder |
| Data transfer | `transfer/` | `base.py`, `wifi.py`, `bluetooth.py` | Placeholder |
| ROS2 (optional) | `ros2/` | `nodes/`, `interfaces/`, `launch/` | Placeholder |

**CLI entry points**: `scripts/score.py` (offline scoring), `scripts/split.py` (video segmentation), `scripts/run_live.py` (camera practice, placeholder)

**Key constants** (from commit `d3e03e9`): `BEATS_PER_SEGMENT=8`, `SLOW_SPEED=0.8`, `TARGET_FPS=30`, `PASS_SCORE=60.0`.

**Input/Output conventions:**
- Videos passed as CLI arguments (`-r`, `-u`), no fixed input directory
- Model auto-downloaded to `~/.cache/dance_scoring/`
- Segment clips output to `output/segments/`
- Low-score practice clips output to `output/low_score_clips/`

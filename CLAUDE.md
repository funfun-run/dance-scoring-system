# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Dance scoring system v1.0 (舞蹈评分系统) that compares a user's dance video against a reference video using MediaPipe pose landmark detection, joint-angle analysis, DTW (Dynamic Time Warping) alignment, and segment-based scoring. The system outputs a terminal score report and generates slow-motion practice clips for segments scored below a configurable threshold.

## Commands

```bash
# Run the main dance scoring system
python scripts/score.py -r assets/videos/reference.mp4 -u assets/videos/user.mp4

# Run with a custom score threshold (default 50)
python scripts/score.py -r assets/videos/reference.mp4 -u assets/videos/user.mp4 -t 60

# Split reference video into 8-beat slow-motion segments
python scripts/split.py -r assets/videos/reference.mp4

# Split with custom BPM
python scripts/split.py -r assets/videos/reference.mp4 -b 100

# Launch GUI
python src/dance_scoring/gui/app.py
```

## Dependencies

The project uses the `.venv` virtual environment. Install dependencies before running:

```bash
pip install -r requirements.txt
```

The MediaPipe pose landmarker model is auto-downloaded on first run (~5.6 MB) to `assets/model/`. No manual model setup is needed.

Note: The `.vscode/settings.json` references a `dance_env` interpreter path which doesn't exist — use `.venv` instead.

## Architecture

Package layout under `src/dance_scoring/`:

- **`core/`** — Scoring engine: `config.py` (constants), `frame.py` (PoseFrame dataclass), `extractor.py` (MediaPipe PoseLandmarker wrapper), `dtw.py` (DTW alignment), `scorer.py` (scoring pipeline), `segments.py` (segment extraction)
- **`video/`** — Video tools: `info.py` (video metadata), `beat_detector.py` (audio/motion beat detection), `splitter.py` (8-beat segmentation), `merger.py` (clip merging)
- **`camera/`** — Camera abstraction: `base.py`, `stream.py`, `usb.py`
- **`gui/`** — Tkinter GUI: `app.py`, `components.py`, `worker.py`
- **`hardware/`** — Hardware integration: `display.py`, `gpio.py`
- **`transfer/`** — Data transfer: `base.py`, `bluetooth.py`, `wifi.py`

**CLI entry points**: `scripts/score.py` (scoring), `scripts/split.py` (video segmentation)

**Key constants**: `BEATS_PER_SEGMENT=8`, `SLOW_SPEED=0.8`, `TARGET_FPS=30`, `PASS_SCORE=60.0`.

**Input/Output conventions:**
- Reference and user videos go in `assets/videos/`
- Segment clips output to `output/segments/`
- Low-score practice clips output to `output/low_score_clips/`

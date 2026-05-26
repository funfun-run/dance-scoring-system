# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Dance scoring system (舞蹈评分系统) that compares a user's dance video against a reference video using MediaPipe pose landmark detection, joint-angle analysis, DTW (Dynamic Time Warping) alignment, and segment-based scoring. The system outputs a terminal score report and generates slow-motion practice clips for segments scored below a configurable threshold.

## Commands

```bash
# Run the main dance scoring system
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4

# Run with a custom score threshold (default 50)
python score_dance.py -r videos/reference.mp4 -u videos/user.mp4 -t 60

# Split reference video into 8-beat slow-motion segments
python split_8beats.py -r videos/reference.mp4

# Split with custom BPM
python split_8beats.py -r videos/reference.mp4 -b 100

# Verify MediaPipe installation
python check_env.py
```

## Dependencies

The project uses the `.venv` virtual environment. Install dependencies before running:

```bash
.venv/Scripts/pip install mediapipe scipy numpy
```

The MediaPipe pose landmarker model (`pose_landmarker_lite.task`) is auto-downloaded on first run (~5.6 MB). No manual model setup is needed.

Note: The `.vscode/settings.json` references a `dance_env` interpreter path which doesn't exist — use `.venv` instead.

## Architecture

**`score_dance.py`** — Main scoring pipeline with three modules:

- `PoseFrame` dataclass — stores 3D world-landmark keypoints per frame and auto-computes 12 joint angles (elbows, knees, hip-ankle lines, shoulder-hip angles) from the configured `ANGLE_JOINTS` triplets.
- `PoseExtractor` — wraps MediaPipe `PoseLandmarker` in VIDEO mode. Samples frames down to `TARGET_FPS` (30), extracts 33 pose landmarks, then interpolates low-confidence keypoints from neighboring frames (`interp_window=3`).
- `Scorer` — full scoring pipeline:
  1. DTW alignment between reference and user angle sequences
  2. Motion change detection via angle-difference peaks to find segment boundaries
  3. Per-frame pair scoring (100 − mean absolute angle difference)
  4. Segments below `SCORE_THRESHOLD` are exported as slow-motion clips to `output/low_score_clips/`

**`split_8beats.py`** — Standalone tool that slices the reference video into `BEATS_PER_SEGMENT` (8) beat segments based on BPM, renders each at 0.8× speed, and merges all segments into one output file. Uses ffmpeg concat when available, falls back to OpenCV.

**Key constants** (both files): `BEATS_PER_SEGMENT=8`, `SLOW_SPEED=0.8`, `TARGET_FPS=30`, `SCORE_THRESHOLD=50.0`.

**Input/Output conventions:**
- Reference and user videos go in `videos/`
- Segment clips output to `output/segments/`
- Low-score practice clips output to `output/low_score_clips/`

# core/config.py — 全局配置与常量

from dataclasses import dataclass
import numpy as np

BEATS_PER_SEGMENT = 8
DEFAULT_BPM = 120
SCORE_THRESHOLD = 50.0
PASS_SCORE = 60.0
SLOW_SPEED = 0.8
TARGET_FPS = 30
MIN_SEGMENT_DURATION = 0.3

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

ANGLE_JOINTS = [
    (1, 0, 4), (7, 0, 8), (7, 0, 11), (8, 0, 12),
    (11, 13, 15), (12, 14, 16), (13, 15, 17), (14, 16, 18),
    (13, 15, 19), (14, 16, 20), (15, 17, 19), (16, 18, 20),
    (13, 11, 23), (14, 12, 24), (11, 23, 24), (12, 24, 23),
    (11, 23, 25), (12, 24, 26), (23, 25, 27), (24, 26, 28),
    (25, 27, 29), (26, 28, 30), (25, 27, 31), (26, 28, 32),
    (27, 29, 31), (28, 30, 32),
]

ANGLE_WEIGHTS = np.array([1.0,1.0,0.8,0.8,1.3,1.3,0.6,0.6,0.5,0.5,
    0.4,0.4,1.2,1.2,1.4,1.4,1.3,1.3,1.5,1.5,0.8,0.8,0.7,0.7,0.6,0.6],
    dtype=np.float32)

SCORE_TOLERANCE = 3.0
SCORE_PENALTY_SMALL = 1.8
SCORE_PENALTY_LARGE = 3.0
SCORE_PENALTY_THRESHOLD = 15.0

DTW_WINDOW_RATIO = 0.1
Z_AXIS_WEIGHT = 0.3

# Visibility filtering — 过滤画面外/低置信度关键点
VISIBILITY_THRESHOLD = 0.5       # MediaPipe landmark visibility 低于此值视为不可见
MIN_VISIBLE_FRAME_RATIO = 0.3    # 关节在某段中可见帧占比低于此值则跳过评分

# 舞蹈评分排除列表 — 面部关键点（与舞蹈动作无关）
# 0=鼻尖, 1-6=眼, 7-8=耳, 9-10=嘴角
DANCE_EXCLUDED_JOINTS = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})


@dataclass
class Config:
    score_threshold: float = 50.0
    target_fps: int = 30
    keypoint_confidence: float = 0.5
    interp_window: int = 3

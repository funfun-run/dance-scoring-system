# config.py - 基础支撑层（全局配置 + OpenVINO模型管理）
# 平台：Intel DK-2500 (Core Ultra 5 225U)
# NPU: 11TOPS | GPU: 8TOPS | SOC: 24TOPS

import os
import urllib.request
import numpy as np

# ==================== 硬件平台 ====================
PLATFORM = "Intel_DK2500"
PROCESSOR = "Core Ultra 5 225U"

# OpenVINO设备优先级
DEVICE_PRIORITY = ["NPU", "GPU", "CPU"]
NPU_DEVICE = "NPU"
GPU_DEVICE = "GPU"
CPU_DEVICE = "CPU"

# ==================== 模型路径 ====================
MODEL_DIR = "models"
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MEDIAPIPE_MODEL_PATH = "pose_landmarker_lite.task"

# ==================== 推理配置 ====================
INFERENCE_PRECISION = "FP16"
NUM_INFERENCE_THREADS = 4
TARGET_FPS = 30
MAX_INFERENCE_TIME_MS = 50

# ==================== 评分配置 ====================
BEATS_PER_SEGMENT = 8
DEFAULT_BPM = 120
PASS_SCORE = 60.0
SCORE_THRESHOLD = 50.0
DTW_WINDOW_RATIO = 0.1

SCORE_TOLERANCE = 3.0
SCORE_PENALTY_SMALL = 1.8
SCORE_PENALTY_LARGE = 3.0
SCORE_PENALTY_THRESHOLD = 15.0

# ==================== 显示与输出 ====================
SLOW_SPEED = 0.8
OUTPUT_SEGMENTS_DIR = "output/segments"
OUTPUT_LOW_SCORE_DIR = "output/low_score_clips"

# 26个关节角度（覆盖33个关键点）
ANGLE_JOINTS = [
    (1, 0, 4), (7, 0, 8), (7, 0, 11), (8, 0, 12),
    (11, 13, 15), (12, 14, 16), (13, 15, 17), (14, 16, 18),
    (13, 15, 19), (14, 16, 20), (15, 17, 19), (16, 18, 20),
    (13, 11, 23), (14, 12, 24), (11, 23, 24), (12, 24, 23),
    (11, 23, 25), (12, 24, 26), (23, 25, 27), (24, 26, 28),
    (25, 27, 29), (26, 28, 30), (25, 27, 31), (26, 28, 32),
    (27, 29, 31), (28, 30, 32),
]

ANGLE_WEIGHTS = [
    1.0, 1.0, 0.8, 0.8, 1.3, 1.3, 0.6, 0.6, 0.5, 0.5,
    0.4, 0.4, 1.2, 1.2, 1.4, 1.4, 1.3, 1.3, 1.5, 1.5,
    0.8, 0.8, 0.7, 0.7, 0.6, 0.6,
]


# ==================== OpenVINO 模型管理器 ====================
try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    print("[提示] OpenVINO未安装，NPU加速不可用。安装命令: pip install openvino>=2024.0.0")


class ModelManager:
    """OpenVINO模型管理器（下载、检测设备）"""
    
    def __init__(self):
        self.core = None
        self.target_device = CPU_DEVICE
        self.available_devices = []
        if HAS_OPENVINO:
            self._init()
    
    def _init(self):
        try:
            self.core = ov.Core()
            all_devs = self.core.available_devices
            print(f"[OpenVINO] 可用设备: {all_devs}")
            for d in DEVICE_PRIORITY:
                if d in all_devs:
                    self.available_devices.append(d)
            if self.available_devices:
                self.target_device = self.available_devices[0]
            else:
                self.target_device = CPU_DEVICE
            print(f"[OpenVINO] 推理设备: {self.target_device}")
        except Exception as e:
            print(f"[OpenVINO] 初始化失败: {e}")
            self.core = None
    
    def download_model(self):
        if not os.path.exists(MEDIAPIPE_MODEL_PATH):
            print("[模型] 下载 MediaPipe Pose Landmarker...")
            try:
                urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
                print("[模型] 下载完成")
            except Exception as e:
                print(f"[模型] 下载失败: {e}")
    
    def get_device_info(self):
        return {
            'openvino_available': HAS_OPENVINO and self.core is not None,
            'target_device': self.target_device,
            'available_devices': self.available_devices
        }


# 全局实例
model_mgr = ModelManager()
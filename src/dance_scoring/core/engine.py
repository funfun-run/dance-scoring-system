# core/engine.py — 姿态估计引擎抽象层

"""
Pose estimation engine abstraction layer.

提供统一接口，业务代码透明切换 MediaPipe / OpenVINO 后端。
GUI 层完全不感知底层推理引擎。
"""

from typing import Protocol, List, Optional
from pathlib import Path

import numpy as np
import cv2

from .config import Config, Z_AXIS_WEIGHT, TARGET_FPS
from .frame import PoseFrame


# ============================================================
# 抽象接口
# ============================================================

class PoseEngine(Protocol):
    """姿态估计引擎接口。"""

    def extract(self, path: str) -> List[PoseFrame]:
        """从视频文件提取姿态序列（离线模式）。"""
        ...

    def extract_frame(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseFrame]:
        """从单帧 RGB 图像提取姿态（实时模式）。"""
        ...

    @property
    def backend_name(self) -> str:
        """人类可读的后端标识。"""
        ...


# ============================================================
# 共享插值逻辑
# ============================================================

def shared_interpolate(poses: List[PoseFrame], cfg: Config) -> List[PoseFrame]:
    """
    共享的丢失关键点邻帧插值逻辑。
    从 PoseExtractor._interpolate() 提取，供两个后端共用。
    """
    if len(poses) < 2:
        return poses

    w = cfg.interp_window
    for i, p in enumerate(poses):
        mask = p.conf < cfg.keypoint_confidence
        if np.any(mask):
            pi, ni = max(0, i - w), min(len(poses) - 1, i + w)
            for j in range(33):
                if mask[j]:
                    a = (i - pi) / max(1, ni - pi)
                    new_val = (1 - a) * poses[pi].kp3d[j] + a * poses[ni].kp3d[j]
                    if j in [13, 14, 25, 26]:
                        parent = j - 2
                        old_len = np.linalg.norm(p.kp3d[j] - p.kp3d[parent])
                        new_len = np.linalg.norm(new_val - p.kp3d[parent])
                        if old_len > 0 and abs(new_len - old_len) / old_len < 0.2:
                            p.kp3d[j] = new_val
                    elif j in [15, 16, 27, 28]:
                        parent = j - 2
                        old_len = np.linalg.norm(p.kp3d[j] - p.kp3d[parent])
                        new_len = np.linalg.norm(new_val - p.kp3d[parent])
                        if old_len > 0 and abs(new_len - old_len) / old_len < 0.3:
                            p.kp3d[j] = new_val
                    else:
                        p.kp3d[j] = new_val
            p.angles = p._calc_angles()
            p.vec = np.concatenate([p.kp3d[:, :2].flatten(), p.angles])
    return poses


# ============================================================
# MediaPipe 后端
# ============================================================

class MediaPipeEngine:
    """
    基于现有 PoseExtractor 的 MediaPipe CPU 后端。

    封装 PoseExtractor 为 PoseEngine 接口，零额外依赖。
    """

    def __init__(self, cfg: Config = Config()):
        from .extractor import PoseExtractor, download_model
        download_model()
        self._extractor = PoseExtractor(cfg)
        self._cfg = cfg
        self._frame_counter = 0

    def extract(self, path: str) -> List[PoseFrame]:
        """离线模式：从视频文件提取全部姿态。"""
        return self._extractor.extract(path)

    def extract_frame(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseFrame]:
        """实时模式：从单帧提取姿态。"""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python

        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._extractor.det.detect_for_video(mp_img, ts_ms)

            if (res.pose_world_landmarks and
                    len(res.pose_world_landmarks) > 0 and
                    len(res.pose_world_landmarks[0]) >= 33):
                kp3d = np.zeros((33, 3), dtype=np.float32)
                cf = np.zeros(33, dtype=np.float32)
                for i, lm in enumerate(res.pose_world_landmarks[0][:33]):
                    kp3d[i] = [lm.x, lm.y, lm.z * Z_AXIS_WEIGHT]
                    cf[i] = lm.visibility if hasattr(lm, 'visibility') else 1.0

                self._frame_counter += 1
                return PoseFrame(self._frame_counter, kp3d, cf)
            return None
        except Exception:
            return None

    @property
    def backend_name(self) -> str:
        return "MediaPipe (CPU)"


# ============================================================
# OpenVINO 后端
# ============================================================

class OpenVINOEngine:
    """
    基于 OpenVINO IR 模型的加速后端。

    支持 NPU / GPU / CPU，自动回退。
    """

    def __init__(
        self,
        cfg: Config = Config(),
        model_dir: Optional[Path] = None,
        device: str = "AUTO",
    ):
        from .inference import PoseInferenceEngine

        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models"

        self._engine = PoseInferenceEngine(Path(model_dir), device)
        self._cfg = cfg
        self._frame_counter = 0

    def extract(self, path: str) -> List[PoseFrame]:
        """离线模式：读取视频 → 逐帧 OpenVINO 推理 → 插值。"""
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = max(1, int(fps / self._cfg.target_fps))

        poses: List[PoseFrame] = []
        fid, proc = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fid % skip == 0:
                # OpenCV 默认 BGR → RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = proc * int(1000 / self._cfg.target_fps)
                pf = self.extract_frame(rgb, ts)
                if pf is not None:
                    poses.append(pf)
                proc += 1
            fid += 1
            if fid % 200 == 0:
                print(f"  进度:{100 * fid // total}%")

        cap.release()
        print(f"  提取:{len(poses)}帧")

        return shared_interpolate(poses, self._cfg)

    def extract_frame(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseFrame]:
        """实时模式：单帧 OpenVINO 推理 → PoseFrame。"""
        try:
            result = self._engine.infer(rgb)

            if result.presence < 0.1:
                return None

            self._frame_counter += 1
            pf = PoseFrame(
                self._frame_counter,
                result.kp3d.copy(),
                result.visibility.copy(),
            )
            return pf
        except Exception:
            return None

    @property
    def backend_name(self) -> str:
        return f"OpenVINO ({self._engine.device})"


# ============================================================
# 工厂函数
# ============================================================

def create_pose_engine(
    backend: str = "auto",
    cfg: Config = Config(),
    model_dir: Optional[Path] = None,
) -> PoseEngine:
    """
    创建最佳可用的姿态估计引擎。

    参数:
        backend:
            "auto"      → 自动选择 OpenVINO (NPU/GPU/CPU) → MediaPipe 回退
            "mediapipe" → 强制 MediaPipe
            "openvino"  → 强制 OpenVINO IR (缺少模型时抛异常)
        cfg: 全局配置
        model_dir: IR 模型目录 (仅 OpenVINO 后端使用)

    返回:
        PoseEngine 实例
    """
    if backend == "mediapipe":
        return MediaPipeEngine(cfg)

    if backend == "openvino":
        return OpenVINOEngine(cfg, model_dir)

    # auto: 优先尝试 OpenVINO，失败则回退 MediaPipe
    try:
        engine = OpenVINOEngine(cfg, model_dir)
        return engine
    except Exception as e:
        print(f"⚠ OpenVINO 不可用 ({e})，回退到 MediaPipe")
        return MediaPipeEngine(cfg)

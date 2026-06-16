# camera/usb.py — USB 摄像头实现

import cv2
import numpy as np
from typing import Optional, Tuple

from dance_scoring.camera.base import CameraBase


class UsbCamera(CameraBase):
    """OpenCV VideoCapture 封装的 USB 摄像头。

    参数:
        device_id: 摄像头设备索引 (0, 1, ...)
        resolution: 目标分辨率 (width, height)
        fps: 目标帧率
    """

    def __init__(
        self,
        device_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        self._device_id = device_id
        self._resolution = resolution
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """打开 USB 摄像头并设置分辨率/帧率。失败返回 False。"""
        try:
            self._cap = cv2.VideoCapture(self._device_id)
            if not self._cap.isOpened():
                self._cap = None
                return False

            w, h = self._resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            return True
        except Exception:
            self._cap = None
            return False

    def read(self) -> Optional[np.ndarray]:
        """
        读取一帧 RGB 图像。
        OpenCV 默认返回 BGR，此处转换为 RGB 以兼容 MediaPipe。
        失败返回 None。
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def close(self) -> None:
        """释放摄像头资源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """返回摄像头是否已打开。"""
        return self._cap is not None and self._cap.isOpened()

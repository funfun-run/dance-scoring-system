# camera/stream.py — 网络流接收

import cv2
import numpy as np
from typing import Optional

from dance_scoring.camera.base import CameraBase


class NetworkStream(CameraBase):
    """RTSP/HTTP 网络视频流接收。

    通过 OpenCV VideoCapture 的通用 URL 能力支持:
        - RTSP: rtsp://username:password@ip:port/stream
        - HTTP: http://ip:port/video
        - 本地文件路径

    参数:
        url: 视频流地址
    """

    def __init__(self, url: str):
        self._url = url
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """打开网络流。失败返回 False，不抛异常。"""
        try:
            self._cap = cv2.VideoCapture(self._url)
            if not self._cap.isOpened():
                self._cap = None
                return False
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
        """释放视频流资源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        """返回视频流是否已连接。"""
        return self._cap is not None and self._cap.isOpened()

# camera/base.py — 摄像头抽象基类

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class CameraBase(ABC):
    """摄像头采集抽象接口。"""

    @abstractmethod
    def open(self) -> bool:
        """打开摄像头设备/流，返回是否成功。失败不抛异常，返回 False。"""
        ...

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        读取一帧 RGB 图像 (H, W, 3) uint8。
        返回 RGB 格式以与 MediaPipe mp.ImageFormat.SRGB 兼容。
        失败返回 None。
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """释放摄像头资源。"""
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """返回摄像头当前是否已打开。"""
        ...

    @staticmethod
    def list_devices(max_index: int = 8) -> List[int]:
        """枚举可用摄像头设备索引列表。"""
        import cv2
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

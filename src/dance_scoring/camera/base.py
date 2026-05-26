# camera/base.py — 摄像头抽象基类（预留）

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class CameraBase(ABC):
    """摄像头采集抽象接口。后续实现 USB 摄像头和网络流。"""

    @abstractmethod
    def open(self, device_id: int = 0) -> bool:
        """打开摄像头，返回是否成功"""
        ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """读取一帧 BGR 图像，失败返回 None"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放摄像头资源"""
        ...

    @staticmethod
    def list_devices() -> List[str]:
        """列出可用摄像头设备名称列表"""
        return []

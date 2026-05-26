# transfer/base.py — 文件传输抽象基类（预留）

from abc import ABC, abstractmethod
from typing import List


class TransferBase(ABC):
    """无线文件传输抽象接口。后续实现 WiFi Direct 和蓝牙传输。"""

    @abstractmethod
    def send(self, file_path: str, target: str) -> bool:
        """发送文件到目标设备"""
        ...

    @abstractmethod
    def receive(self, save_dir: str) -> str | None:
        """接收文件，返回保存路径"""
        ...

    @abstractmethod
    def discover(self) -> List[str]:
        """发现附近可用设备"""
        ...

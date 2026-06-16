# platform/npu.py — DK-2500 NPU 加速接口

"""
Intel Core Ultra 5 225U (Meteor Lake) NPU 设备管理。
通过 OpenVINO Runtime 检测和管理 NPU/GPU/CPU 设备。
"""

from typing import Dict


class NPUManager:
    """NPU device manager for Intel Core Ultra 5 225U."""

    @staticmethod
    def available() -> bool:
        """检查 NPU 设备是否可用。"""
        try:
            import openvino as ov
            return "NPU" in ov.Core().available_devices
        except Exception:
            return False

    @staticmethod
    def best_device() -> str:
        """
        获取最佳可用推理设备。

        优先级: NPU > GPU > CPU
        """
        try:
            import openvino as ov
            devices = ov.Core().available_devices
            for d in ["NPU", "GPU", "CPU"]:
                if d in devices:
                    return d
            return "CPU"
        except Exception:
            return "CPU"

    @staticmethod
    def device_info(device: str = "NPU") -> Dict:
        """
        返回设备属性信息。

        返回:
            {"available": bool, "device": str, "name": str, ...}
        """
        try:
            import openvino as ov
            core = ov.Core()
            if device not in core.available_devices:
                return {"available": False, "device": device}

            props = core.get_property(device, "FULL_DEVICE_NAME")
            return {
                "available": True,
                "device": device,
                "name": props,
            }
        except Exception as e:
            return {"available": False, "device": device, "error": str(e)}

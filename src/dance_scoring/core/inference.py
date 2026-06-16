# core/inference.py — OpenVINO 加速姿态推理引擎

"""
OpenVINO IR 推理引擎，加载 pose_landmarker.xml + .bin 进行 NPU/GPU/CPU 推理。

预处理：resize(256×256) → RGB→[0,1] → NCHW
后处理：解析多输出张量 → PoseInferenceResult
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import cv2


@dataclass
class PoseInferenceResult:
    """OpenVINO 推理结果 — 单帧姿态数据。"""
    kp3d: np.ndarray          # (33, 3) 世界坐标
    kp2d: np.ndarray          # (33, 2) 像素坐标
    visibility: np.ndarray    # (33,) 置信度 0~1
    presence: float           # 人体存在分数


class PoseInferenceEngine:
    """
    OpenVINO-based pose inference engine.

    加载 IR 模型，处理预处理/后处理，支持 NPU/GPU/CPU。
    如果目标设备编译失败，自动回退到 best_device()。

    参数:
        model_dir: IR 模型目录 (包含 .xml, .bin, meta.json)
        device: 目标推理设备 ("NPU" | "GPU" | "CPU" | "AUTO")
    """

    def __init__(self, model_dir: Path, device: str = "AUTO"):
        import openvino as ov

        self.model_dir = Path(model_dir)
        self._device = device

        # 1. 读取 meta.json
        meta_path = self.model_dir / "pose_landmarker_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {}

        # 2. 确定输入尺寸
        inp_spec = self._meta.get("input", {})
        inp_shape = inp_spec.get("shape", [1, 256, 256, 3])
        self.input_h = inp_shape[1]
        self.input_w = inp_shape[2]

        # 3. 加载并编译模型
        xml_path = self.model_dir / "pose_landmarker.xml"
        if not xml_path.exists():
            raise FileNotFoundError(
                f"IR 模型不存在: {xml_path}\n"
                f"请先运行: python scripts/convert_model.py"
            )

        core = ov.Core()
        model = core.read_model(str(xml_path))

        # 尝试编译，失败则回退
        try:
            self._compiled = core.compile_model(model, device)
        except Exception as e:
            fallback = self._fallback_device()
            print(f"⚠️ {device} 编译失败 ({e})，回退到 {fallback}")
            self._compiled = core.compile_model(model, fallback)

        self._device = self._resolve_device_name()
        self._infer_request = self._compiled.create_infer_request()
        self._output_names = [o.any_name for o in model.outputs]

        print(f"✅ PoseInferenceEngine: {self._device} | "
              f"输入={self.input_w}×{self.input_h} | "
              f"输出={len(self._output_names)}个张量")

    @property
    def device(self) -> str:
        """当前实际使用的推理设备。"""
        return str(self._device)

    @property
    def input_size(self) -> tuple:
        """模型输入尺寸 (height, width)。"""
        return (self.input_h, self.input_w)

    # ---------- 推理 ----------

    def infer(self, rgb_image: np.ndarray) -> PoseInferenceResult:
        """
        对单帧 RGB 图像执行姿态推理。

        参数:
            rgb_image: (H, W, 3) uint8 RGB 图像

        返回:
            PoseInferenceResult — 包含 kp3d, kp2d, visibility, presence
        """
        # 预处理
        tensor = self._preprocess(rgb_image)

        # 推理 (使用 infer_request.infer，返回 {output_name: tensor})
        results = self._infer_request.infer({self._input_name(): tensor})

        # 后处理
        return self._postprocess(results, rgb_image.shape[:2])

    def warmup(self, rounds: int = 3) -> None:
        """用空图预热推理引擎，避免首帧冷启动延迟尖峰。"""
        dummy = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        for i in range(rounds):
            _ = self.infer(dummy)
        print(f"  预热完成 ({rounds} 轮)")

    # ---------- 内部方法 ----------

    def _input_name(self) -> str:
        """获取输入张量名称。"""
        meta_input = self._meta.get("input", {})
        return meta_input.get("name", "input_1")

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """
        预处理：resize → float32 → [0,1] 归一化 → NHWC (batch dim)。
        根据 meta.json 中的 input shape 确定布局 (NHWC 或 NCHW)。
        """
        # Resize 到模型输入尺寸
        resized = cv2.resize(rgb, (self.input_w, self.input_h))
        # 归一化 [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        # 添加 batch 维度: (H, W, 3) → (1, H, W, 3) 保持 NHWC
        tensor = np.expand_dims(normalized, axis=0)
        return tensor

    def _postprocess(
        self,
        outputs: Dict[str, np.ndarray],
        original_shape: tuple,
    ) -> PoseInferenceResult:
        """
        解析多输出张量为结构化姿态数据。

        MediaPipe Pose Landmarker 输出格式:
          - output_0: [1, 195] → 关键点坐标 (可能是 39×5 格点)
          - output_1: [1, 1]   → 人体存在分数
          - output_2: [1, 256, 256, 1] → 分割蒙版
          - output_3: [1, 64, 64, 39] → 热力图 (39 个关键点)
          - output_4: [1, 117] → 补充坐标

        本方法提取前 33 个关键点的世界坐标和像素坐标。
        """
        oh, ow = original_shape

        # 获取输出张量
        names = list(outputs.keys())

        # Identity: [1, 195] — 关键点展平数据
        identity = outputs.get(names[0], None)
        if identity is not None:
            flat = identity.flatten()
            # 尝试解析为 39 个关键点 × 5 个值 (x, y, z, vis, presence)
            n_kpts = min(len(flat) // 5, 39)
            reshaped = flat[:n_kpts * 5].reshape(n_kpts, 5)

            # 取前 33 个关键点
            n_out = min(33, n_kpts)
            kp3d = np.zeros((33, 3), dtype=np.float32)
            kp2d = np.zeros((33, 2), dtype=np.float32)
            visibility = np.zeros(33, dtype=np.float32)

            for i in range(n_out):
                x, y, z = reshaped[i, 0], reshaped[i, 1], reshaped[i, 2]
                kp3d[i] = [x, y, z * 0.3]
                # 像素坐标 (归一化 → 实际像素)
                kp2d[i] = [x * ow, y * oh]
                visibility[i] = min(1.0, max(0.0, reshaped[i, 3])) if reshaped.shape[1] > 3 else 1.0
        else:
            kp3d = np.zeros((33, 3), dtype=np.float32)
            kp2d = np.zeros((33, 2), dtype=np.float32)
            visibility = np.zeros(33, dtype=np.float32)

        # 人体存在分数 (Identity_1: [1, 1])
        presence = 1.0
        if len(names) >= 2:
            p = outputs.get(names[1], None)
            if p is not None:
                presence = float(p.flatten()[0])

        return PoseInferenceResult(
            kp3d=kp3d,
            kp2d=kp2d,
            visibility=visibility,
            presence=presence,
        )

    def _resolve_device_name(self) -> str:
        """解析编译后的实际设备名称。"""
        try:
            return self._compiled.get_property("FULL_DEVICE_NAME")
        except Exception:
            pass
        try:
            # AUTO 设备不支持 FULL_DEVICE_NAME，获取实际执行设备
            return str(self._compiled.get_property("EXECUTION_DEVICES"))
        except Exception:
            return "CPU"

    @staticmethod
    def _fallback_device() -> str:
        """获取回退设备。"""
        try:
            from dance_scoring.platform.npu import NPUManager
            return NPUManager.best_device()
        except Exception:
            return "CPU"

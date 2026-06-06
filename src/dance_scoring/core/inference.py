"""
OpenVINO inference acceleration layer (placeholder).
TODO: MediaPipe supports OpenVINO delegate. Integration options:
- Configure OpenVINO backend via MediaPipe Python API
- Export MediaPipe Pose as OpenVINO IR, load via inference engine
Ref: Intel Core Ultra 5 225U NPU, Ubuntu 22.04 + OpenVINO toolkit.
"""

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Placeholder config. Fields TBD based on actual integration approach."""
    device: str = "NPU"          # "NPU" | "CPU" | "GPU"
    precision: str = "INT8"      # "FP32" | "FP16" | "INT8"


class PoseInferenceEngine:
    """
    Accelerated pose inference engine (placeholder).
    Currently using MediaPipe PoseLandmarker directly (core/extractor.py).
    Fill implementation when OpenVINO acceleration is needed.
    """

    def __init__(self, model_path: str, cfg: InferenceConfig) -> None:
        raise NotImplementedError(
            "OpenVINO inference not yet implemented. "
            "See core/inference.py docstring for integration guidance."
        )

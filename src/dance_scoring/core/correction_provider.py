# core/correction_provider.py — 纠正建议提供者抽象层
#
# 提供统一接口，业务代码透明切换 RuleBased / LLM 后端。
# 与 PoseEngine Protocol + create_pose_engine() 模式一致。

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Deviation:
    """单个关节的角度偏差。"""
    joint_name: str       # 中文关节名，如 "右肘"
    joint_idx: int        # MediaPipe 33 关键点索引
    deviation_deg: float  # 角度偏差（正=用户角度偏大, 负=用户角度偏小）
    direction: str        # "too_bent" | "too_straight"


@dataclass
class SegmentInfo:
    """单个舞蹈段的结构化信息，供 CorrectionProvider 使用。"""
    id: int                          # 段号 (1-based)
    score: float                     # 该段得分 0-100
    qualified: bool                  # 是否合格
    start_time: float                # 起始时间 (秒)
    end_time: float                  # 结束时间 (秒)
    deviations: List[Deviation] = field(default_factory=list)
    correction_text: str = ""        # 纠正文本（由 Provider 填充）


# ============================================================
# 抽象接口
# ============================================================

class CorrectionProvider(Protocol):
    """纠正建议提供者抽象接口。"""

    def generate_correction(
        self,
        segment: SegmentInfo,
        overall_score: float,
        bpm: float,
    ) -> str:
        """
        为单个段生成中文纠正建议。

        参数:
            segment: 段信息
            overall_score: 总分
            bpm: 节拍速度

        返回:
            一段中文纠正文本。已合格的段返回空字符串 ""。
        """
        ...

    @property
    def provider_name(self) -> str:
        """人类可读的提供者名称，如 "RuleEngine" / "Qwen2.5-1.5B"."""
        ...


# ============================================================
# 规则引擎实现（封装现有 correction.py）
# ============================================================

class RuleBasedProvider:
    """
    基于规则模板的纠正建议生成器。

    封装 correction.py 的 generate_correction()，适配单段调用接口。
    """

    def generate_correction(
        self,
        segment: SegmentInfo,
        overall_score: float,
        bpm: float,
    ) -> str:
        """使用规则模板生成纠正文本。"""
        from .correction import _format_correction

        if segment.qualified:
            return ""

        if not segment.deviations:
            return ""

        corrections = []
        for d in segment.deviations[:3]:  # Top-3
            text = _format_correction(d.joint_idx, d.deviation_deg, d.direction)
            corrections.append(text)

        if corrections:
            return f"第{segment.id}段：" + "；".join(corrections)
        return ""

    @property
    def provider_name(self) -> str:
        return "RuleEngine"


# ============================================================
# LLM 实现（适配器）
# ============================================================

class LLMAdapterProvider:
    """
    将 LLMProvider 适配为 CorrectionProvider 接口。

    不直接调用 LLM — 委托给 LLMProvider 子类实例。
    这个类解决 LLMProvider (ABC) 与 CorrectionProvider (Protocol) 之间的类型适配。
    """

    def __init__(self, llm_provider):
        """
        参数:
            llm_provider: LLMProvider 子类实例（已加载模型的）
        """
        self._llm = llm_provider

    def generate_correction(
        self,
        segment: SegmentInfo,
        overall_score: float,
        bpm: float,
    ) -> str:
        return self._llm.generate_correction(
            seg_id=segment.id,
            seg_score=segment.score,
            qualified=segment.qualified,
            deviations=segment.deviations,
            overall_score=overall_score,
            bpm=bpm,
        )

    @property
    def provider_name(self) -> str:
        return self._llm.provider_name


# ============================================================
# 工具函数
# ============================================================

def _ensure_project_root_on_path():
    """确保项目根目录在 sys.path 中，以便导入 LLM/ 包。"""
    # correction_provider.py → core/ → dance_scoring/ → src/ → 项目根
    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# ============================================================
# 工厂函数
# ============================================================

def create_correction_provider(
    backend: str = "rule",
    model: str = "1.5b",
    **kwargs,
) -> CorrectionProvider:
    """
    创建纠正建议提供者。

    参数:
        backend:
            "rule" → RuleBasedProvider（默认，零依赖）
            "llm"  → LLMProvider 子类
        model: 模型选择（仅 backend="llm" 时有效）
            "1.5b" → Qwen2.5 1.5B OpenVINO（默认，速度快）
            "3b"   → Qwen2.5 3B llama.cpp（回答精准）
        **kwargs: 传递给 LLMProvider 构造函数的额外参数

    返回:
        CorrectionProvider 实例
    """
    if backend == "rule":
        return RuleBasedProvider()

    if backend == "llm":
        _ensure_project_root_on_path()

        try:
            from LLM.model_manager import load_model as load_llm_model
            llm = load_llm_model(model)
            return LLMAdapterProvider(llm)
        except ImportError as e:
            raise ImportError(
                f"无法导入 LLM 模块: {e}\n"
                "请确认 LLM/ 目录存在且包含 model_manager.py"
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"模型 {model} 加载失败: {e}\n"
                "请确认模型文件存在且依赖已安装。"
            )

    raise ValueError(f"未知的纠正后端: {backend}，可选值: rule, llm")

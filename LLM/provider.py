# LLM/provider.py — LLMProvider 抽象基类
#
# 接入者只需:
#   1. 继承 LLMProvider
#   2. 实现 _call_llm(prompt: str) -> str
#   3. 实现 provider_name 属性
#
# Prompt 构建、响应解析、纠正文段拼接全部由基类完成。

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from .prompts import (
    SYSTEM_PROMPT,
    CORRECTION_PROMPT_TEMPLATE,
    format_deviations,
)


class LLMProvider(ABC):
    """
    LLM 纠正建议提供者抽象基类。

    子类必须实现:
        - _call_llm(prompt) → str
        - provider_name → str
    """

    # —— 子类必须实现 ——

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """
        调用本地 LLM 进行推理，返回原始文本。

        参数:
            prompt: 完整的 Prompt 字符串（已由 _build_prompt 构建）

        返回:
            模型输出的原始文本
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """人类可读的提供者名称，如 "Qwen2.5-1.5B"."""
        ...

    # —— Prompt 构建（基类实现） ——

    def _build_prompt(
        self,
        seg_id: int,
        seg_score: float,
        qualified: bool,
        deviations: list,
        overall_score: float,
        bpm: float,
    ) -> str:
        """
        构建单段纠正建议的完整 Prompt。

        参数:
            seg_id: 段号
            seg_score: 该段得分
            qualified: 是否合格
            deviations: Deviation 对象列表
            overall_score: 总分
            bpm: 节拍速度

        返回:
            完整的 Prompt 字符串，可直接喂给 LLM
        """
        qualified_str = "合格" if qualified else "不合格"
        deviations_text = format_deviations(deviations, top_n=3)

        return CORRECTION_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            overall_score=overall_score,
            bpm=bpm,
            seg_id=seg_id,
            seg_score=seg_score,
            qualified_str=qualified_str,
            deviations_text=deviations_text,
        )

    # —— 响应解析（基类实现） ——

    def _parse_response(self, text: str) -> str:
        """
        解析 LLM 原始输出，提取纠正文本。

        策略:
          1. 尝试 JSON 解析 → 取 "correction" 字段
          2. Fallback: 取第一行非空、非 JSON 括号的文本
          3. 兜底: 返回原始文本的前 80 个字符

        参数:
            text: LLM 返回的原始文本

        返回:
            清理后的纠正文本。如果文本为 "合格"，返回空字符串。
        """
        text = text.strip()

        # 策略 1: JSON 解析
        try:
            data = json.loads(text)
            correction = data.get("correction", "")
            if correction and correction != "合格":
                return correction.strip()
            return ""
        except (json.JSONDecodeError, TypeError):
            pass

        # 尝试提取 JSON 片段 (模型可能在 JSON 前后加了废话)
        json_match = re.search(r'\{[^{}]*"correction"[^{}]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                correction = data.get("correction", "")
                if correction and correction != "合格":
                    return correction.strip()
                return ""
            except (json.JSONDecodeError, TypeError):
                pass

        # 策略 2: 行式解析 — 跳过纯 JSON 括号行，取第一行有意义文本
        for line in text.split("\n"):
            line = line.strip()
            # 跳过空行、JSON 结构行
            if not line or line in ("{", "}", "```", "```json"):
                continue
            # 跳过看起来像 JSON key 的行
            if re.match(r'^["\']?\w+["\']?\s*:', line):
                continue
            # 如果输出就是"合格"，返回空
            if line == "合格":
                return ""
            return line

        # 策略 3: 兜底 — 取前 80 字符（去除 JSON 标记）
        clean = text.strip("{}`\"'\n ")
        return clean[:80] if clean else ""

    # —— 公共接口 ——

    def generate_correction(
        self,
        seg_id: int,
        seg_score: float,
        qualified: bool,
        deviations: list,
        overall_score: float,
        bpm: float,
    ) -> str:
        """
        为单个段生成中文纠正建议（完整流程）。

        参数:
            seg_id: 段号
            seg_score: 该段得分
            qualified: 是否合格
            deviations: Deviation 对象列表，按偏差绝对值降序排列
            overall_score: 总分
            bpm: 节拍速度

        返回:
            中文纠正文本。合格段返回 ""。
        """
        # 合格段不调用 LLM，直接返回空
        if qualified:
            return ""

        prompt = self._build_prompt(
            seg_id=seg_id,
            seg_score=seg_score,
            qualified=qualified,
            deviations=deviations,
            overall_score=overall_score,
            bpm=bpm,
        )

        raw = self._call_llm(prompt)
        return self._parse_response(raw)

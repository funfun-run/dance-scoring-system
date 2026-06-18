# LLM/prompts.py — Qwen2.5 1.5B 中文 Prompt 模板
#
# 设计原则（针对 1.5B 小模型）:
#   - 中文 system prompt，直接拼接结构化数据
#   - 要求输出简短（≤50字），降低模型跑偏概率
#   - 每段单独调用，一次只处理一个段
#   - 合格段直接输出"合格"，不浪费推理

SYSTEM_PROMPT = (
    "你是一个专业的舞蹈教练。根据学员的关节角度偏差数据，"
    "给出一句简短的中文纠正建议，不超过50个字。"
    "建议要具体、可执行，直接告诉学员哪个部位应该怎么调整。"
    "如果该段得分≥60分，只输出「合格」两个字，不要额外说明。"
)

# Prompt 模板 — {fields} 由 LLMProvider._build_prompt() 填充
CORRECTION_PROMPT_TEMPLATE = """{system_prompt}

总分: {overall_score:.1f} | BPM: {bpm:.0f}
第{seg_id}段: 得分{seg_score:.1f} | {qualified_str}
偏差最大的关节:
{deviations_text}
纠正建议:"""


def format_deviation(joint_name: str, deviation_deg: float, direction: str) -> str:
    """
    将单个关节偏差格式化为一行可读文本。

    参数:
        joint_name: 中文关节名，如 "右肘"
        deviation_deg: 角度偏差（正=用户偏大/弯曲过度, 负=用户偏小/伸直不足）
        direction: "too_bent" | "too_straight"
    """
    abs_dev = abs(deviation_deg)
    if direction == "too_bent":
        hint = f"手臂弯曲过度" if "肘" in joint_name else (
            f"站得太直" if "膝" in joint_name else f"角度偏大"
        )
    else:
        hint = f"手臂未伸直" if "肘" in joint_name else (
            f"下蹲不够" if "膝" in joint_name else f"角度偏小"
        )
    return f"- {joint_name}: {deviation_deg:+.1f}° ({hint})"


def format_deviations(deviations: list, top_n: int = 3) -> str:
    """
    将关节偏差列表格式化为 Prompt 中的偏差文本块。

    参数:
        deviations: Deviation 对象列表，按 abs(deviation_deg) 降序排列
        top_n: 最多显示的关节数
    """
    lines = []
    for d in deviations[:top_n]:
        lines.append(format_deviation(d.joint_name, d.deviation_deg, d.direction))
    return "\n".join(lines) if lines else "（无明显偏差）"

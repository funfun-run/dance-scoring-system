"""
Dance posture correction hint generation — 规则引擎.

根据分段得分和关节偏差，自动生成中文纠正建议。
"""

from typing import Dict, List, Optional, Tuple

# ============================================================
# MediaPipe 33 个关键点索引 → 中文身体部位名
# ============================================================
JOINT_NAMES_CN: Dict[int, str] = {
    0:  "鼻尖",
    1:  "左眼内角", 2: "左眼",   3: "左眼外角",
    4:  "右眼内角", 5: "右眼",   6: "右眼外角",
    7:  "左耳",     8: "右耳",
    9:  "嘴角左",  10: "嘴角右",
    11: "左肩",    12: "右肩",
    13: "左肘",    14: "右肘",
    15: "左腕",    16: "右腕",
    17: "左小指",  18: "右小指",
    19: "左食指",  20: "右食指",
    21: "左拇指",  22: "右拇指",
    23: "左髋",    24: "右髋",
    25: "左膝",    26: "右膝",
    27: "左踝",    28: "右踝",
    29: "左脚跟",  30: "右脚跟",
    31: "左脚尖",  32: "右脚尖",
}

# ============================================================
# 关节分组 — 用于确定纠正方向提示词
# ============================================================

# 肘关节: 偏大 → 弯曲过度, 偏小 → 伸直不足
ELBOW_JOINTS = {13, 14}

# 膝关节: 偏大 → 站得太直, 偏小 → 蹲得太深
KNEE_JOINTS = {25, 26}

# 肩关节: 偏大 → 手臂抬太高, 偏小 → 手臂未抬起
SHOULDER_JOINTS = {11, 12}

# 髋关节: 偏大 → 髋部过高, 偏小 → 髋部过低
HIP_JOINTS = {23, 24}

# ============================================================
# 纠正模板
# ============================================================

def _format_correction(
    joint_idx: int,
    deviation: float,
    direction: str  # "too_bent" | "too_straight"
) -> str:
    """根据关节类型和偏差方向生成具体的纠正文本。"""
    name = JOINT_NAMES_CN.get(joint_idx, f"关节{joint_idx}")
    abs_dev = abs(deviation)

    if joint_idx in ELBOW_JOINTS:
        if direction == "too_bent":
            return f"{name}角度偏差{abs_dev:.1f}°，请伸直{name}"
        else:
            return f"{name}角度偏差{abs_dev:.1f}°，请弯曲{name}"

    elif joint_idx in KNEE_JOINTS:
        if direction == "too_bent":
            return f"{name}弯曲不足{abs_dev:.1f}°，请降低重心"
        else:
            return f"{name}过度弯曲{abs_dev:.1f}°，请站直一些"

    elif joint_idx in SHOULDER_JOINTS:
        if direction == "too_bent":
            return f"{name}角度偏差{abs_dev:.1f}°，请放松{name}"
        else:
            return f"{name}角度偏差{abs_dev:.1f}°，请抬起手臂"

    elif joint_idx in HIP_JOINTS:
        if direction == "too_bent":
            return f"{name}角度偏差{abs_dev:.1f}°，请降低{name}"
        else:
            return f"{name}角度偏差{abs_dev:.1f}°，请收紧{name}"

    else:
        # 默认模板
        if direction == "too_bent":
            return f"{name}角度偏差{abs_dev:.1f}°，请调整{name}位置"
        else:
            return f"{name}角度偏差{abs_dev:.1f}°，请调整{name}位置"


# ============================================================
# 主接口
# ============================================================

def generate_correction(
    segment_scores: list,
    top_n: int = 3,
    threshold_deg: float = 10.0,
) -> Dict[int, str]:
    """
    根据分段打分结果生成中文纠正建议。

    参数:
        segment_scores: 分段得分列表，每项需包含:
            - id: 段号 (int)
            - joint_deviations: {joint_idx: mean_deviation_deg}  (dict)
              正偏差 = 用户角度偏大, 负偏差 = 用户角度偏小
        top_n: 每段最多返回的纠正项数
        threshold_deg: 偏差超过此阈值 (绝对值) 才生成建议

    返回:
        {segment_id: "第X段：右肘角度偏差12.5°，请伸直右肘；左膝弯曲不足8.3°，请降低重心"}
        仅返回有至少一个关节超过阈值的段
    """
    result: Dict[int, str] = {}

    for seg in segment_scores:
        seg_id = seg.id if hasattr(seg, 'id') else seg['id']
        deviations = (
            seg.joint_deviations if hasattr(seg, 'joint_deviations')
            else seg.get('joint_deviations', {})
        )

        if not deviations:
            continue

        # 按偏差绝对值排序
        sorted_joints = sorted(
            deviations.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        corrections: List[str] = []
        for joint_idx, dev in sorted_joints:
            if abs(dev) < threshold_deg:
                continue
            if len(corrections) >= top_n:
                break

            # 正偏差 = 用户角度偏大 = too_bent
            # 负偏差 = 用户角度偏小 = too_straight
            direction = "too_bent" if dev > 0 else "too_straight"
            text = _format_correction(int(joint_idx), dev, direction)
            corrections.append(text)

        if corrections:
            result[seg_id] = f"第{seg_id}段：" + "；".join(corrections)

    return result

"""
Dance posture correction hint generation (placeholder).
TODO: Joint-to-body-part name mapping + correction direction inference.
"""

from typing import Dict

# Joint index → Chinese body part name (placeholder, to be populated)
JOINT_NAMES_CN: Dict[int, str] = {}   # e.g. {15: "右肘", 13: "右膝"}


def generate_correction(
    joint_index: int,
    ref_angle: float,
    user_angle: float,
    deviation: float
) -> dict:
    """
    Generate a correction hint for a single joint.

    Returns:
        {
            'joint_name': '右肘',
            'direction': 'too_bent' | 'too_straight' | 'misaligned',
            'suggestion': '请将右肘稍抬高',
            'severity': 'minor' | 'moderate' | 'major'
        }
    """
    raise NotImplementedError()

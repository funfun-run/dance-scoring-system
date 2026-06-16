# core/alignment.py — 快速 DTW 对齐（基于 fastdtw）

import numpy as np
from scipy.spatial.distance import euclidean

try:
    from fastdtw import fastdtw as _fastdtw_impl
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False


def fastdtw_alignment(seq1: np.ndarray, seq2: np.ndarray, radius: int = None):
    """
    使用 fastdtw 进行快速序列对齐（实时模式默认）。

    参数:
        seq1: (N, D) 参考序列
        seq2: (M, D) 用户序列
        radius: 搜索窗口半径，None 则自动设为 max(N,M)//10

    返回:
        (distance: float, path: list of (int, int))

    与 dtw.py 的区别:
        - dtw.py: Sakoe-Chiba 约束窗口 + 完整距离矩阵 → O(N*W)，精确
        - alignment.py: fastdtw 近似算法 → O(N)，快速，适合实时场景
    """
    if not HAS_FASTDTW:
        raise ImportError(
            "fastdtw 未安装，请执行: pip install fastdtw\n"
            "或使用标准 DTW: from dance_scoring.core.dtw import dtw_constrained"
        )

    if radius is None:
        radius = max(len(seq1), len(seq2)) // 10

    distance, path = _fastdtw_impl(seq1, seq2, radius=radius, dist=euclidean)
    return distance, path

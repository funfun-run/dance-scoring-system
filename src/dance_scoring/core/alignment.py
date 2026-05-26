# core/dtw.py — 约束窗口动态时间规整

import numpy as np


def dtw_constrained(mat: np.ndarray, window: int):
    """
    约束窗口 DTW。
    mat: (n_ref, n_user) 距离矩阵
    window: 搜索窗口大小
    返回: (对齐路径 list[(ref_idx, user_idx)], 总代价)
    """
    nr, nu = mat.shape
    cost = np.full((nr, nu), np.inf)
    cost[0, 0] = mat[0, 0]
    for i in range(1, nr):
        cost[i, 0] = cost[i-1, 0] + mat[i, 0]
    for j in range(1, nu):
        cost[0, j] = cost[0, j-1] + mat[0, j]
    for i in range(1, nr):
        for j in range(max(1, i-window), min(nu, i+window+1)):
            cost[i, j] = mat[i, j] + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    path = []
    i, j = nr-1, nu-1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            cand = {(i-1, j-1): cost[i-1, j-1], (i-1, j): cost[i-1, j], (i, j-1): cost[i, j-1]}
            i, j = min(cand, key=cand.get)
    path.append((0, 0))
    path.reverse()
    return path, cost[-1, -1]

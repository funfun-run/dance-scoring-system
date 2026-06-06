# core/frame.py — 单帧姿态数据结构

import numpy as np
from .config import ANGLE_JOINTS


class PoseFrame:
    __slots__ = ('fid', 'kp3d', 'conf', 'angles', 'vec')

    def __init__(self, fid: int, kp3d: np.ndarray, conf: np.ndarray = None):
        self.fid = fid
        self.kp3d = kp3d
        if conf is None:
            conf = np.ones(33, dtype=np.float32)
        self.conf = conf
        self.angles = self._calc_angles()
        self.vec = np.concatenate([self.kp3d[:,:2].flatten(), self.angles])

    def _calc_angles(self):
        kp = self.kp3d[:,:2]
        angles = []
        for a, b, c in ANGLE_JOINTS:
            ba, bc = kp[a]-kp[b], kp[c]-kp[b]
            cos_val = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))
        return np.array(angles, dtype=np.float32)

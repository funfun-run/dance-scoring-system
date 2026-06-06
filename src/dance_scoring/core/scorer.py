# core/scorer.py — 舞蹈评分引擎

import numpy as np
from scipy.spatial.distance import cdist
from typing import List

from .config import (
    Config, DEFAULT_BPM, BEATS_PER_SEGMENT, SCORE_TOLERANCE, SCORE_PENALTY_SMALL,
    SCORE_PENALTY_LARGE, SCORE_PENALTY_THRESHOLD, DTW_WINDOW_RATIO, ANGLE_WEIGHTS, PASS_SCORE
)
from .dtw import dtw_constrained
from .segments import seg_by_beats


class Scorer:
    """DTW 舞蹈评分引擎"""

    def __init__(self, cfg: Config, bpm=DEFAULT_BPM):
        self.cfg = cfg
        self.bpm = bpm
        self.spb = 60.0 / bpm
        self.sps = self.spb * BEATS_PER_SEGMENT
        self.frames_per_seg = int(self.sps * cfg.target_fps)

    def score(self, ref, user, progress_callback=None):
        """对参考视频和用户视频进行评分"""
        nr, nu = len(ref), len(user)
        print(f"  参考:{nr}帧 用户:{nu}帧 BPM:{self.bpm}")

        if progress_callback: progress_callback(0, "DTW对齐...")
        print("  [1/3] DTW对齐...")
        ref_vec = np.array([p.vec for p in ref])
        user_vec = np.array([p.vec for p in user])
        mat = cdist(ref_vec, user_vec, metric='euclidean')
        window = max(int(max(nr, nu)*DTW_WINDOW_RATIO), 1)
        path, cost = dtw_constrained(mat, window)
        print(f"  对齐:{len(path)}对 窗口:{window}")

        if progress_callback: progress_callback(0, "逐帧评分...")
        print("  [2/3] 逐帧评分...")
        fs = []
        for ri, ui in path:
            ang_diff = np.mean(np.abs(ref[ri].angles-user[ui].angles)*ANGLE_WEIGHTS)
            fs.append(self._nonlinear_score(ang_diff))

        if progress_callback: progress_callback(0, "分段评分...")
        print("  [3/3] 八拍分段评分...")
        segs = seg_by_beats(ref, path, fs, self.cfg.target_fps, self.bpm)
        overall = self._grade_overall(fs, segs)
        low = [s for s in segs if s['score'] < self.cfg.score_threshold]
        return overall, segs, low, path

    def _grade_overall(self, fs, segs):
        """综合评分等级"""
        n = len(fs)
        ok_ratio = sum(1 for s in fs if s >= 60) / n
        bad_ratio = sum(1 for s in fs if s < 40) / n
        good_ratio = sum(1 for s in fs if s >= 85) / n

        fail_segs = [s for s in segs if s['score'] < PASS_SCORE]
        if fail_segs:
            print(f"  ⚠️ {len(fail_segs)}/{len(segs)}段不合格 (<{PASS_SCORE:.0f}分)")

        if ok_ratio < 0.6:
            final = max(3, np.mean(fs)*0.6 - bad_ratio*20)
            grade = "❌不合格"
        elif good_ratio >= 0.7 and bad_ratio < 0.03:
            final = min(100, np.mean(fs)+5)
            grade = "⭐优秀"
        elif ok_ratio >= 0.6 and bad_ratio < 0.12:
            final = np.mean(fs)
            grade = "👍良好"
        elif bad_ratio >= 0.25:
            final = max(3, 15 - bad_ratio*30)
            grade = "💪需重练"
        else:
            final = max(3, np.mean(fs) - bad_ratio*25)
            grade = "⚠️需改进"

        final = round(max(3, min(100, final)), 1)
        print(f"  总评: {grade} → {final:.1f}分")
        return final

    def _nonlinear_score(self, avg_diff):
        """非线性评分函数"""
        if avg_diff <= SCORE_TOLERANCE:
            return 100.0
        elif avg_diff <= SCORE_PENALTY_THRESHOLD:
            return 100.0 - (avg_diff-SCORE_TOLERANCE)*SCORE_PENALTY_SMALL
        else:
            base = 100.0 - (SCORE_PENALTY_THRESHOLD-SCORE_TOLERANCE)*SCORE_PENALTY_SMALL
            return max(3, base - (avg_diff-SCORE_PENALTY_THRESHOLD)*SCORE_PENALTY_LARGE)

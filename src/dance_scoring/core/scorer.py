# core/scorer.py — 舞蹈评分引擎

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Optional

from .config import (
    Config, DEFAULT_BPM, BEATS_PER_SEGMENT, SCORE_TOLERANCE, SCORE_PENALTY_SMALL,
    SCORE_PENALTY_LARGE, SCORE_PENALTY_THRESHOLD, DTW_WINDOW_RATIO, ANGLE_WEIGHTS,
    ANGLE_JOINTS, PASS_SCORE
)
from .dtw import dtw_constrained
from .alignment import fastdtw_alignment
from .segments import seg_by_beats
from .correction_provider import (
    CorrectionProvider, RuleBasedProvider,
    Deviation, SegmentInfo,
)

# MediaPipe 33 关键点 → 中文名（与 correction.py 保持同步）
_JOINT_NAMES_CN = {
    0: "鼻尖", 1: "左眼内角", 2: "左眼", 3: "左眼外角",
    4: "右眼内角", 5: "右眼", 6: "右眼外角",
    7: "左耳", 8: "右耳", 9: "嘴角左", 10: "嘴角右",
    11: "左肩", 12: "右肩", 13: "左肘", 14: "右肘",
    15: "左腕", 16: "右腕", 17: "左小指", 18: "右小指",
    19: "左食指", 20: "右食指", 21: "左拇指", 22: "右拇指",
    23: "左髋", 24: "右髋", 25: "左膝", 26: "右膝",
    27: "左踝", 28: "右踝", 29: "左脚跟", 30: "右脚跟",
    31: "左脚尖", 32: "右脚尖",
}

# 与 ANGLE_JOINTS 对应的关节索引 —— 用于确定偏差方向的主关节
# ANGLE_JOINTS 是 (a, b, c) 三元组，取中间关节点 b 作为偏差归属关节
_ANGLE_JOINT_INDICES = [b for _, b, _ in ANGLE_JOINTS]


class Scorer:
    def __init__(self, cfg: Config, bpm=DEFAULT_BPM, alignment_method: str = "dtw"):
        self.cfg = cfg
        self.bpm = bpm
        self.alignment_method = alignment_method
        self.spb = 60.0 / bpm
        self.sps = self.spb * BEATS_PER_SEGMENT
        self.frames_per_seg = int(self.sps * cfg.target_fps)

    def score(
        self,
        ref,
        user,
        progress_callback=None,
        correction_provider: Optional[CorrectionProvider] = None,
    ):
        """
        执行完整评分流程。

        参数:
            ref, user: PoseFrame 列表
            progress_callback: callable(pct, msg) 进度回调
            correction_provider: 纠正建议提供者，None 则使用 RuleBasedProvider

        返回:
            (overall, segs, low, path)
            segs 中每个段附带:
                - deviations: [Deviation] 关节偏差列表
                - correction_text: str 纠正文本
        """
        if correction_provider is None:
            correction_provider = RuleBasedProvider()

        nr, nu = len(ref), len(user)
        print(f"  参考:{nr}帧 用户:{nu}帧 BPM:{self.bpm}")

        if progress_callback: progress_callback(0, "DTW对齐...")
        print(f"  [1/3] 对齐 (方法: {self.alignment_method})...")
        ref_vec = np.array([p.vec for p in ref])
        user_vec = np.array([p.vec for p in user])

        if self.alignment_method == "fastdtw":
            radius = max(int(max(nr, nu) * DTW_WINDOW_RATIO), 1)
            cost, path = fastdtw_alignment(ref_vec, user_vec, radius=radius)
            print(f"  对齐:{len(path)}对 fastdtw radius:{radius}")
        else:
            mat = cdist(ref_vec, user_vec, metric='euclidean')
            window = max(int(max(nr, nu) * DTW_WINDOW_RATIO), 1)
            path, cost = dtw_constrained(mat, window)
            print(f"  对齐:{len(path)}对 窗口:{window}")

        if progress_callback: progress_callback(0, "逐帧评分...")
        print("  [2/3] 逐帧评分...")
        fs = []
        # 逐帧记录每个关节的原始角度差（未加权、未取绝对值，保留方向）
        n_angles = len(ANGLE_JOINTS)
        frame_joint_diffs = []  # List[np.ndarray of shape (n_angles,)]
        for ri, ui in path:
            raw_diffs = ref[ri].angles - user[ui].angles  # (n_angles,)
            frame_joint_diffs.append(raw_diffs)
            ang_diff = np.mean(np.abs(raw_diffs) * ANGLE_WEIGHTS)
            fs.append(self._nonlinear_score(ang_diff))

        if progress_callback: progress_callback(0, "分段评分...")
        print("  [3/3] 八拍分段评分...")
        segs = seg_by_beats(ref, path, fs, self.cfg.target_fps, self.bpm)

        # —— 逐段聚合关节偏差 ——
        ref_to_idx = {}
        for idx, (ri, ui) in enumerate(path):
            if ri not in ref_to_idx:
                ref_to_idx[ri] = idx

        for seg in segs:
            sf, ef = seg['ref_start'], seg['ref_end']
            seg_diffs = []
            for ri in range(sf, ef):
                if ri in ref_to_idx:
                    seg_diffs.append(frame_joint_diffs[ref_to_idx[ri]])

            if seg_diffs:
                mean_diffs = np.mean(np.stack(seg_diffs), axis=0)  # (n_angles,)
                deviations = []
                for j, (_, b_idx, _) in enumerate(ANGLE_JOINTS):
                    dev = float(mean_diffs[j])
                    if abs(dev) < 3.0:  # 忽略 < 3° 的偏差
                        continue
                    direction = "too_bent" if dev > 0 else "too_straight"
                    joint_name = _JOINT_NAMES_CN.get(b_idx, f"关节{b_idx}")
                    deviations.append(Deviation(
                        joint_name=joint_name,
                        joint_idx=b_idx,
                        deviation_deg=dev,
                        direction=direction,
                    ))
                deviations.sort(key=lambda d: abs(d.deviation_deg), reverse=True)
            else:
                deviations = []

            seg['deviations'] = deviations

        # 先计算总分，再生成纠正文本（总分是 Prompt 的上下文）
        overall = self._grade_overall(fs, segs)

        # —— 生成纠正文本 ——
        if progress_callback: progress_callback(0, "生成纠正建议...")
        for seg in segs:
            seg_info = SegmentInfo(
                id=seg['id'],
                score=seg['score'],
                qualified=seg['score'] >= PASS_SCORE,
                start_time=seg['start_time'],
                end_time=seg['end_time'],
                deviations=seg.get('deviations', []),
            )
            seg['correction_text'] = correction_provider.generate_correction(
                seg_info, overall_score=overall, bpm=self.bpm,
            )

        low = [s for s in segs if s['score'] < self.cfg.score_threshold]
        return overall, segs, low, path

    def _grade_overall(self, fs, segs):
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
        if avg_diff <= SCORE_TOLERANCE:
            return 100.0
        elif avg_diff <= SCORE_PENALTY_THRESHOLD:
            return 100.0 - (avg_diff-SCORE_TOLERANCE)*SCORE_PENALTY_SMALL
        else:
            base = 100.0 - (SCORE_PENALTY_THRESHOLD-SCORE_TOLERANCE)*SCORE_PENALTY_SMALL
            return max(3, base - (avg_diff-SCORE_PENALTY_THRESHOLD)*SCORE_PENALTY_LARGE)

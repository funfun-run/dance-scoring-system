# core/scorer.py — 舞蹈评分引擎

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Optional

from .config import (
    Config, DEFAULT_BPM, BEATS_PER_SEGMENT, SCORE_TOLERANCE, SCORE_PENALTY_SMALL,
    SCORE_PENALTY_LARGE, SCORE_PENALTY_THRESHOLD, DTW_WINDOW_RATIO, ANGLE_WEIGHTS,
    ANGLE_JOINTS, PASS_SCORE, VISIBILITY_THRESHOLD, MIN_VISIBLE_FRAME_RATIO,
    DANCE_EXCLUDED_JOINTS, COVERAGE_FLOOR, MOTION_FLOOR,
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
        # 逐帧记录每个关节的原始角度差 + 加权平均偏差（用于异常帧检测）
        n_angles = len(ANGLE_JOINTS)
        frame_ang_diffs = []   # 逐帧的加权平均偏差值
        frame_joint_diffs = []
        frame_visibility = []
        for ri, ui in path:
            raw_diffs = ref[ri].angles - user[ui].angles
            ref_vis = np.zeros(n_angles, dtype=bool)
            both_vis = np.zeros(n_angles, dtype=bool)
            for j, (a, b, c) in enumerate(ANGLE_JOINTS):
                if b in DANCE_EXCLUDED_JOINTS:
                    continue
                ref_ok = all(ref[ri].conf[idx] >= VISIBILITY_THRESHOLD
                            for idx in (a, b, c))
                if ref_ok:
                    ref_vis[j] = True
                    if all(user[ui].conf[idx] >= VISIBILITY_THRESHOLD
                           for idx in (a, b, c)):
                        both_vis[j] = True
            safe_diffs = raw_diffs.copy()
            safe_diffs[~both_vis] = np.nan
            frame_joint_diffs.append(safe_diffs)
            frame_visibility.append(both_vis)
            if both_vis.any():
                effective_weights = ANGLE_WEIGHTS.copy()
                effective_weights[~both_vis] = 0.0
                both_weight = effective_weights.sum()
                ang_diff = (np.nansum(np.abs(safe_diffs) * effective_weights) / both_weight
                            if both_weight > 0 else 0.0)
                ref_weight = float(sum(ANGLE_WEIGHTS[j] for j in range(n_angles)
                                      if ref_vis[j]))
                coverage = both_weight / max(ref_weight, 1e-6) if ref_weight > 0 else 1.0
                coverage_factor = COVERAGE_FLOOR + (1.0 - COVERAGE_FLOOR) * coverage
            else:
                ang_diff = 0.0
                coverage_factor = COVERAGE_FLOOR
            frame_ang_diffs.append(ang_diff)
            raw_score = self._nonlinear_score(ang_diff)
            fs.append(raw_score * coverage_factor)

        # —— 异常帧检测：片尾 logo / 非舞蹈内容 ——
        # 如果一帧的加权平均偏差超过全局中位数的 2 倍（且≥30°），标记为异常
        _diffs = np.array(frame_ang_diffs)
        _median_diff = float(np.median(_diffs[_diffs > 0])) if (_diffs > 0).any() else 10.0
        _outlier_threshold = max(_median_diff * 2.0, 30.0)
        outlier_count = 0
        for i in range(len(fs)):
            if frame_ang_diffs[i] > _outlier_threshold:
                fs[i] = np.nan
                outlier_count += 1
        if outlier_count > 0:
            print(f"  ⚡ 检测到 {outlier_count} 个异常帧（偏差>{_outlier_threshold:.0f}°），已自动排除")

        if progress_callback: progress_callback(0, "分段评分...")
        print("  [3/3] 八拍分段评分...")
        segs = seg_by_beats(ref, path, fs, self.cfg.target_fps, self.bpm)

        # —— 构建 ref 帧 → 路径位置的映射 ——
        ref_to_idx = {}
        for idx, (ri, ui) in enumerate(path):
            if ri not in ref_to_idx:
                ref_to_idx[ri] = idx

        # 重新计算段得分（排除 NaN 异常帧）
        for seg in segs:
            sf, ef = seg['ref_start'], seg['ref_end']
            seg_fs = []
            for ri in range(sf, ef):
                if ri in ref_to_idx:
                    seg_fs.append(fs[ref_to_idx[ri]])
            if seg_fs:
                valid = [s for s in seg_fs if not np.isnan(s)]
                seg['score'] = round(np.mean(valid), 1) if valid else 0.0

        # —— 逐段聚合关节偏差（同时排除异常帧） ——
        for seg in segs:
            sf, ef = seg['ref_start'], seg['ref_end']
            seg_diffs = []       # 偏差 (含 NaN)
            seg_visibilities = []  # 可见性 mask
            for ri in range(sf, ef):
                if ri in ref_to_idx:
                    idx = ref_to_idx[ri]
                    seg_diffs.append(frame_joint_diffs[idx])
                    seg_visibilities.append(frame_visibility[idx])

            if seg_diffs:
                n_frames = len(seg_diffs)
                deviations = []
                skipped_joints = set()  # 用 set 去重
                excluded_joints = set()  # 面部等舞蹈无关关节
                joint_visibility = {}

                for j, (_, b_idx, _) in enumerate(ANGLE_JOINTS):
                    joint_name = _JOINT_NAMES_CN.get(b_idx, f"关节{b_idx}")

                    # 面部关节（眼鼻耳嘴）永远排除，与舞蹈无关
                    if b_idx in DANCE_EXCLUDED_JOINTS:
                        excluded_joints.add(joint_name)
                        continue

                    # 提取所有帧中该角度的偏差值（含 NaN）
                    joint_diffs = np.array([fd[j] for fd in seg_diffs])
                    visible_count = int((~np.isnan(joint_diffs)).sum())
                    vis_ratio = visible_count / max(n_frames, 1)
                    joint_visibility[joint_name] = round(vis_ratio, 3)

                    if vis_ratio < MIN_VISIBLE_FRAME_RATIO:
                        skipped_joints.add(joint_name)
                        continue

                    mean_dev = float(np.nanmean(joint_diffs))
                    if abs(mean_dev) < 3.0:  # 忽略 < 3° 的偏差
                        continue
                    direction = "too_bent" if mean_dev > 0 else "too_straight"
                    deviations.append(Deviation(
                        joint_name=joint_name,
                        joint_idx=b_idx,
                        deviation_deg=mean_dev,
                        direction=direction,
                    ))
                # 同一关节可能关联多个角度，去重：保留偏差最大的
                best: dict = {}
                for d in deviations:
                    if d.joint_name not in best or abs(d.deviation_deg) > abs(best[d.joint_name].deviation_deg):
                        best[d.joint_name] = d
                deviations = sorted(best.values(), key=lambda d: abs(d.deviation_deg), reverse=True)
                # 合并：排除 + 不可见
                all_skipped = sorted(excluded_joints | skipped_joints)
            else:
                deviations = []
                all_skipped = []
                joint_visibility = {}

            seg['deviations'] = deviations
            seg['skipped_joints'] = all_skipped
            seg['joint_visibility'] = joint_visibility

        # —— 动作幅度惩罚：用户几乎不动则大幅扣分 ——
        # 计算对齐后 pose 向量的方差，衡量动作幅度
        ref_aligned = np.array([ref[ri].vec for ri, _ in path])
        user_aligned = np.array([user[ui].vec for _, ui in path])
        ref_var = float(np.var(ref_aligned))
        user_var = float(np.var(user_aligned))
        # motion_ratio: 0=完全静止, 1=和参考一样活跃, >1=比参考还活跃
        motion_ratio = user_var / max(ref_var, 1e-8)
        # 如果用户动作幅度 < 参考的 30%，开始扣分
        motion_factor = MOTION_FLOOR + (1.0 - MOTION_FLOOR) * min(motion_ratio / 0.3, 1.0)
        motion_factor = max(MOTION_FLOOR, min(1.0, motion_factor))

        # 对逐帧分和段分统一施加动作惩罚
        for i in range(len(fs)):
            fs[i] = fs[i] * motion_factor
        for seg in segs:
            seg['score'] = seg['score'] * motion_factor

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
                skipped_joints=seg.get('skipped_joints', []),
            )
            seg['correction_text'] = correction_provider.generate_correction(
                seg_info, overall_score=overall, bpm=self.bpm,
            )

        low = [s for s in segs if s['score'] < self.cfg.score_threshold]
        return overall, segs, low, path

    def _grade_overall(self, fs, segs):
        # 总体分 = 各段得分的加权平均（段长不一，用逐帧分做加权）
        if not segs:
            return 0.0

        # 按段计算平均分
        seg_scores = [s['score'] for s in segs]
        seg_avg = sum(seg_scores) / len(seg_scores)

        # 帧级统计用于评级
        n = len(fs)
        ok_ratio = sum(1 for s in fs if s >= 60) / max(n, 1)
        bad_ratio = sum(1 for s in fs if s < 40) / max(n, 1)
        good_ratio = sum(1 for s in fs if s >= 85) / max(n, 1)

        fail_segs = [s for s in segs if s['score'] < PASS_SCORE]
        if fail_segs:
            print(f"  ⚠️ {len(fail_segs)}/{len(segs)}段不合格 (<{PASS_SCORE:.0f}分)")

        # 总评 = 段均分 × 帧一致性修正
        if bad_ratio > 0.5:
            final = seg_avg * 0.7  # 超过半数帧不及格，扣30%
        elif bad_ratio > 0.3:
            final = seg_avg * 0.85  # 超过30%帧不及格，扣15%
        else:
            final = seg_avg

        if final >= 85:
            grade = "⭐优秀"
        elif final >= 70:
            grade = "👍良好"
        elif final >= 60:
            grade = "✅合格"
        elif final >= 40:
            grade = "⚠️需改进"
        else:
            grade = "💪需重练"

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

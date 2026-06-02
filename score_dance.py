# score_dance.py - AI推理层(NPU) + DTW评分 + 交互反馈
# 修复：每个视频独立创建PoseLandmarker，避免时间戳冲突

import cv2
import numpy as np
import os
import shutil
import time
import argparse
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import List

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    model_mgr, MEDIAPIPE_MODEL_PATH,
    TARGET_FPS, PASS_SCORE, SCORE_THRESHOLD, DTW_WINDOW_RATIO,
    SCORE_TOLERANCE, SCORE_PENALTY_SMALL, SCORE_PENALTY_LARGE, SCORE_PENALTY_THRESHOLD,
    ANGLE_JOINTS, ANGLE_WEIGHTS, OUTPUT_SEGMENTS_DIR, OUTPUT_LOW_SCORE_DIR,
    HAS_OPENVINO
)


@dataclass
class PoseFrame:
    fid: int
    kp3d: np.ndarray
    conf: np.ndarray
    angles: np.ndarray = None
    vec: np.ndarray = None
    
    def __post_init__(self):
        self.angles = self._calc_angles()
        self.vec = np.concatenate([self.kp3d[:,:2].flatten(), self.angles])
    
    def _calc_angles(self):
        kp = self.kp3d[:,:2]
        ang = []
        for a,b,c in ANGLE_JOINTS:
            ba, bc = kp[a]-kp[b], kp[c]-kp[b]
            cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
            ang.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
        return np.array(ang, dtype=np.float32)


def create_pose_detector():
    """创建新的PoseLandmarker实例（确保时间戳独立）"""
    model_mgr.download_model()
    
    # 尝试使用OpenVINO delegate加速
    try:
        base_opt = python.BaseOptions(
            model_asset_path=MEDIAPIPE_MODEL_PATH,
            delegate=python.BaseOptions.Delegate.GPU if HAS_OPENVINO else python.BaseOptions.Delegate.CPU
        )
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opt,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_tracking_confidence=0.5
        )
        return vision.PoseLandmarker.create_from_options(opts)
    except Exception as e:
        # 回退到CPU
        base_opt = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opt,
            running_mode=vision.RunningMode.VIDEO
        )
        return vision.PoseLandmarker.create_from_options(opts)


def extract_poses(video_path: str) -> List[PoseFrame]:
    """
    从视频提取姿态序列
    每次调用创建新的PoseLandmarker，保证时间戳独立
    """
    detector = create_pose_detector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(fps / TARGET_FPS))
    
    print(f"  提取: {os.path.basename(video_path)} ({fps:.0f}fps, {total}帧)")
    
    poses = []
    fid, proc = 0, 0
    times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if fid % skip == 0:
            t0 = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # 时间戳从0开始递增
            ts = proc * int(1000 / TARGET_FPS)
            res = detector.detect_for_video(mp_img, ts)
            times.append((time.time() - t0) * 1000)
            
            if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                kp = np.zeros((33, 3), dtype=np.float32)
                cf = np.zeros(33, dtype=np.float32)
                wlm = res.pose_world_landmarks[0]
                for i in range(min(33, len(wlm))):
                    kp[i] = [wlm[i].x, wlm[i].y, wlm[i].z * 0.3]
                    cf[i] = wlm[i].visibility if hasattr(wlm[i], 'visibility') else 1.0
                poses.append(PoseFrame(fid, kp, cf))
            
            proc += 1
        
        fid += 1
        if fid % 200 == 0:
            avg = np.mean(times[-100:]) if times else 0
            print(f"  进度:{100*fid//total}% 推理:{avg:.1f}ms")
    
    cap.release()
    if times:
        print(f"  提取:{len(poses)}帧 平均推理:{np.mean(times):.1f}ms")
    
    return _interpolate_poses(poses)


def _interpolate_poses(poses):
    """插值修复低置信度关键点"""
    if len(poses) < 2:
        return poses
    
    w = 3
    for i, p in enumerate(poses):
        mask = p.conf < 0.5
        if np.any(mask):
            pi, ni = max(0, i-w), min(len(poses)-1, i+w)
            for j in range(33):
                if mask[j]:
                    a = (i-pi) / max(1, ni-pi)
                    new_val = (1-a)*poses[pi].kp3d[j] + a*poses[ni].kp3d[j]
                    # 骨骼长度约束
                    if j in [13, 14, 25, 26]:
                        parent = j - 2
                        old_len = np.linalg.norm(p.kp3d[j] - p.kp3d[parent])
                        if old_len > 0 and abs(np.linalg.norm(new_val - p.kp3d[parent]) - old_len) / old_len < 0.2:
                            p.kp3d[j] = new_val
                    elif j in [15, 16, 27, 28]:
                        parent = j - 2
                        old_len = np.linalg.norm(p.kp3d[j] - p.kp3d[parent])
                        if old_len > 0 and abs(np.linalg.norm(new_val - p.kp3d[parent]) - old_len) / old_len < 0.3:
                            p.kp3d[j] = new_val
                    else:
                        p.kp3d[j] = new_val
            p.angles = p._calc_angles()
            p.vec = np.concatenate([p.kp3d[:,:2].flatten(), p.angles])
    
    return poses


class Scorer:
    """DTW评分器"""
    def __init__(self):
        self.w = np.array(ANGLE_WEIGHTS, dtype=np.float32)
    
    def score(self, ref, user):
        nr, nu = len(ref), len(user)
        mat = cdist(np.array([p.vec for p in ref]), np.array([p.vec for p in user]), 'euclidean')
        win = max(int(max(nr, nu) * DTW_WINDOW_RATIO), 1)
        path, cost = self._dtw(mat, win)
        fs = [self._sf(np.mean(np.abs(ref[ri].angles - user[ui].angles) * self.w)) for ri, ui in path]
        overall = self._grade(fs)
        return overall, fs, path
    
    def _dtw(self, mat, win):
        nr, nu = mat.shape
        cost = np.full((nr, nu), np.inf)
        cost[0, 0] = mat[0, 0]
        for i in range(1, nr):
            cost[i, 0] = cost[i-1, 0] + mat[i, 0]
        for j in range(1, nu):
            cost[0, j] = cost[0, j-1] + mat[0, j]
        for i in range(1, nr):
            for j in range(max(1, i-win), min(nu, i+win+1)):
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
                candidates = {(i-1, j-1): cost[i-1, j-1], (i-1, j): cost[i-1, j], (i, j-1): cost[i, j-1]}
                i, j = min(candidates, key=candidates.get)
        path.append((0, 0))
        path.reverse()
        return path, cost[-1, -1]
    
    def _sf(self, d):
        if d <= SCORE_TOLERANCE:
            return 100.0
        elif d <= SCORE_PENALTY_THRESHOLD:
            return 100.0 - (d - SCORE_TOLERANCE) * SCORE_PENALTY_SMALL
        else:
            b = 100.0 - (SCORE_PENALTY_THRESHOLD - SCORE_TOLERANCE) * SCORE_PENALTY_SMALL
            return max(3.0, b - (d - SCORE_PENALTY_THRESHOLD) * SCORE_PENALTY_LARGE)
    
    def _grade(self, fs):
        s = np.array(fs)
        raw = np.mean(s)
        ok = np.mean(s >= 60)
        bad = np.mean(s < 40)
        good = np.mean(s >= 85)
        if ok < 0.6:
            g = (max(3, raw*0.6 - bad*20), "❌不合格")
        elif good >= 0.7 and bad < 0.03:
            g = (min(100, raw+5), "⭐优秀")
        elif ok >= 0.6 and bad < 0.12:
            g = (raw, "👍良好")
        elif bad >= 0.25:
            g = (max(3, 15 - bad*30), "💪需重练")
        else:
            g = (max(3, raw - bad*25), "⚠️需改进")
        return round(float(g[0]), 1), g[1]


def segment_scores(segments, frame_scores, path):
    """按预分段给每段打分"""
    path_to_ref = {idx: ri for idx, (ri, ui) in enumerate(path)}
    for seg in segments:
        sf = int(seg['start'] * TARGET_FPS)
        ef = int(seg['end'] * TARGET_FPS)
        fs = [frame_scores[idx] for idx, ri in path_to_ref.items() if sf <= ri < ef and idx < len(frame_scores)]
        seg['score'] = round(np.mean(fs), 1) if fs else 0.0
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', default='videos/reference.mp4')
    parser.add_argument('-u', '--user', default='videos/user.mp4')
    parser.add_argument('-b', '--bpm', type=int, default=120, help='与split_8beats保持一致')
    parser.add_argument('-t', '--threshold', type=float, default=50.0)
    args = parser.parse_args()
    
    for p, n in [(args.reference, "参考"), (args.user, "用户")]:
        if not os.path.exists(p):
            print(f"❌ {n}视频不存在"); exit(1)
    
    print("\n" + "=" * 60)
    print("   🕺 舞蹈评分系统 | OpenVINO NPU")
    print("=" * 60)
    print(f"  BPM:{args.bpm} | 合格线:{PASS_SCORE} | 阈值:{args.threshold}")
    
    device_info = model_mgr.get_device_info()
    print(f"[推理] 设备: {device_info['target_device']}")
    
    # 1. 姿态提取（每个视频独立创建detector）
    print("\n[1/3] 提取姿态特征...")
    ref_poses = extract_poses(args.reference)
    user_poses = extract_poses(args.user)
    
    # 2. 评分
    print("\n[2/3] DTW对齐与评分...")
    scorer = Scorer()
    overall, frame_scores, path = scorer.score(ref_poses, user_poses)
    print(f"  总评: {overall[0]:.1f}/100 [{overall[1]}]")
    
    # 3. 分段评分（与split_8beats统一分段逻辑）
    print("\n[3/3] 分段评分与输出...")
    spb = 60.0 / args.bpm
    sps = spb * 8
    duration = len(ref_poses) / TARGET_FPS
    n_seg = max(1, int(duration / sps))
    if n_seg * sps < duration:
        n_seg += 1
    segments = [{'id': i+1, 'start': round(i*sps, 2), 'end': round(min((i+1)*sps, duration), 2)} for i in range(n_seg)]
    
    segments = segment_scores(segments, frame_scores, path)
    
    fail_segs = [s for s in segments if s['score'] < PASS_SCORE]
    low_segs = [s for s in segments if PASS_SCORE <= s['score'] < SCORE_THRESHOLD]
    
    print(f"\n{'段号':<6}{'时间':<16}{'得分':<10}{'判定'}")
    print("-" * 50)
    for s in segments:
        t = f"{s['start']:.1f}s-{s['end']:.1f}s"
        if s['score'] >= PASS_SCORE:
            q = "✅合格"
        elif s['score'] >= SCORE_THRESHOLD:
            q = "⚠️注意"
        else:
            q = "❌不合格"
        print(f"{s['id']:<6}{t:<16}{s['score']:<10.1f}{q}")
    
    # 输出练习片段
    if fail_segs or low_segs:
        target = fail_segs if fail_segs else low_segs
        os.makedirs(OUTPUT_LOW_SCORE_DIR, exist_ok=True)
        files = []
        for seg in target:
            src = os.path.join(OUTPUT_SEGMENTS_DIR, f"ref_seg_{seg['id']:02d}_slow.mp4")
            dst = os.path.join(OUTPUT_LOW_SCORE_DIR, f"practice_seg{seg['id']:02d}_score{seg['score']:.0f}_slow.mp4")
            if os.path.exists(src):
                shutil.copy(src, dst)
                files.append(dst)
                print(f"  ✓ {os.path.basename(dst)}")
            else:
                print(f"  ⚠️ 找不到: {src} (请先运行 split_8beats.py)")
        if files:
            print(f"已保存到 {OUTPUT_LOW_SCORE_DIR}/")
    else:
        print("\n🎉 全部合格！")
    
    print("=" * 60)
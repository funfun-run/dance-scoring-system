# score_dance.py  v6.2 - 统一分段逻辑 + 移除kp2d + 自包含兜底

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import List
import os
import shutil
import urllib.request
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================

BEATS_PER_SEGMENT = 8
DEFAULT_BPM = 120
SCORE_THRESHOLD = 50.0
PASS_SCORE = 60.0
SLOW_SPEED = 0.8
TARGET_FPS = 30
MIN_SEGMENT_DURATION = 0.3

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

ANGLE_JOINTS = [
    (1, 0, 4), (7, 0, 8), (7, 0, 11), (8, 0, 12),
    (11, 13, 15), (12, 14, 16), (13, 15, 17), (14, 16, 18),
    (13, 15, 19), (14, 16, 20), (15, 17, 19), (16, 18, 20),
    (13, 11, 23), (14, 12, 24), (11, 23, 24), (12, 24, 23),
    (11, 23, 25), (12, 24, 26), (23, 25, 27), (24, 26, 28),
    (25, 27, 29), (26, 28, 30), (25, 27, 31), (26, 28, 32),
    (27, 29, 31), (28, 30, 32),
]

ANGLE_WEIGHTS = np.array([1.0,1.0,0.8,0.8,1.3,1.3,0.6,0.6,0.5,0.5,
    0.4,0.4,1.2,1.2,1.4,1.4,1.3,1.3,1.5,1.5,0.8,0.8,0.7,0.7,0.6,0.6],
    dtype=np.float32)

SCORE_TOLERANCE = 3.0
SCORE_PENALTY_SMALL = 1.8
SCORE_PENALTY_LARGE = 3.0
SCORE_PENALTY_THRESHOLD = 15.0

DTW_WINDOW_RATIO = 0.1
Z_AXIS_WEIGHT = 0.3


@dataclass
class Config:
    score_threshold: float = 50.0
    target_fps: int = 30
    keypoint_confidence: float = 0.5
    interp_window: int = 3


@dataclass
class PoseFrame:
    fid: int
    kp3d: np.ndarray
    conf: np.ndarray
    angles: np.ndarray = None
    vec: np.ndarray = None
    
    def __post_init__(self):
        if self.conf is None:
            self.conf = np.ones(33, dtype=np.float32)
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


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[下载] 模型...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class PoseExtractor:
    def __init__(self, cfg: Config):
        base = python.BaseOptions(model_asset_path=MODEL_PATH)
        opt = vision.PoseLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.VIDEO
        )
        self.det = vision.PoseLandmarker.create_from_options(opt)
        self.cfg = cfg
    
    def extract(self, path: str) -> List[PoseFrame]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = max(1, int(fps/self.cfg.target_fps))
        print(f"  {os.path.basename(path)} | {fps:.0f}fps | {total}帧")
        
        poses, fid, proc = [], 0, 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if fid % skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = proc * int(1000/self.cfg.target_fps)
                res = self.det.detect_for_video(mp_img, ts)
                if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                    kp3d = np.zeros((33,3), dtype=np.float32)
                    cf = np.zeros(33, dtype=np.float32)
                    for i, lm in enumerate(res.pose_world_landmarks[0][:33]):
                        kp3d[i] = [lm.x, lm.y, lm.z*Z_AXIS_WEIGHT]
                        cf[i] = lm.visibility if hasattr(lm, 'visibility') else 1.0
                    poses.append(PoseFrame(fid, kp3d, cf))
                proc += 1
            fid += 1
            if fid % 200 == 0: print(f"  进度:{100*fid//total}%")
        cap.release()
        print(f"  提取:{len(poses)}帧")
        return self._interpolate(poses)
    
    def _interpolate(self, poses):
        if len(poses) < 2: return poses
        w = self.cfg.interp_window
        for i, p in enumerate(poses):
            mask = p.conf < self.cfg.keypoint_confidence
            if np.any(mask):
                pi, ni = max(0,i-w), min(len(poses)-1,i+w)
                for j in range(33):
                    if mask[j]:
                        a = (i-pi)/max(1, ni-pi)
                        new_val = (1-a)*poses[pi].kp3d[j] + a*poses[ni].kp3d[j]
                        if j in [13,14,25,26]:
                            parent=j-2
                            old_len=np.linalg.norm(p.kp3d[j]-p.kp3d[parent])
                            new_len=np.linalg.norm(new_val-p.kp3d[parent])
                            if old_len>0 and abs(new_len-old_len)/old_len<0.2: p.kp3d[j]=new_val
                        elif j in [15,16,27,28]:
                            parent=j-2
                            old_len=np.linalg.norm(p.kp3d[j]-p.kp3d[parent])
                            new_len=np.linalg.norm(new_val-p.kp3d[parent])
                            if old_len>0 and abs(new_len-old_len)/old_len<0.3: p.kp3d[j]=new_val
                        else: p.kp3d[j]=new_val
                p.angles=p._calc_angles()
                p.vec=np.concatenate([p.kp3d[:,:2].flatten(),p.angles])
        return poses


class Scorer:
    def __init__(self, cfg: Config, bpm=DEFAULT_BPM):
        self.cfg = cfg
        self.bpm = bpm
        # 计算每段对应的帧数（与split_8beats一致）
        self.spb = 60.0 / bpm
        self.sps = self.spb * BEATS_PER_SEGMENT
        self.frames_per_seg = int(self.sps * cfg.target_fps)
    
    def score(self, ref, user):
        nr, nu = len(ref), len(user)
        print(f"  参考:{nr}帧 用户:{nu}帧 BPM:{self.bpm}")
        
        print("  [1/3] DTW对齐...")
        ref_vec = np.array([p.vec for p in ref])
        user_vec = np.array([p.vec for p in user])
        mat = cdist(ref_vec, user_vec, metric='euclidean')
        window = max(int(max(nr, nu)*DTW_WINDOW_RATIO), 1)
        path, cost = self._dtw_constrained(mat, window)
        print(f"  对齐:{len(path)}对 窗口:{window}")
        
        print("  [2/3] 逐帧评分...")
        fs = []
        for ri, ui in path:
            ang_diff = np.mean(np.abs(ref[ri].angles-user[ui].angles)*ANGLE_WEIGHTS)
            fs.append(self._nonlinear_score(ang_diff))
        
        print("  [3/3] 八拍分段评分(与split_8beats统一)...")
        segs = self._seg_by_beats(ref, path, fs)
        overall = self._grade_overall(fs, segs)
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
    
    def _dtw_constrained(self, mat, window):
        nr, nu = mat.shape
        cost = np.full((nr, nu), np.inf)
        cost[0,0] = mat[0,0]
        for i in range(1, nr): cost[i,0] = cost[i-1,0] + mat[i,0]
        for j in range(1, nu): cost[0,j] = cost[0,j-1] + mat[0,j]
        for i in range(1, nr):
            for j in range(max(1,i-window), min(nu,i+window+1)):
                cost[i,j] = mat[i,j] + min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1])
        path = []; i,j = nr-1, nu-1
        while i > 0 or j > 0:
            path.append((i,j))
            if i==0: j-=1
            elif j==0: i-=1
            else:
                cand = {(i-1,j-1):cost[i-1,j-1], (i-1,j):cost[i-1,j], (i,j-1):cost[i,j-1]}
                i,j = min(cand, key=cand.get)
        path.append((0,0)); path.reverse()
        return path, cost[-1,-1]
    
    def _nonlinear_score(self, avg_diff):
        if avg_diff <= SCORE_TOLERANCE: return 100.0
        elif avg_diff <= SCORE_PENALTY_THRESHOLD:
            return 100.0 - (avg_diff-SCORE_TOLERANCE)*SCORE_PENALTY_SMALL
        else:
            base = 100.0 - (SCORE_PENALTY_THRESHOLD-SCORE_TOLERANCE)*SCORE_PENALTY_SMALL
            return max(3, base - (avg_diff-SCORE_PENALTY_THRESHOLD)*SCORE_PENALTY_LARGE)
    
    def _seg_by_beats(self, ref, path, fs):
        """
        【N2修复】与split_8beats统一的分段逻辑
        按固定BPM/节拍分段，段号与split_8beats输出文件一致
        """
        total_ref = len(ref)
        fps = self.frames_per_seg
        if fps <= 0:
            fps = total_ref
        
        nseg = max(1, (total_ref + fps - 1) // fps)
        
        # 构建对齐映射 ref_idx -> path_idx
        ref_to_path = {}
        for idx, (ri, ui) in enumerate(path):
            if ri not in ref_to_path:
                ref_to_path[ri] = idx
        
        segs = []
        for sid in range(nseg):
            sf = sid * fps
            ef = min((sid+1) * fps, total_ref)
            
            # 收集该段内所有对齐帧的分数
            seg_fs = []
            for ri in range(sf, ef):
                if ri in ref_to_path:
                    seg_fs.append(fs[ref_to_path[ri]])
            
            if seg_fs:
                ss = round(np.mean(seg_fs), 1)
            else:
                ss = 0.0
            
            qualified = "合格" if ss >= PASS_SCORE else "不合格"
            
            segs.append({
                'id': sid + 1,
                'ref_start': sf,
                'ref_end': ef,
                'start_time': round(sf/self.cfg.target_fps, 2),
                'end_time': round(ef/self.cfg.target_fps, 2),
                'score': ss,
                'qualified': qualified
            })
        
        return segs
    
    def extract_clips_from_segments(self, segs, segments_dir="output/segments", out_dir="output/low_score_clips"):
        """
        从 split_8beats 生成的慢动作片段中复制低分段落
        段号统一，直接按 id 匹配
        """
        fail_segs = [s for s in segs if s['score'] < PASS_SCORE]
        if not fail_segs:
            fail_segs = [s for s in segs if s['score'] < self.cfg.score_threshold]
        if not fail_segs:
            return []
        
        # 检查 segments 目录是否存在
        if not os.path.isdir(segments_dir):
            print(f"  ⚠️ 未找到分段目录 {segments_dir}，从参考视频实时提取...")
            return self._extract_fallback(fail_segs, out_dir)
        
        existing = [f for f in os.listdir(segments_dir) if f.startswith('ref_seg_')]
        if not existing:
            print(f"  ⚠️ {segments_dir} 为空，从参考视频实时提取...")
            return self._extract_fallback(fail_segs, out_dir)
        
        # 复制匹配的慢动作片段
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for seg in fail_segs:
            src = os.path.join(segments_dir, f"ref_seg_{seg['id']:02d}_slow.mp4")
            dst = os.path.join(out_dir, f"practice_seg{seg['id']:02d}_score{seg['score']:.0f}_slow.mp4")
            if os.path.exists(src):
                shutil.copy(src, dst)
                files.append(dst)
                print(f"    ✓ {os.path.basename(dst)}")
            else:
                print(f"    ⚠️ 找不到: {os.path.basename(src)}，实时提取...")
                # 兜底：实时提取这个段
                self._extract_single_clip(seg, dst)
                if os.path.exists(dst):
                    files.append(dst)
        return files
    
    def _extract_fallback(self, segs, out_dir):
        """兜底：从参考视频实时提取"""
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for seg in segs:
            dst = os.path.join(out_dir, f"practice_seg{seg['id']:02d}_score{seg['score']:.0f}_slow.mp4")
            self._extract_single_clip(seg, dst)
            if os.path.exists(dst):
                files.append(dst)
                print(f"    ✓ {os.path.basename(dst)} (实时提取)")
        return files
    
    def _extract_single_clip(self, seg, output_path, ref_video=None):
        """从参考视频实时提取单个慢动作片段"""
        if ref_video is None:
            # 尝试找默认路径
            for p in ['videos/reference.mp4', 'videos/ref.mp4']:
                if os.path.exists(p):
                    ref_video = p
                    break
        if ref_video is None or not os.path.exists(ref_video):
            print(f"    ✗ 无法找到参考视频，跳过段{seg['id']}")
            return
        
        cap = cv2.VideoCapture(ref_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(3)), int(cap.get(4))
        repeat = max(1, int(1/SLOW_SPEED))
        
        sf = int(seg['ref_start'] * fps / self.cfg.target_fps)
        ef = int(seg['ref_end'] * fps / self.cfg.target_fps)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (w,h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        for _ in range(sf, ef):
            ret, frame = cap.read()
            if not ret: break
            cv2.putText(frame, f"Seg{seg['id']} Score:{seg['score']:.0f}", (30,40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            for _ in range(repeat): out.write(frame)
        out.release(); cap.release()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--reference', default='videos/reference.mp4')
    parser.add_argument('-u','--user', default='videos/user.mp4')
    parser.add_argument('-b','--bpm', type=int, default=DEFAULT_BPM, help='BPM（需与split_8beats一致）')
    parser.add_argument('-t','--threshold', type=float, default=50.0)
    parser.add_argument('-s','--segments', default='output/segments', help='split_8beats输出目录')
    args = parser.parse_args()
    
    for p,n in [(args.reference,"参考"),(args.user,"用户")]:
        if not os.path.exists(p): print(f"❌ {n}视频不存在"); exit(1)
    
    print("\n"+"="*60)
    print("   🕺 舞蹈评分 v6.2 | 统一分段逻辑")
    print("="*60)
    print(f"  BPM:{args.bpm} | 每段{BEATS_PER_SEGMENT}拍={60/args.bpm*BEATS_PER_SEGMENT:.1f}秒")
    print(f"  分段与split_8beats完全一致，段号对应")
    print(f"  合格线:{PASS_SCORE:.0f}分 | 阈值:{args.threshold}分")
    
    download_model()
    cfg = Config(score_threshold=args.threshold)
    
    print("\n[1/3] 提取参考...")
    ref = PoseExtractor(cfg).extract(args.reference)
    print("\n[2/3] 提取用户...")
    user = PoseExtractor(cfg).extract(args.user)
    print("\n[3/3] 评分...")
    scorer = Scorer(cfg, bpm=args.bpm)
    overall, segs, low, path = scorer.score(ref, user)
    
    fail_segs = [s for s in segs if s['score'] < PASS_SCORE]
    
    print("\n"+"="*60)
    print(f"      总评: {overall:.1f}/100")
    if overall>=90: print("      ⭐优秀")
    elif overall>=78: print("      👍良好")
    elif overall>=60: print("      📝还行")
    elif overall>=35: print("      ⚠️需改进")
    else: print("      💪需重练")
    if fail_segs: print(f"      ⚠️ {len(fail_segs)}/{len(segs)}段不合格")
    print("="*60)
    
    print(f"\n{'段号':<6}{'时间':<16}{'得分':<10}{'判定'}")
    print("-"*50)
    for s in segs:
        t=f"{s['start_time']:.1f}s-{s['end_time']:.1f}s"
        q="✅合格" if s['score']>=PASS_SCORE else "❌不合格"
        print(f"{s['id']:<6}{t:<16}{s['score']:<10.1f}{q}")
    
    if fail_segs:
        print(f"\n❌ {len(fail_segs)}段不合格，输出慢动作视频:")
        for s in fail_segs:
            print(f"  第{s['id']}段 [{s['start_time']:.1f}s-{s['end_time']:.1f}s] {s['score']:.1f}分")
        files = scorer.extract_clips_from_segments(segs, args.segments)
        if files: print(f"已保存到 output/low_score_clips/")
    elif low:
        print(f"\n⚠️ 全部合格，但以下片段低于{args.threshold}分:")
        for s in low:
            print(f"  第{s['id']}段 [{s['start_time']:.1f}s-{s['end_time']:.1f}s] {s['score']:.1f}分")
        files = scorer.extract_clips_from_segments(segs, args.segments)
        if files: print(f"已保存到 output/low_score_clips/")
    else:
        print(f"\n🎉 全部合格！")
    print("="*60)
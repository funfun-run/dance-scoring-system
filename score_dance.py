# score_dance.py  v5.10 - 33关节 + 每段60分合格线 + 复用8拍分段

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
SCORE_THRESHOLD = 50.0
PASS_SCORE = 60.0          # 每段合格线
SLOW_SPEED = 0.8
TARGET_FPS = 30

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

KP_WEIGHTS = np.array([1.0,0.5,0.5,0.5,0.5,0.5,0.5,0.8,0.8,0.3,0.3,
    1.5,1.5,1.2,1.2,1.0,1.0,0.6,0.6,0.6,0.6,0.3,0.3,1.5,1.5,
    1.3,1.3,1.0,1.0,0.6,0.6,0.5,0.5], dtype=np.float32)

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
    kp2d: np.ndarray = None
    conf: np.ndarray = None
    angles: np.ndarray = None
    vec: np.ndarray = None
    
    def __post_init__(self):
        if self.conf is None: self.conf = np.ones(33,dtype=np.float32)
        self.angles = self._calc_angles()
        kp = self.kp2d if self.kp2d is not None else self.kp3d[:,:2]
        self.vec = np.concatenate([kp.flatten(), self.angles])
    
    def _calc_angles(self):
        kp = self.kp2d if self.kp2d is not None else self.kp3d[:,:2]
        angles = []
        for a,b,c in ANGLE_JOINTS:
            ba,bc = kp[a]-kp[b], kp[c]-kp[b]
            cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
        return np.array(angles,dtype=np.float32)


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[下载] 模型...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class PoseExtractor:
    def __init__(self, cfg: Config):
        base = python.BaseOptions(model_asset_path=MODEL_PATH)
        opt = vision.PoseLandmarkerOptions(base_options=base,running_mode=vision.RunningMode.VIDEO)
        self.det = vision.PoseLandmarker.create_from_options(opt)
        self.cfg = cfg
    
    def extract(self, path: str) -> List[PoseFrame]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = max(1,int(fps/self.cfg.target_fps))
        print(f"  {os.path.basename(path)} | {fps:.0f}fps | {total}帧")
        poses,fid,proc = [],0,0
        while True:
            ret,frame = cap.read()
            if not ret: break
            if fid%skip==0:
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                ts = proc*int(1000/self.cfg.target_fps)
                res = self.det.detect_for_video(img,ts)
                if res.pose_world_landmarks and len(res.pose_world_landmarks)>0:
                    kp3d = np.zeros((33,3),dtype=np.float32)
                    kp2d = np.zeros((33,2),dtype=np.float32)
                    cf = np.zeros(33,dtype=np.float32)
                    wlm = res.pose_world_landmarks[0]
                    for i in range(min(33,len(wlm))):
                        kp3d[i]=[wlm[i].x,wlm[i].y,wlm[i].z*Z_AXIS_WEIGHT]
                        cf[i]=wlm[i].visibility if hasattr(wlm[i],'visibility') else 1.0
                    if res.pose_landmarks and len(res.pose_landmarks)>0:
                        ilm=res.pose_landmarks[0]
                        for i in range(min(33,len(ilm))): kp2d[i]=[ilm[i].x,ilm[i].y]
                    poses.append(PoseFrame(fid,kp3d,kp2d,cf))
                proc+=1
            fid+=1
            if fid%200==0: print(f"  进度:{100*fid//total}%")
        cap.release()
        print(f"  提取:{len(poses)}帧")
        return self._fix(poses)
    
    def _fix(self, poses):
        if len(poses)<2: return poses
        w=self.cfg.interp_window
        for i,p in enumerate(poses):
            mask=p.conf<self.cfg.keypoint_confidence
            if np.any(mask):
                pi,ni=max(0,i-w),min(len(poses)-1,i+w)
                for j in range(33):
                    if mask[j]:
                        a=(i-pi)/max(1,ni-pi)
                        new_val=(1-a)*poses[pi].kp3d[j]+a*poses[ni].kp3d[j]
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
                p.kp2d=p.kp3d[:,:2].copy()
                p.angles=p._calc_angles()
                p.vec=np.concatenate([p.kp2d.flatten(),p.angles])
        return poses


class Scorer:
    def __init__(self, cfg: Config): self.cfg=cfg
    
    def score(self, ref, user):
        nr,nu = len(ref),len(user)
        print(f"  参考:{nr}帧 用户:{nu}帧")
        
        print("  [1/4] DTW对齐...")
        ref_vec=np.array([p.vec for p in ref])
        user_vec=np.array([p.vec for p in user])
        mat=cdist(ref_vec,user_vec,metric='euclidean')
        window=max(int(max(nr,nu)*DTW_WINDOW_RATIO),1)
        path,cost=self._dtw_constrained(mat,window)
        print(f"  对齐:{len(path)}对 窗口:{window}")
        
        print("  [2/4] 动作变化检测...")
        cp=self._changes_adaptive(ref)
        print(f"  检测到{len(cp)-1}个段落")
        
        print("  [3/4] 逐帧评分...")
        fs=[]
        for ri,ui in path:
            kp_diff=np.abs(ref[ri].kp2d.flatten()-user[ui].kp2d.flatten())
            kp_score=np.mean(kp_diff.reshape(-1,2).mean(axis=1)*KP_WEIGHTS)
            ang_diff=np.abs(ref[ri].angles-user[ui].angles)
            ang_score=np.mean(ang_diff*ANGLE_WEIGHTS)
            total_diff=0.4*kp_score+0.6*ang_score
            fs.append(self._nonlinear_score(total_diff))
        
        print("  [4/4] 分段评分(合格线60分)...")
        segs=self._seg_adaptive(ref,path,fs,cp)
        overall = self._grade_overall(fs, segs)
        low=[s for s in segs if s['score']<self.cfg.score_threshold]
        return overall,segs,low,path
    
    def _grade_overall(self, fs, segs):
        """总评：基于各段分数"""
        n = len(fs)
        
        perfect = sum(1 for s in fs if s >= 95) / n
        excellent = sum(1 for s in fs if 85 <= s < 95) / n
        good = sum(1 for s in fs if 75 <= s < 85) / n
        ok = sum(1 for s in fs if 60 <= s < 75) / n
        poor = sum(1 for s in fs if 40 <= s < 60) / n
        bad = sum(1 for s in fs if 20 <= s < 40) / n
        terrible = sum(1 for s in fs if s < 20) / n
        
        print(f"  完美(≥95):{perfect:.0%} 优秀(85-94):{excellent:.0%} 良好(75-84):{good:.0%}")
        print(f"  一般(60-74):{ok:.0%} 较差(40-59):{poor:.0%} 差(20-39):{bad:.0%} 极差(<20):{terrible:.0%}")
        
        # 统计不合格段数
        fail_segs = [s for s in segs if s['score'] < PASS_SCORE]
        fail_count = len(fail_segs)
        
        if fail_count > 0:
            print(f"  ⚠️ {fail_count}/{len(segs)} 段不合格 (<{PASS_SCORE:.0f}分)")
        
        if ok + good + excellent + perfect < 0.6:
            final = max(3, np.mean(fs) * 0.6 - bad*20 - terrible*30)
            grade = "❌不合格"
        elif good + excellent + perfect >= 0.7 and bad + terrible < 0.03:
            final = 90 + perfect*8
            grade = "⭐优秀"
        elif ok + good + excellent + perfect >= 0.6 and poor + bad + terrible < 0.12:
            final = 78 + ok*6 + good*8
            grade = "👍良好"
        elif bad + terrible >= 0.25:
            final = max(3, 15 - bad*30 - terrible*50)
            grade = "💪需重练"
        elif bad + terrible >= 0.15:
            final = 45 + ok*8 + good*4 - poor*15 - bad*30 - terrible*45
            grade = "⚠️需改进"
        else:
            final = np.mean(fs)
            grade = "📝一般"
        
        final = round(max(3, min(100, final)), 1)
        print(f"  总评: {grade} → {final:.1f}分")
        return final
    
    def _dtw_constrained(self, mat, window):
        nr,nu=mat.shape
        cost=np.full((nr,nu),np.inf)
        cost[0,0]=mat[0,0]
        for i in range(1,nr):
            for j in range(max(1,i-window),min(nu,i+window+1)):
                cost[i,j]=mat[i,j]+min(cost[i-1,j] if j<=i+window else np.inf,
                    cost[i,j-1] if j-1>=i-window else np.inf,
                    cost[i-1,j-1] if abs(i-1-(j-1))<=window else np.inf)
            if i<=window: cost[i,0]=cost[i-1,0]+mat[i,0]
        for j in range(1,nu):
            if j<=window: cost[0,j]=cost[0,j-1]+mat[0,j]
        path=[]; i,j=nr-1,nu-1
        while i>0 or j>0:
            path.append((i,j))
            if i==0: j-=1
            elif j==0: i-=1
            else:
                cand={}
                if i>0 and j>0 and abs(i-1-(j-1))<=window: cand[(i-1,j-1)]=cost[i-1,j-1]
                if i>0 and abs(i-1-j)<=window: cand[(i-1,j)]=cost[i-1,j]
                if j>0 and abs(i-(j-1))<=window: cand[(i,j-1)]=cost[i,j-1]
                if cand: i,j=min(cand,key=cand.get)
                else: i-=1; j-=1
        path.append((0,0)); path.reverse()
        return path,cost[-1,-1]
    
    def _nonlinear_score(self, avg_diff):
        if avg_diff<=SCORE_TOLERANCE: return 100.0
        elif avg_diff<=SCORE_PENALTY_THRESHOLD:
            return 100.0-(avg_diff-SCORE_TOLERANCE)*SCORE_PENALTY_SMALL
        else:
            base=100.0-(SCORE_PENALTY_THRESHOLD-SCORE_TOLERANCE)*SCORE_PENALTY_SMALL
            extra=(avg_diff-SCORE_PENALTY_THRESHOLD)*SCORE_PENALTY_LARGE
            return max(3,base-extra)
    
    def _changes_adaptive(self, poses):
        if len(poses)<3: return [0,len(poses)-1]
        diffs=[]; w=max(2,len(poses)//30)
        for i in range(w,len(poses)):
            local=[np.mean(np.abs(poses[j].vec-poses[j-1].vec)) for j in range(i-w,i)]
            diffs.append(np.std(local) if local else 0)
        diffs=np.array(diffs)
        if len(diffs)==0: return [0,len(poses)-1]
        th=np.median(diffs)+np.std(diffs)*1.0
        peaks=[0]
        for i in range(1,len(diffs)-1):
            if diffs[i]>th and diffs[i]>diffs[i-1] and diffs[i]>diffs[i+1]: peaks.append(i+w)
        peaks.append(len(poses)-1)
        min_sep=int(TARGET_FPS*0.5)
        merged=[peaks[0]]
        for p in peaks[1:]:
            if p-merged[-1]>=min_sep: merged.append(p)
        if merged[-1]!=peaks[-1]: merged[-1]=peaks[-1]
        return merged
    
    def _seg_adaptive(self, ref, path, fs, cp):
        sb=[]
        for c in cp:
            found=False
            for idx,(ri,ui) in enumerate(path):
                if ri>=c: sb.append(idx); found=True; break
            if not found and sb: sb.append(sb[-1])
        if len(sb)<2:
            total=len(path); sz=max(1,total//BEATS_PER_SEGMENT)
            sb=[i*sz for i in range(BEATS_PER_SEGMENT)]; sb.append(total)
        segs=[]
        for i in range(len(sb)-1):
            st,ed=sb[i],sb[i+1] if i+1<len(sb) else len(path)
            if st>=len(fs) or ed>len(fs) or st>=ed: continue
            rs=path[st][0] if st<len(path) else 0
            re=path[min(ed-1,len(path)-1)][0] if ed>0 else 0
            if (re-rs)/self.cfg.target_fps<0.3: continue
            ss=round(np.mean(fs[st:ed]),1)
            # 每段判定是否合格
            qualified = "合格" if ss >= PASS_SCORE else "不合格"
            segs.append({'id':len(segs)+1,'ref_start':rs,'ref_end':re,
                        'start_time':round(rs/self.cfg.target_fps,2),
                        'end_time':round(re/self.cfg.target_fps,2),
                        'score':ss,'qualified':qualified})
        return segs
    
    def save_low_clips(self, segs, segments_dir="output/segments", out_dir="output/low_score_clips"):
        """复制已分好的8拍慢动作片段"""
        if not segs:
            return []
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for seg in segs:
            seg_id = seg['id']
            src = os.path.join(segments_dir, f"ref_seg_{seg_id:02d}_slow.mp4")
            dst = os.path.join(out_dir, f"practice_seg{seg_id:02d}_score{seg['score']:.0f}_slow.mp4")
            if os.path.exists(src):
                shutil.copy(src, dst)
                files.append(dst)
                print(f"    ✓ {os.path.basename(dst)}")
            else:
                print(f"    ⚠️ 找不到: {src}")
        return files


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-r','--reference',default='videos/reference.mp4')
    parser.add_argument('-u','--user',default='videos/user.mp4')
    parser.add_argument('-t','--threshold',type=float,default=50.0)
    parser.add_argument('-s','--segments',default='output/segments',help='8拍分段目录')
    args=parser.parse_args()
    
    for p,n in [(args.reference,"参考"),(args.user,"用户")]:
        if not os.path.exists(p): print(f"❌ {n}视频不存在"); exit(1)
    
    print("\n"+"="*60)
    print("   🕺 舞蹈评分 v5.10 | 每段60分合格线")
    print("="*60)
    print(f"  容忍度:3° | 扣分:1.8/3.0 | 加速:15°")
    print(f"  每段合格线: {PASS_SCORE:.0f}分 | 低分输出阈值: {args.threshold}分")
    
    download_model()
    cfg=Config(score_threshold=args.threshold)
    
    print("\n[1/3] 提取参考...")
    ref=PoseExtractor(cfg).extract(args.reference)
    print("\n[2/3] 提取用户...")
    user=PoseExtractor(cfg).extract(args.user)
    print("\n[3/3] 评分...")
    scorer=Scorer(cfg)
    overall,segs,low,path=scorer.score(ref,user)
    
    # 统计不合格段
    fail_segs = [s for s in segs if s['score'] < PASS_SCORE]
    
    print("\n"+"="*60)
    print(f"      总评: {overall:.1f}/100")
    if overall>=90: print("      ⭐优秀")
    elif overall>=78: print("      👍良好")
    elif overall>=60: print("      📝还行")
    elif overall>=35: print("      ⚠️需改进")
    else: print("      💪需重练")
    if fail_segs:
        print(f"      ⚠️ {len(fail_segs)}/{len(segs)} 段不合格 (<{PASS_SCORE:.0f}分)")
    print("="*60)
    
    print(f"\n{'段号':<6}{'时间':<16}{'得分':<10}{'判定'}")
    print("-"*50)
    for s in segs:
        t=f"{s['start_time']:.1f}s-{s['end_time']:.1f}s"
        q = "✅合格" if s['score'] >= PASS_SCORE else "❌不合格"
        print(f"{s['id']:<6}{t:<16}{s['score']:<10.1f}{q}")
    
    # 输出逻辑
    if fail_segs:
        # 有不合格段 → 输出不合格段
        print(f"\n❌ {len(fail_segs)}段不合格，输出对应慢动作视频:")
        for s in fail_segs:
            print(f"  第{s['id']}段 [{s['start_time']:.1f}s-{s['end_time']:.1f}s] {s['score']:.1f}分")
        files = scorer.save_low_clips(fail_segs, args.segments)
        print(f"已保存到 output/low_score_clips/")
    elif low:
        print(f"\n⚠️ 全部合格，但以下片段低于 {args.threshold} 分:")
        for s in low:
            print(f"  第{s['id']}段 [{s['start_time']:.1f}s-{s['end_time']:.1f}s] {s['score']:.1f}分")
        files = scorer.save_low_clips(low, args.segments)
        print(f"已保存到 output/low_score_clips/")
    else:
        print(f"\n🎉 全部合格且无低分片段！")
    print("="*60)
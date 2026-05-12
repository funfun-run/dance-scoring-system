# score_dance.py
# 舞蹈评分系统 | 关节角度对比 | 动作变化切段 | 适中评分
# 输入：videos/reference.mp4 + videos/user.mp4
# 输出：终端评分报告 + output/low_score_clips/*.mp4

import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial.distance import euclidean
from dataclasses import dataclass
from typing import List
import os
import urllib.request
import warnings
warnings.filterwarnings('ignore')

BEATS_PER_SEGMENT = 8
SCORE_THRESHOLD = 50.0
SLOW_SPEED = 0.8
TARGET_FPS = 30

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

ANGLE_JOINTS = [
    (11, 13, 15), (12, 14, 16), (13, 15, 17), (14, 16, 18),
    (23, 25, 27), (24, 26, 28), (25, 27, 31), (26, 28, 32),
    (11, 23, 25), (12, 24, 26), (13, 11, 23), (14, 12, 24),
]

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
        self.angles = self._calc_angles()
        if self.vec is None: self.vec = self.angles
    
    def _calc_angles(self):
        angles = []
        for a, b, c in ANGLE_JOINTS:
            ba, bc = self.kp3d[a]-self.kp3d[b], self.kp3d[c]-self.kp3d[b]
            cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
            angles.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
        return np.array(angles, dtype=np.float32)


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[下载] 模型...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class PoseExtractor:
    def __init__(self, cfg: Config):
        base = python.BaseOptions(model_asset_path=MODEL_PATH)
        opt = vision.PoseLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO)
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
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = proc * int(1000/self.cfg.target_fps)
                res = self.det.detect_for_video(img, ts)
                if res.pose_world_landmarks and len(res.pose_world_landmarks)>0:
                    kp = np.zeros((33,3), dtype=np.float32)
                    cf = np.zeros(33, dtype=np.float32)
                    for i, lm in enumerate(res.pose_world_landmarks[0][:33]):
                        kp[i]=[lm.x,lm.y,lm.z]
                        cf[i]=lm.visibility if hasattr(lm,'visibility') else 1.0
                    poses.append(PoseFrame(fid,kp,cf))
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
                        p.kp3d[j]=(1-a)*poses[pi].kp3d[j]+a*poses[ni].kp3d[j]
                p.angles=p._calc_angles()
                p.vec=p.angles
        return poses


class Scorer:
    def __init__(self, cfg: Config): self.cfg=cfg
    
    def score(self, ref, user):
        nr, nu = len(ref), len(user)
        print(f"  参考:{nr}帧 用户:{nu}帧")
        print("  [1/4] DTW对齐...")
        mat=np.zeros((nr,nu))
        for i in range(nr):
            for j in range(nu): mat[i,j]=np.mean(np.abs(ref[i].angles-user[j].angles))
        path,cost=self._dtw(mat)
        print(f"  对齐:{len(path)}对")
        print("  [2/4] 检测动作变化...")
        cp=self._changes(ref)
        print(f"  分段:{len(cp)-1}段")
        print("  [3/4] 逐帧评分...")
        fs=[]
        for ri,ui in path:
            diff=np.mean(np.abs(ref[ri].angles-user[ui].angles))
            s=100-diff*1.0
            s=max(5,min(100,s))
            fs.append(s)
        print("  [4/4] 分段评分...")
        overall=round(np.mean(fs),1)
        segs=self._seg(ref,path,fs,cp)
        low=[s for s in segs if s['score']<self.cfg.score_threshold]
        return overall,segs,low,path
    
    def _dtw(self, mat):
        nr,nu=mat.shape
        cost=np.full((nr,nu),np.inf)
        cost[0,0]=mat[0,0]
        for i in range(1,nr): cost[i,0]=cost[i-1,0]+mat[i,0]
        for j in range(1,nu): cost[0,j]=cost[0,j-1]+mat[0,j]
        for i in range(1,nr):
            for j in range(1,nu):
                cost[i,j]=mat[i,j]+min(cost[i-1,j],cost[i,j-1],cost[i-1,j-1])
        path=[]; i,j=nr-1,nu-1
        while i>0 or j>0:
            path.append((i,j))
            if i==0: j-=1
            elif j==0: i-=1
            else:
                m=min(cost[i-1,j],cost[i,j-1],cost[i-1,j-1])
                if m==cost[i-1,j-1]: i-=1; j-=1
                elif m==cost[i-1,j]: i-=1
                else: j-=1
        path.append((0,0)); path.reverse()
        return path,cost[-1,-1]
    
    def _changes(self, poses):
        if len(poses)<3: return [0,len(poses)-1]
        diffs=[]
        for i in range(1,len(poses)):
            diffs.append(np.mean(np.abs(poses[i].angles-poses[i-1].angles)))
        diffs=np.array(diffs)
        th=np.mean(diffs)+np.std(diffs)*0.5
        peaks=[0]
        for i in range(1,len(diffs)):
            if diffs[i]>th: peaks.append(i)
        peaks.append(len(poses)-1)
        if len(peaks)>BEATS_PER_SEGMENT+1:
            pv=[(p,diffs[p] if p<len(diffs) else 0) for p in peaks[1:-1]]
            pv.sort(key=lambda x:x[1],reverse=True)
            return [0]+sorted([p for p,v in pv[:BEATS_PER_SEGMENT-1]])+[len(poses)-1]
        return peaks
    
    def _seg(self, ref, path, fs, cp):
        sb=[]
        for c in cp:
            for idx,(ri,ui) in enumerate(path):
                if ri>=c: sb.append(idx); break
        if len(sb)<2:
            total=len(path); sz=total//BEATS_PER_SEGMENT
            sb=[i*sz for i in range(BEATS_PER_SEGMENT)]; sb.append(total)
        segs=[]
        for i in range(len(sb)-1):
            st,ed=sb[i],sb[i+1] if i+1<len(sb) else len(path)
            if st>=len(fs) or ed>len(fs) or st>=ed: continue
            ss=round(np.mean(fs[st:ed]),1)
            rs=path[st][0] if st<len(path) else 0
            re=path[min(ed-1,len(path)-1)][0] if ed>0 else 0
            segs.append({'id':i+1,'ref_start':rs,'ref_end':re,
                        'start_time':round(rs/self.cfg.target_fps,2),
                        'end_time':round(re/self.cfg.target_fps,2),'score':ss})
        while len(segs)<BEATS_PER_SEGMENT:
            total=len(path); sz=total//BEATS_PER_SEGMENT; segs=[]
            for i in range(BEATS_PER_SEGMENT):
                st,ed=i*sz,(i+1)*sz if i<BEATS_PER_SEGMENT-1 else total
                ss=round(np.mean(fs[st:ed]),1) if st<len(fs) else 0
                rs=path[st][0] if st<len(path) else 0
                re=path[min(ed-1,len(path)-1)][0] if ed>0 else 0
                segs.append({'id':i+1,'ref_start':rs,'ref_end':re,
                            'start_time':round(rs/self.cfg.target_fps,2),
                            'end_time':round(re/self.cfg.target_fps,2),'score':ss})
            break
        return segs[:BEATS_PER_SEGMENT]
    
    def save_low_clips_slow(self, segs, ref_video, out_dir="output/low_score_clips"):
        low=[s for s in segs if s['score']<self.cfg.score_threshold]
        if not low: return []
        os.makedirs(out_dir,exist_ok=True)
        cap=cv2.VideoCapture(ref_video)
        fps=cap.get(cv2.CAP_PROP_FPS)
        w,h=int(cap.get(3)),int(cap.get(4))
        repeat=max(1,int(1/SLOW_SPEED))
        files=[]
        for seg in low:
            sf=int(seg['ref_start']*fps/self.cfg.target_fps)
            ef=int(seg['ref_end']*fps/self.cfg.target_fps)
            op=os.path.join(out_dir,f"practice_seg{seg['id']:02d}_score{seg['score']:.0f}_slow.mp4")
            out=cv2.VideoWriter(op,cv2.VideoWriter_fourcc(*'mp4v'),TARGET_FPS,(w,h))
            cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
            for _ in range(sf,ef):
                ret,frame=cap.read()
                if not ret: break
                cv2.putText(frame,f"Seg{seg['id']} Score:{seg['score']:.0f}",(30,40),
                           cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                for _ in range(repeat): out.write(frame)
            out.release()
            files.append(op)
            print(f"    {os.path.basename(op)}")
        cap.release()
        return files


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-r','--reference',default='videos/reference.mp4')
    parser.add_argument('-u','--user',default='videos/user.mp4')
    parser.add_argument('-t','--threshold',type=float,default=50.0)
    args=parser.parse_args()
    
    for p,n in [(args.reference,"参考"),(args.user,"用户")]:
        if not os.path.exists(p): print(f"❌ {n}视频不存在"); exit(1)
    
    print("\n"+"="*60)
    print("   🕺 舞蹈评分 | 关节角度对比 | 适中评分")
    print("="*60)
    print(f"  阈值:{args.threshold}分 | 慢动作:{SLOW_SPEED}x")
    print(f"  评分: 角度差1度≈扣1分")
    
    download_model()
    cfg=Config(score_threshold=args.threshold)
    
    print("\n[1/3] 提取参考...")
    ref=PoseExtractor(cfg).extract(args.reference)
    print("\n[2/3] 提取用户...")
    user=PoseExtractor(cfg).extract(args.user)
    print("\n[3/3] 评分...")
    scorer=Scorer(cfg)
    overall,segs,low,path=scorer.score(ref,user)
    
    print("\n"+"="*60)
    print(f"      总评: {overall:.1f}/100")
    if overall>=90: print("      ⭐优秀")
    elif overall>=75: print("      👍良好")
    elif overall>=60: print("      📝还行")
    elif overall>=40: print("      ⚠️需改进")
    else: print("      💪多练习")
    print("="*60)
    
    print(f"\n{'段号':<6}{'时间':<16}{'得分':<10}{'状态'}")
    print("-"*45)
    for s in segs:
        t=f"{s['start_time']:.1f}s-{s['end_time']:.1f}s"
        st="✅" if s['score']>=args.threshold else "⚠️"
        print(f"{s['id']:<6}{t:<16}{s['score']:<10.1f}{st}")
    
    if low:
        print(f"\n⚠️ 低于{args.threshold}分: {len(low)}段")
        files=scorer.save_low_clips_slow(segs,args.reference)
        print(f"慢动作片段已保存到 output/low_score_clips/")
    else:
        print("\n✅ 全部通过")
    print("="*60)
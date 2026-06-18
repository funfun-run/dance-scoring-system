# core/extractor.py — MediaPipe 姿态提取

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
from typing import List

from .config import Config, MODEL_URL, MODEL_PATH, Z_AXIS_WEIGHT, TARGET_FPS
from .frame import PoseFrame


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
        self._closed = False

    def close(self):
        """显式关闭底层 MediaPipe Landmarker，释放 C++ 资源。

        必须在创建 PoseExtractor 的同一线程调用，避免 llvmpipe 崩溃。
        """
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self.det, 'close'):
                self.det.close()
        except Exception as e:
            print(f"[WARN] PoseExtractor.close() 失败: {e}")

    def __del__(self):
        """析构保护：尝试关闭但不应依赖此机制（__del__ 在 llvmpipe 上不可靠）。"""
        try:
            self.close()
        except Exception:
            pass

    def extract(self, path: str, progress_callback=None) -> List[PoseFrame]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = max(1, int(fps / self.cfg.target_fps))
        print(f"  {os.path.basename(path)} | {fps:.0f}fps | {total}帧")

        poses, fid, proc = [], 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fid % skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = proc * int(1000 / self.cfg.target_fps)
                res = self.det.detect_for_video(mp_img, ts)
                if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                    kp3d = np.zeros((33, 3), dtype=np.float32)
                    cf = np.zeros(33, dtype=np.float32)
                    for i, lm in enumerate(res.pose_world_landmarks[0][:33]):
                        kp3d[i] = [lm.x, lm.y, lm.z * Z_AXIS_WEIGHT]
                        cf[i] = lm.visibility if hasattr(lm, 'visibility') else 1.0
                    poses.append(PoseFrame(fid, kp3d, cf))
                proc += 1
            fid += 1
            if fid % 200 == 0:
                print(f"  进度:{100*fid//total}%")
            if progress_callback and fid % 30 == 0:
                progress_callback(int(100*fid//total))
        cap.release()
        print(f"  提取:{len(poses)}帧")
        return self._interpolate(poses)

    def _interpolate(self, poses):
        if len(poses) < 2:
            return poses
        w = self.cfg.interp_window
        for i, p in enumerate(poses):
            mask = p.conf < self.cfg.keypoint_confidence
            if np.any(mask):
                pi, ni = max(0, i-w), min(len(poses)-1, i+w)
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

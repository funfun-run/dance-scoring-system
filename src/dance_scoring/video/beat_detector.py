# video/beat_detector.py — 节拍检测（音频 + 运动光流）

import cv2
import numpy as np
import os
import subprocess

from dance_scoring.core.config import DEFAULT_BPM

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def _extract_audio(video_path: str, audio_path: str = "temp_audio.wav"):
    """从视频中提取音频"""
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video_path,
                       '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
                       audio_path], check=True)
        return audio_path
    except Exception:
        return None


def detect_beats_from_audio(video_path: str):
    """音频节拍检测，返回 (beat_times, tempo) 或 None"""
    if not HAS_LIBROSA:
        return None
    audio_path = None
    try:
        audio_path = _extract_audio(video_path)
        if audio_path is None or not os.path.exists(audio_path):
            return None
        y, sr = librosa.load(audio_path, sr=22050)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if tempo < 60 or tempo > 180:
            print(f"  检测到BPM={tempo:.1f}，超出合理范围(60-180)，拒绝使用")
            return None
        if len(beat_frames) < 2:
            return None
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print(f"  检测到BPM: {tempo:.1f}，节拍数: {len(beat_times)}")
        return beat_times.tolist(), tempo
    except Exception as e:
        print(f"  音频分析失败: {e}")
        return None
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def detect_beats_from_motion(video_path: str):
    """运动光流节拍检测，返回 (beat_times, estimated_bpm) 或 None"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total < 30:
        cap.release()
        return None

    skip = max(1, int(fps / 15))
    rx, ry, rw, rh = int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6)

    prev_gray = None
    motion_scores, frame_times = [], []
    fid = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if fid % skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_roi = gray[ry:ry+rh, rx:rx+rw]
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))
                motion_scores.append(mag)
                frame_times.append(fid/fps)
            prev_gray = gray_roi
        fid += 1
    cap.release()

    if len(motion_scores) < 10:
        return None

    motion_scores = np.array(motion_scores)
    threshold = np.mean(motion_scores) + np.std(motion_scores)*0.5
    peaks = []
    for i in range(1, len(motion_scores)-1):
        if motion_scores[i] > threshold and motion_scores[i] > motion_scores[i-1] and motion_scores[i] > motion_scores[i+1]:
            peaks.append(frame_times[i])

    if len(peaks) >= 3:
        intervals = np.diff(peaks)
        avg_interval = np.median(intervals)
        if avg_interval > 0.1:
            est_bpm = 60/avg_interval
            if 60 <= est_bpm <= 180:
                print(f"  运动检测BPM: {est_bpm:.1f}")
                return peaks, est_bpm
    return None

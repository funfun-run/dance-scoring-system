# split_8beats.py - 感知采集层 + 数据处理层 + 慢动作输出
# 功能：视频读取、音频节拍检测、八拍分段、生成慢动作视频

import cv2
import numpy as np
import os
import subprocess
import warnings
warnings.filterwarnings('ignore')

from config import (
    BEATS_PER_SEGMENT, DEFAULT_BPM, TARGET_FPS, SLOW_SPEED, OUTPUT_SEGMENTS_DIR
)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class VideoReader:
    """感知采集：读取视频信息与帧"""
    def __init__(self, path):
        self.path = path
        self.cap = None
    
    def open(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开: {self.path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {'fps': fps, 'frames': frames, 'duration': frames/fps, 'width': w, 'height': h}
    
    def read_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        if self.cap: self.cap.release()


class BeatSegmenter:
    """数据处理：节拍检测与八拍分段"""
    def __init__(self, bpm_hint=DEFAULT_BPM):
        self.bpm_hint = bpm_hint
    
    def extract_audio(self, video_path):
        out = "temp_audio.wav"
        try:
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',video_path,
                           '-vn','-acodec','pcm_s16le','-ar','22050','-ac','1',out],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out
        except:
            return None
    
    def process(self, video_path, duration):
        segments = None; bpm = self.bpm_hint; method = f"固定BPM={self.bpm_hint}"
        
        if HAS_LIBROSA:
            audio_path = self.extract_audio(video_path)
            if audio_path and os.path.exists(audio_path):
                try:
                    y, sr = librosa.load(audio_path, sr=22050)
                    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                    if 60 <= tempo <= 180 and len(beat_frames) >= BEATS_PER_SEGMENT:
                        beats = librosa.frames_to_time(beat_frames, sr=sr)
                        bpm = float(tempo)
                        method = f"音频节拍 (BPM={bpm:.1f})"
                        segments = self._build_segments(beats, duration)
                except Exception as e:
                    print(f"  音频分析失败: {e}")
                finally:
                    try: os.remove(audio_path)
                    except: pass
        
        if segments is None:
            print(f"  回退到固定BPM={self.bpm_hint}")
            segments = self._fixed_segments(duration)
        return segments, bpm, method
    
    def _build_segments(self, beats, duration):
        segs = []; sid=1; i=0
        while i+BEATS_PER_SEGMENT <= len(beats):
            st, et = beats[i], beats[i+BEATS_PER_SEGMENT-1]
            if et-st >= 0.5:
                segs.append({'id':sid, 'start':round(st,2), 'end':round(min(et,duration),2)})
                sid+=1
            i+=BEATS_PER_SEGMENT
        if i < len(beats) and duration-beats[i] >= 0.5:
            segs.append({'id':sid, 'start':round(beats[i],2), 'end':round(duration,2)})
        return segs
    
    def _fixed_segments(self, duration):
        spb = 60.0/self.bpm_hint; sps = spb*BEATS_PER_SEGMENT
        n = max(1, int(duration/sps))
        if n*sps < duration: n+=1
        return [{'id':i+1, 'start':round(i*sps,2), 'end':round(min((i+1)*sps,duration),2)} for i in range(n)]


def generate_slow_clips(video_path, segments):
    """生成慢动作视频片段"""
    os.makedirs(OUTPUT_SEGMENTS_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(3)); h=int(cap.get(4))
    repeat = max(1, int(1/SLOW_SPEED))
    clips = []
    for seg in segments:
        sf, ef = int(seg['start']*fps), int(seg['end']*fps)
        out_path = os.path.join(OUTPUT_SEGMENTS_DIR, f"ref_seg_{seg['id']:02d}_slow.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (w,h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        for _ in range(sf, ef):
            ret, frm = cap.read()
            if not ret: break
            for _ in range(repeat): out.write(frm)
        out.release()
        clips.append(out_path)
    cap.release()
    return clips


def merge_clips(clip_list, out_path):
    if not clip_list: return
    # 尺寸统一
    cap = cv2.VideoCapture(clip_list[0])
    tw, th = int(cap.get(3)), int(cap.get(4))
    cap.release()
    temp = []
    for i, v in enumerate(clip_list):
        c = cv2.VideoCapture(v); w=int(c.get(3)); h=int(c.get(4)); c.release()
        if w!=tw or h!=th:
            tp = f"temp_scale_{i}.mp4"
            c = cv2.VideoCapture(v); fps_v = c.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(tp, cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (tw,th))
            while True:
                ret, frm = c.read()
                if not ret: break
                out.write(cv2.resize(frm, (tw,th)))
            c.release(); out.release()
            temp.append(tp)
        else:
            temp.append(v)
    list_file = 'temp_list.txt'
    with open(list_file, 'w', encoding='utf-8') as f:
        for v in temp: f.write(f"file '{os.path.abspath(v)}'\n")
    try:
        subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0',
                       '-i',list_file,'-c','copy',out_path], check=True)
    except:
        cap0 = cv2.VideoCapture(temp[0]); fps0 = cap0.get(cv2.CAP_PROP_FPS); cap0.release()
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps0, (tw,th))
        for v in temp:
            c = cv2.VideoCapture(v)
            while True:
                ret, frm = c.read()
                if not ret: break
                out.write(frm)
            c.release()
        out.release()
    if os.path.exists(list_file): os.remove(list_file)
    for f in temp:
        if f.startswith("temp_scale_"): os.remove(f) if os.path.exists(f) else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--reference', default='videos/reference.mp4')
    parser.add_argument('-b','--bpm', type=int, default=DEFAULT_BPM)
    args = parser.parse_args()
    
    if not os.path.exists(args.reference):
        print(f"❌ 视频不存在: {args.reference}"); exit(1)
    
    print("\n"+"="*60)
    print("   🎬 八拍慢动作分段")
    print("="*60)
    
    reader = VideoReader(args.reference)
    info = reader.open()
    print(f"  视频: {info['duration']:.1f}秒, {info['fps']:.0f}fps")
    
    segmenter = BeatSegmenter(args.bpm)
    segments, bpm, method = segmenter.process(args.reference, info['duration'])
    print(f"\n  方法: {method}")
    for seg in segments:
        print(f"    第{seg['id']:2d}段: {seg['start']:6.2f}s - {seg['end']:6.2f}s")
    
    print(f"\n[生成] 慢动作 ({SLOW_SPEED}x)...")
    clips = generate_slow_clips(args.reference, segments)
    merged = os.path.join(OUTPUT_SEGMENTS_DIR, "all_segments_merged.mp4")
    merge_clips(clips, merged)
    
    reader.release()
    print(f"✅ 完成 → {OUTPUT_SEGMENTS_DIR}/")
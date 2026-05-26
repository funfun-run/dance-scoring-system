# split_8beats.py  v2.3 - 修复音频泄漏 + BPM校验 + 尺寸自动缩放

import cv2
import numpy as np
import os
import subprocess
import warnings
warnings.filterwarnings('ignore')

BEATS_PER_SEGMENT = 8
SLOW_SPEED = 0.8
TARGET_FPS = 30
MIN_SEGMENT_DURATION = 0.5
DEFAULT_BPM = 120

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[提示] pip install librosa 可启用音频节拍检测")


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frames, duration, w, h


def extract_audio(video_path, audio_path="temp_audio.wav"):
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video_path,
                       '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
                       audio_path], check=True)
        return audio_path
    except:
        return None


def detect_beats_from_audio(video_path):
    if not HAS_LIBROSA:
        return None
    audio_path = None
    try:
        audio_path = extract_audio(video_path)
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


def detect_beats_from_motion(video_path):
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


def get_beat_segments(beat_times, duration, beats_per_seg=8):
    if beat_times is None or len(beat_times) < beats_per_seg:
        return None
    segments = []
    seg_id, i = 1, 0
    while i + beats_per_seg <= len(beat_times):
        st, et = beat_times[i], beat_times[min(i+beats_per_seg, len(beat_times)-1)]
        if et - st > MIN_SEGMENT_DURATION:
            segments.append({'id': seg_id, 'start': round(st, 2),
                           'end': round(min(et, duration), 2)})
            seg_id += 1
        i += beats_per_seg
    if i < len(beat_times) and duration - beat_times[i] > MIN_SEGMENT_DURATION:
        segments.append({'id': seg_id, 'start': round(beat_times[i], 2),
                        'end': round(duration, 2)})
    return segments


def calculate_segments_fixed(duration_seconds, bpm):
    spb = 60/bpm
    sps = spb * BEATS_PER_SEGMENT
    num = max(1, int(duration_seconds/sps))
    if num*sps < duration_seconds:
        num += 1
    segments = []
    for i in range(num):
        segments.append({'id': i+1, 'start': round(i*sps, 2),
                        'end': round(min((i+1)*sps, duration_seconds), 2)})
    return segments


def extract_slow_segment(video_path, start_time, end_time, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sf = int(start_time*fps)
    ef = int(end_time*fps)
    repeat = max(1, int(1/SLOW_SPEED))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    for _ in range(sf, ef):
        ret, frame = cap.read()
        if not ret: break
        for _ in range(repeat): out.write(frame)
    cap.release(); out.release()


def merge_videos(video_list, output_path):
    if not video_list:
        return
    
    # 统一目标尺寸（取第一个视频的尺寸）
    cap = cv2.VideoCapture(video_list[0])
    target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # 先检查是否需要缩放，统一处理
    temp_files = []
    need_cleanup = False
    for i, v in enumerate(video_list):
        cap = cv2.VideoCapture(v)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w != target_w or h != target_h:
            print(f"    缩放片段 {os.path.basename(v)}: {w}x{h} → {target_w}x{target_h}")
            temp_path = f"temp_scaled_{i}.mp4"
            cap = cv2.VideoCapture(v)
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps_val, (target_w, target_h))
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(cv2.resize(frame, (target_w, target_h)))
            cap.release(); out.release()
            temp_files.append(temp_path)
            need_cleanup = True
        else:
            temp_files.append(v)
    
    list_file = 'temp_list.txt'
    with open(list_file, 'w', encoding='utf-8') as f:
        for v in temp_files:
            f.write(f"file '{os.path.abspath(v)}'\n")
    
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat',
                       '-safe', '0', '-i', list_file, '-c', 'copy', output_path], check=True)
    except:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             target_fps_val, (target_w, target_h))
        for vpath in temp_files:
            cap = cv2.VideoCapture(vpath)
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
            cap.release()
        out.release()
    
    if os.path.exists(list_file):
        os.remove(list_file)
    if need_cleanup:
        for i in range(len(video_list)):
            tp = f"temp_scaled_{i}.mp4"
            if os.path.exists(tp): os.remove(tp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', default='videos/reference.mp4')
    parser.add_argument('-b', '--bpm', type=int, default=DEFAULT_BPM)
    parser.add_argument('-o', '--output', default='output/segments')
    args = parser.parse_args()
    
    if not os.path.exists(args.reference):
        print(f"❌ 视频不存在: {args.reference}"); exit(1)
    
    print("\n" + "="*60)
    print("   🎬 8拍慢动作分段 v2.3")
    print("="*60)
    
    fps, frames, duration, w, h = get_video_info(args.reference)
    print(f"  视频: {duration:.1f}秒 | {fps:.0f}fps | {w}x{h}")
    
    segments = None
    used_method = f"固定BPM={args.bpm}"
    final_bpm = args.bpm
    
    print("\n[检测] 音频节拍...")
    result = detect_beats_from_audio(args.reference)
    if result is not None:
        beat_times, tempo = result
        segments = get_beat_segments(beat_times, duration, BEATS_PER_SEGMENT)
        if segments:
            used_method = f"音频节拍 (BPM={tempo:.1f})"
            final_bpm = tempo
    
    if segments is None:
        print("  尝试运动检测...")
        result = detect_beats_from_motion(args.reference)
        if result is not None:
            beat_times, tempo = result
            segments = get_beat_segments(beat_times, duration, BEATS_PER_SEGMENT)
            if segments:
                used_method = f"运动检测 (BPM≈{tempo:.0f})"
                final_bpm = tempo
    
    if segments is None:
        print(f"  使用固定BPM={args.bpm}")
        segments = calculate_segments_fixed(duration, args.bpm)
        final_bpm = args.bpm
    
    print(f"\n  分段方式: {used_method}")
    print(f"  共 {len(segments)} 段:")
    for seg in segments:
        print(f"    第{seg['id']:2d}段: {seg['start']:6.2f}s - {seg['end']:6.2f}s")
    
    print(f"\n[生成] 慢动作 ({SLOW_SPEED}x)...")
    os.makedirs(args.output, exist_ok=True)
    all_clips = []
    for seg in segments:
        sid = seg['id']
        out_path = os.path.join(args.output, f"ref_seg_{sid:02d}_slow.mp4")
        extract_slow_segment(args.reference, seg['start'], seg['end'], out_path)
        all_clips.append(out_path)
        print(f"    ✓ {os.path.basename(out_path)}")
    
    merged = os.path.join(args.output, "all_segments_merged.mp4")
    print(f"\n[合并] -> {merged}")
    merge_videos(all_clips, merged)
    
    print(f"\n✅ 完成！{len(segments)}段 → {args.output}/")
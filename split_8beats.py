# split_8beats.py  v2.0 - 音频节拍驱动版
# 改进：librosa音频节拍检测 | 自适应分段 | 保留固定BPM回退
# 输入：videos/reference.mp4
# 输出：output/segments/ 所有8拍慢动作片段

import cv2
import numpy as np
import os
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================

BPM = 120                      # 默认BPM（音频检测失败时使用）
BEATS_PER_SEGMENT = 8          # 每段8拍
SLOW_SPEED = 0.8               # 慢动作倍速
TARGET_FPS = 30                # 输出帧率

SECONDS_PER_BEAT = 60 / BPM
SECONDS_PER_SEGMENT = SECONDS_PER_BEAT * BEATS_PER_SEGMENT

# 尝试导入librosa
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[提示] librosa未安装，将使用固定BPM。安装命令: pip install librosa")


# ==================== 工具函数 ====================

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """从视频提取音频"""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1', audio_path
        ], check=True)
        return audio_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def detect_beats_from_audio(video_path):
    """
    【P2-优化】从音频检测节拍时间点
    返回: beat_times列表（秒），或None
    """
    if not HAS_LIBROSA:
        return None
    
    audio_path = extract_audio(video_path)
    if audio_path is None or not os.path.exists(audio_path):
        return None
    
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        if len(beat_frames) < 2:
            return None
        
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # 清理临时音频
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        print(f"  检测到BPM: {tempo:.1f}")
        print(f"  节拍数: {len(beat_times)}")
        return beat_times, tempo
    
    except Exception as e:
        print(f"  音频分析失败: {e}，回退到固定BPM")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None


def detect_beats_from_motion(video_path):
    """
    【P2-优化】从运动周期性检测节拍（无音频时回退方案）
    使用光流变化检测动作节奏
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total < 30:
        cap.release()
        return None
    
    # 降采样以加速
    skip = max(1, int(fps / 15))
    
    prev_gray = None
    motion_scores = []
    frame_times = []
    fid = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if fid % skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                motion_scores.append(mag)
                frame_times.append(fid / fps)
            prev_gray = gray
        fid += 1
    
    cap.release()
    
    if len(motion_scores) < 10:
        return None
    
    motion_scores = np.array(motion_scores)
    # 找运动峰值之间的间距作为节拍周期
    threshold = np.mean(motion_scores) + np.std(motion_scores) * 0.5
    peaks = []
    for i in range(1, len(motion_scores)-1):
        if motion_scores[i] > threshold and motion_scores[i] > motion_scores[i-1] and motion_scores[i] > motion_scores[i+1]:
            peaks.append(frame_times[i])
    
    if len(peaks) >= 3:
        # 估算BPM
        intervals = np.diff(peaks)
        avg_interval = np.median(intervals)
        if avg_interval > 0.1:
            est_bpm = 60 / avg_interval
            print(f"  运动检测BPM: {est_bpm:.1f}")
            return peaks, est_bpm
    
    return None


def get_beat_segments(beat_times, duration, beats_per_seg=8):
    """
    根据节拍时间点生成分段
    每段固定N拍（如8拍）
    """
    if beat_times is None or len(beat_times) < beats_per_seg:
        return None
    
    segments = []
    seg_id = 1
    i = 0
    
    while i + beats_per_seg <= len(beat_times):
        start_time = beat_times[i]
        end_idx = min(i + beats_per_seg, len(beat_times) - 1)
        end_time = beat_times[end_idx]
        
        if end_time - start_time > 0.5:  # 至少0.5秒
            segments.append({
                'id': seg_id,
                'start': round(start_time, 2),
                'end': round(min(end_time, duration), 2),
                'beat_start': i + 1,
                'beat_end': end_idx + 1
            })
            seg_id += 1
        
        i += beats_per_seg
    
    # 处理最后不足8拍的尾巴
    if i < len(beat_times) and i > 0:
        start_time = beat_times[i]
        if duration - start_time > 0.5:
            segments.append({
                'id': seg_id,
                'start': round(start_time, 2),
                'end': round(duration, 2),
                'beat_start': i + 1,
                'beat_end': len(beat_times)
            })
    
    return segments


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


def calculate_segments_fixed(duration_seconds, bpm):
    """回退方案：固定BPM分段"""
    spb = 60 / bpm
    sps = spb * BEATS_PER_SEGMENT
    num = max(1, int(duration_seconds / sps))
    if num * sps < duration_seconds:
        num += 1
    
    segments = []
    for i in range(num):
        segments.append({
            'id': i + 1,
            'start': round(i * sps, 2),
            'end': round(min((i+1) * sps, duration_seconds), 2),
            'beat_start': i * BEATS_PER_SEGMENT + 1,
            'beat_end': min((i+1) * BEATS_PER_SEGMENT, int(duration_seconds / spb))
        })
    return segments


def extract_slow_segment(video_path, start_time, end_time, output_path, label=""):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    sf = int(start_time * fps)
    ef = int(end_time * fps)
    repeat = max(1, int(1 / SLOW_SPEED))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (w, h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    for _ in range(sf, ef):
        ret, frame = cap.read()
        if not ret: break
        if label:
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        for _ in range(repeat): out.write(frame)
    
    cap.release()
    out.release()


def merge_videos(video_list, output_path):
    if not video_list: return
    list_file = 'temp_list.txt'
    with open(list_file, 'w', encoding='utf-8') as f:
        for v in video_list:
            f.write(f"file '{os.path.abspath(v)}'\n")
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error',
                       '-f', 'concat', '-safe', '0',
                       '-i', list_file, '-c', 'copy', output_path], check=True)
    except:
        cap = cv2.VideoCapture(video_list[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))
        for vpath in video_list:
            cap = cv2.VideoCapture(vpath)
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
            cap.release()
        out.release()
    if os.path.exists(list_file):
        os.remove(list_file)


# ==================== 主程序 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='8拍慢动作分段 - 音频节拍驱动')
    parser.add_argument('-r', '--reference', default='videos/reference.mp4')
    parser.add_argument('-b', '--bpm', type=int, default=120, help='默认BPM（音频检测失败时使用）')
    parser.add_argument('-o', '--output', default='output/segments')
    args = parser.parse_args()
    
    if not os.path.exists(args.reference):
        print(f"❌ 视频不存在: {args.reference}")
        exit(1)
    
    print("\n" + "=" * 60)
    print("   🎬 8拍慢动作分段 | 音频节拍驱动")
    print("=" * 60)
    
    # 获取视频信息
    fps, frames, duration, w, h = get_video_info(args.reference)
    print(f"  视频: {duration:.1f}秒 | {fps:.0f}fps | {frames}帧")
    
    # 【P2-优化】优先用音频节拍检测
    print("\n[检测] 分析音频节拍...")
    segments = None
    used_method = "固定BPM"
    final_bpm = args.bpm
    
    # 方案1：音频节拍
    result = detect_beats_from_audio(args.reference)
    if result is not None:
        beat_times, tempo = result
        segments = get_beat_segments(beat_times, duration, BEATS_PER_SEGMENT)
        if segments:
            used_method = f"音频节拍 (BPM={tempo:.1f})"
            final_bpm = tempo
    
    # 方案2：运动检测
    if segments is None:
        print("  音频检测失败，尝试运动周期性检测...")
        result = detect_beats_from_motion(args.reference)
        if result is not None:
            beat_times, tempo = result
            segments = get_beat_segments(beat_times, duration, BEATS_PER_SEGMENT)
            if segments:
                used_method = f"运动周期性 (BPM≈{tempo:.0f})"
                final_bpm = tempo
    
    # 方案3：固定BPM回退
    if segments is None:
        print(f"  使用固定BPM={args.bpm}分段")
        BPM = args.bpm
        SECONDS_PER_BEAT = 60 / BPM
        SECONDS_PER_SEGMENT = SECONDS_PER_BEAT * BEATS_PER_SEGMENT
        segments = calculate_segments_fixed(duration, args.bpm)
        final_bpm = args.bpm
    
    print(f"\n  分段方式: {used_method}")
    print(f"  共 {len(segments)} 段:")
    for seg in segments:
        beat_info = f"(第{seg.get('beat_start','?')}-{seg.get('beat_end','?')}拍)" if 'beat_start' in seg else ""
        print(f"    第{seg['id']:2d}段: {seg['start']:6.2f}s - {seg['end']:6.2f}s {beat_info}")
    
    # 生成慢动作片段
    print(f"\n[生成] 提取慢动作 ({SLOW_SPEED}x)...")
    os.makedirs(args.output, exist_ok=True)
    all_clips = []
    
    for seg in segments:
        sid = seg['id']
        out_path = os.path.join(args.output, f"ref_seg_{sid:02d}_slow.mp4")
        extract_slow_segment(args.reference, seg['start'], seg['end'], out_path)
        all_clips.append(out_path)
        print(f"    ✓ {os.path.basename(out_path)}")
    
    # 合并
    merged = os.path.join(args.output, "all_segments_merged.mp4")
    print(f"\n[合并] -> {merged}")
    merge_videos(all_clips, merged)
    
    print(f"\n{'='*60}")
    print(f"  ✅ 完成！{len(segments)}段慢动作 → {args.output}/")
    print(f"  分段方式: {used_method}")
    print(f"{'='*60}")
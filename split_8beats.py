# split_8beats.py
# 功能：将参考视频按8拍分段，每段慢动作0.8x
# 输入：videos/reference.mp4
# 输出：output/segments/ 文件夹
#   - ref_seg_01_slow.mp4, ref_seg_02_slow.mp4, ... （各段单独文件）
#   - all_segments_merged.mp4 （所有段合并成一个文件）

import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import os
import subprocess

# ==================== 配置 ====================

BPM = 120                      # 音乐BPM
BEATS_PER_SEGMENT = 8          # 每段8拍
SLOW_SPEED = 0.8               # 慢动作倍速
TARGET_FPS = 30                # 输出帧率

SECONDS_PER_BEAT = 60 / BPM
SECONDS_PER_SEGMENT = SECONDS_PER_BEAT * BEATS_PER_SEGMENT


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


def calculate_segments(duration_seconds):
    num = max(1, int(duration_seconds / SECONDS_PER_SEGMENT))
    if num * SECONDS_PER_SEGMENT < duration_seconds:
        num += 1
    segments = []
    for i in range(num):
        segments.append({
            'id': i + 1,
            'start': round(i * SECONDS_PER_SEGMENT, 2),
            'end': round(min((i+1) * SECONDS_PER_SEGMENT, duration_seconds), 2)
        })
    return segments


def extract_slow_segment(video_path, start_time, end_time, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    repeat = max(1, int(1 / SLOW_SPEED))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (w, h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Seg {os.path.basename(output_path)[7:9]} | 0.8x", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for _ in range(repeat):
            out.write(frame)
    
    cap.release()
    out.release()


def merge_videos(video_list, output_path):
    if not video_list:
        return
    list_file = 'temp_list.txt'
    with open(list_file, 'w', encoding='utf-8') as f:
        for v in video_list:
            f.write(f"file '{os.path.abspath(v)}'\n")
    try:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error',
                       '-f', 'concat', '-safe', '0',
                       '-i', list_file, '-c', 'copy', output_path], check=True)
    except:
        # OpenCV备选
        cap = cv2.VideoCapture(video_list[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', default='videos/reference.mp4')
    parser.add_argument('-b', '--bpm', type=int, default=120)
    args = parser.parse_args()
    
    BPM = args.bpm
    SECONDS_PER_BEAT = 60 / BPM
    SECONDS_PER_SEGMENT = SECONDS_PER_BEAT * BEATS_PER_SEGMENT
    
    if not os.path.exists(args.reference):
        print(f"❌ 视频不存在: {args.reference}")
        exit(1)
    
    print("\n" + "="*50)
    print("  参考视频 8拍慢动作分段")
    print("="*50)
    print(f"  BPM: {BPM} | 每段: {SECONDS_PER_SEGMENT:.1f}秒 | 慢动作: {SLOW_SPEED}x")
    
    fps, frames, dur, w, h = get_video_info(args.reference)
    print(f"  视频: {dur:.1f}秒, {fps}fps, {w}x{h}")
    
    segments = calculate_segments(dur)
    print(f"  分段: {len(segments)} 段")
    
    os.makedirs('output/segments', exist_ok=True)
    all_clips = []
    
    for seg in segments:
        sid = seg['id']
        out_path = f'output/segments/ref_seg_{sid:02d}_slow.mp4'
        print(f"  第{sid}段 [{seg['start']:.1f}s-{seg['end']:.1f}s] -> {out_path}")
        extract_slow_segment(args.reference, seg['start'], seg['end'], out_path)
        all_clips.append(out_path)
    
    merged = 'output/segments/all_segments_merged.mp4'
    print(f"\n  合并 -> {merged}")
    merge_videos(all_clips, merged)
    
    print(f"\n✅ 完成！{len(segments)}段慢动作视频已保存到 output/segments/")
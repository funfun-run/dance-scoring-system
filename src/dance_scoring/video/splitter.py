# video/splitter.py — 视频分段与慢动作提取

import cv2
import numpy as np

from dance_scoring.core.config import BEATS_PER_SEGMENT, SLOW_SPEED, TARGET_FPS, MIN_SEGMENT_DURATION


def get_beat_segments(beat_times, duration: float, beats_per_seg: int = BEATS_PER_SEGMENT):
    """根据节拍时间点生成分段列表，返回 [{'id','start','end'}, ...] 或 None"""
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


def calculate_segments_fixed(duration_seconds: float, bpm: int):
    """固定 BPM 计算分段"""
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


def extract_slow_segment(video_path: str, start_time: float, end_time: float, output_path: str):
    """提取单个慢动作片段（0.8x 速度）"""
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

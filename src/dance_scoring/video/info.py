# video/info.py — 视频信息提取

import cv2


def get_video_info(video_path: str):
    """
    获取视频基本信息。
    返回: (fps, frame_count, duration_seconds, width, height)
    """
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

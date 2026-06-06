# video/merger.py — 视频片段合并

import cv2
import os
import subprocess


def merge_videos(video_list, output_path: str):
    """将多个视频片段合并为一个文件"""
    if not video_list:
        return

    cap = cv2.VideoCapture(video_list[0])
    target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

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
    except Exception:
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

# core/segments.py — 分段评分与练习片段提取

import os
import shutil
import cv2
import numpy as np

from .config import PASS_SCORE, SLOW_SPEED, TARGET_FPS, BEATS_PER_SEGMENT


def seg_by_beats(ref, path, fs, target_fps, bpm):
    """
    按固定 BPM/节拍分段，段号与视频分割输出文件一致。
    """
    total_ref = len(ref)
    spb = 60.0 / bpm
    sps = spb * BEATS_PER_SEGMENT
    frames_per_seg = int(sps * target_fps)
    if frames_per_seg <= 0:
        frames_per_seg = total_ref

    nseg = max(1, (total_ref + frames_per_seg - 1) // frames_per_seg)

    ref_to_path = {}
    for idx, (ri, ui) in enumerate(path):
        if ri not in ref_to_path:
            ref_to_path[ri] = idx

    segs = []
    for sid in range(nseg):
        sf = sid * frames_per_seg
        ef = min((sid+1) * frames_per_seg, total_ref)

        seg_fs = []
        for ri in range(sf, ef):
            if ri in ref_to_path:
                seg_fs.append(fs[ref_to_path[ri]])

        if seg_fs:
            ss = round(np.mean(seg_fs), 1)
        else:
            ss = 0.0

        qualified = "合格" if ss >= PASS_SCORE else "不合格"

        segs.append({
            'id': sid + 1,
            'ref_start': sf,
            'ref_end': ef,
            'start_time': round(sf/target_fps, 2),
            'end_time': round(ef/target_fps, 2),
            'score': ss,
            'qualified': qualified
        })

    return segs


def extract_clips_from_segments(segs, segments_dir="output/segments",
                                out_dir="output/low_score_clips", cfg=None):
    """
    从视频分割工具生成的慢动作片段中复制低分段落。段号统一，直接按 id 匹配。
    """
    fail_segs = [s for s in segs if s['score'] < PASS_SCORE]
    if not fail_segs:
        threshold = cfg.score_threshold if cfg else 50.0
        fail_segs = [s for s in segs if s['score'] < threshold]
    if not fail_segs:
        return []

    if not os.path.isdir(segments_dir):
        print(f"  ⚠️ 未找到分段目录 {segments_dir}，从参考视频实时提取...")
        return _extract_fallback(fail_segs, out_dir)

    existing = [f for f in os.listdir(segments_dir) if f.startswith('ref_seg_')]
    if not existing:
        print(f"  ⚠️ {segments_dir} 为空，从参考视频实时提取...")
        return _extract_fallback(fail_segs, out_dir)

    os.makedirs(out_dir, exist_ok=True)
    files = []
    for seg in fail_segs:
        src = os.path.join(segments_dir, f"ref_seg_{seg['id']:02d}_slow.mp4")
        dst = os.path.join(out_dir, f"practice_seg{seg['id']:02d}_score{seg['score']:.0f}_slow.mp4")
        if os.path.exists(src):
            shutil.copy(src, dst)
            files.append(dst)
            print(f"    ✓ {os.path.basename(dst)}")
        else:
            print(f"    ⚠️ 找不到: {os.path.basename(src)}，实时提取...")
            _extract_single_clip(seg, dst)
            if os.path.exists(dst):
                files.append(dst)
    return files


def _extract_fallback(segs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for seg in segs:
        dst = os.path.join(out_dir, f"practice_seg{seg['id']:02d}_score{seg['score']:.0f}_slow.mp4")
        _extract_single_clip(seg, dst)
        if os.path.exists(dst):
            files.append(dst)
            print(f"    ✓ {os.path.basename(dst)} (实时提取)")
    return files


def _extract_single_clip(seg, output_path, ref_video=None):
    if ref_video is None:
        for p in ['assets/videos/reference.mp4', 'assets/videos/ref.mp4',
                   'videos/reference.mp4', 'videos/ref.mp4']:
            if os.path.exists(p):
                ref_video = p
                break
    if ref_video is None or not os.path.exists(ref_video):
        print(f"    ✗ 无法找到参考视频，跳过段{seg['id']}")
        return

    cap = cv2.VideoCapture(ref_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    repeat = max(1, int(1/SLOW_SPEED))

    sf = int(seg['ref_start'] * fps / TARGET_FPS)
    ef = int(seg['ref_end'] * fps / TARGET_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (w,h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    for _ in range(sf, ef):
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame, f"Seg{seg['id']} Score:{seg['score']:.0f}", (30,40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        for _ in range(repeat): out.write(frame)
    out.release(); cap.release()

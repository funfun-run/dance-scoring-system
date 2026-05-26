# scripts/split.py — CLI 视频分割入口

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dance_scoring.video.info import get_video_info
from dance_scoring.video.beat_detector import detect_beats_from_audio, detect_beats_from_motion
from dance_scoring.video.splitter import get_beat_segments, calculate_segments_fixed, extract_slow_segment
from dance_scoring.video.merger import merge_videos
from dance_scoring.core.config import BEATS_PER_SEGMENT, SLOW_SPEED, DEFAULT_BPM


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="8拍慢动作视频分割")
    parser.add_argument('-r', '--reference', default='assets/videos/reference.mp4')
    parser.add_argument('-b', '--bpm', type=int, default=DEFAULT_BPM)
    parser.add_argument('-o', '--output', default='output/segments')
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"❌ 视频不存在: {args.reference}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("   🎬 8拍慢动作分段 v2.3")
    print("=" * 60)

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

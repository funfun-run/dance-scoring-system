# scripts/score.py — CLI 舞蹈评分入口

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dance_scoring.core.config import Config, DEFAULT_BPM, BEATS_PER_SEGMENT, PASS_SCORE
from dance_scoring.core.extractor import PoseExtractor, download_model
from dance_scoring.core.scorer import Scorer
from dance_scoring.core.segments import extract_clips_from_segments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="舞蹈评分系统 CLI")
    parser.add_argument('-r', '--reference', default='assets/videos/reference.mp4')
    parser.add_argument('-u', '--user', default='assets/videos/user.mp4')
    parser.add_argument('-b', '--bpm', type=int, default=DEFAULT_BPM, help='BPM（需与视频分割工具一致）')
    parser.add_argument('-t', '--threshold', type=float, default=50.0)
    parser.add_argument('-s', '--segments', default='output/segments', help='视频分段输出目录')
    args = parser.parse_args()

    for p, n in [(args.reference, "参考"), (args.user, "用户")]:
        if not os.path.exists(p):
            print(f"❌ {n}视频不存在: {p}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("   🕺 舞蹈评分 v1.0 | 统一分段逻辑")
    print("=" * 60)
    print(f"  BPM:{args.bpm} | 每段{BEATS_PER_SEGMENT}拍={60/args.bpm*BEATS_PER_SEGMENT:.1f}秒")
    print(f"  分段与视频分割工具完全一致，段号对应")
    print(f"  合格线:{PASS_SCORE:.0f}分 | 阈值:{args.threshold}分")

    download_model()
    cfg = Config(score_threshold=args.threshold)

    print("\n[1/3] 提取参考...")
    ref = PoseExtractor(cfg).extract(args.reference)
    print("\n[2/3] 提取用户...")
    user = PoseExtractor(cfg).extract(args.user)
    print("\n[3/3] 评分...")
    scorer = Scorer(cfg, bpm=args.bpm)
    overall, segs, low, path = scorer.score(ref, user)

    fail_segs = [s for s in segs if s['score'] < PASS_SCORE]

    print("\n" + "=" * 60)
    print(f"      总评: {overall:.1f}/100")
    if overall >= 90:      print("      ⭐优秀")
    elif overall >= 78:    print("      👍良好")
    elif overall >= 60:    print("      📝还行")
    elif overall >= 35:    print("      ⚠️需改进")
    else:                  print("      💪需重练")
    if fail_segs:          print(f"      ⚠️ {len(fail_segs)}/{len(segs)}段不合格")
    print("=" * 60)

    print(f"\n{'段号':<6}{'时间':<16}{'得分':<10}{'判定'}")
    print("-" * 50)
    for s in segs:
        t = f"{s['start_time']:.1f}s-{s['end_time']:.1f}s"
        q = "✅合格" if s['score'] >= PASS_SCORE else "❌不合格"
        print(f"{s['id']:<6}{t:<16}{s['score']:<10.1f}{q}")

    if fail_segs:
        print(f"\n❌ {len(fail_segs)}段不合格，输出慢动作视频:")
        for s in fail_segs:
            print(f"  第{s['id']}段 [{s['start_time']:.1f}s-{s['end_time']:.1f}s] {s['score']:.1f}分")
        files = extract_clips_from_segments(segs, args.segments, cfg=cfg)
        if files: print(f"已保存到 output/low_score_clips/")
    elif low:
        print(f"\n⚠️ 全部合格，但以下片段低于{args.threshold}分:")
        for s in low:
            print(f"  第{s['id']}段 [{s['start_time']:.1f}s-{s['end_time']:.1f}s] {s['score']:.1f}分")
        files = extract_clips_from_segments(segs, args.segments, cfg=cfg)
        if files: print(f"已保存到 output/low_score_clips/")
    else:
        print(f"\n🎉 全部合格！")
    print("=" * 60)

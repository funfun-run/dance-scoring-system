#!/usr/bin/env python3
"""score_bridge.py — 子进程评分桥接

在独立进程中完成 MediaPipe 提取 + 评分，输出 JSON 结果到 stdout。
用于隔离 MediaPipe C++（llvmpipe 环境下与 GUI 主线程冲突）。

用法:
    python scripts/score_bridge.py --ref <ref.mp4> --user <user.mp4> [options]

输出: JSON（包含 overall, segs, corrections, joint_devs, files 等）
"""
import argparse, json, sys, os, traceback

def _dev_to_dict(d):
    """Deviation 对象 → JSON 可序列化 dict。"""
    return {
        "joint_name": d.joint_name,
        "joint_idx": d.joint_idx,
        "deviation_deg": d.deviation_deg,
        "direction": d.direction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--algo", default="dtw", choices=["dtw", "fastdtw"])
    parser.add_argument("--correction", default="rule", choices=["rule", "llm"])
    parser.add_argument("--model", default="3b", choices=["1.5b", "3b"])
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    # 抑制 extractor 的 print() 输出，防止污染 JSON stdout
    import contextlib, io as _io
    _quiet = contextlib.redirect_stdout(_io.StringIO())

    try:
        from dance_scoring.core.config import Config
        from dance_scoring.core.extractor import PoseExtractor, download_model
        from dance_scoring.core.scorer import Scorer
        from dance_scoring.core.correction_provider import create_correction_provider
        from dance_scoring.core.segments import extract_clips_from_segments

        download_model()
        cfg = Config(score_threshold=args.threshold)

        # 1) 提取参考（抑制 print）
        with _quiet:
            ext = PoseExtractor(cfg)
            ref = ext.extract(args.ref)
            ext.close()

        # 2) 提取用户
        with _quiet:
            ext = PoseExtractor(cfg)
            user = ext.extract(args.user)
            ext.close()

        # 3) 创建纠正提供者
        try:
            corr = create_correction_provider(args.correction, model=args.model)
        except Exception:
            corr = None  # 回退 rule

        # 4) 评分（抑制 print）
        with _quiet:
            scorer = Scorer(cfg, bpm=args.bpm, alignment_method=args.algo)
            overall, segs, low, path = scorer.score(ref, user, correction_provider=corr)

        # 5) 输出练习视频
        with _quiet:
            segs_dir = os.path.join(args.output_dir, "segments")
            files = extract_clips_from_segments(segs, segs_dir, cfg=cfg)

        # 6) 序列化 segs（Deviation 对象 → dict）
        corrections = {}
        joint_devs = {}
        serializable_segs = []
        for s in segs:
            d = {
                "id": s["id"],
                "score": s["score"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "deviations": [_dev_to_dict(dev) for dev in s.get("deviations", [])],
                "correction_text": s.get("correction_text", ""),
                "skipped_joints": s.get("skipped_joints", []),
                "joint_visibility": s.get("joint_visibility", {}),
            }
            serializable_segs.append(d)

            if s.get("correction_text"):
                corrections[str(s["id"])] = s["correction_text"]
            if s.get("deviations"):
                joint_devs[str(s["id"])] = [_dev_to_dict(dev) for dev in s["deviations"]]

        result = {
            "success": True,
            "overall": overall,
            "segs": serializable_segs,
            "corrections": corrections,
            "joint_devs": joint_devs,
            "files": files,
            "ref_frames": len(ref),
            "user_frames": len(user),
        }
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                import numpy as np
                if isinstance(o, (np.floating, np.integer)):
                    return float(o)
                return super().default(o)

        json.dump(result, sys.stdout, ensure_ascii=False, cls=_NumpyEncoder)
        sys.stdout.flush()

    except Exception:
        json.dump({"success": False, "error": traceback.format_exc()}, sys.stdout)
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()

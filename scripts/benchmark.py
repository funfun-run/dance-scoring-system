#!/usr/bin/env python3
"""
姿态推理性能基准测试。

对比 MediaPipe (CPU) 和 OpenVINO (NPU/GPU/CPU) 的推理性能，
输出竞赛指标达标情况。

用法:
    python scripts/benchmark.py <video_path>
    python scripts/benchmark.py <video_path> --frames 100 --rounds 3
"""

import sys
import os
import time
import json
import argparse
import zipfile
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dance_scoring.core.config import Config
from dance_scoring.platform.npu import NPUManager


def benchmark_backend(
    engine,
    frames: List[np.ndarray],
    name: str,
    warmup_rounds: int = 3,
) -> Dict:
    """
    对指定后端运行性能基准测试。

    返回:
        {
            'name': str,
            'first_latency_ms': float,
            'mean_latency_ms': float,
            'median_latency_ms': float,
            'p99_latency_ms': float,
            'throughput_fps': float,
            'total_frames': int,
        }
    """
    print(f"\n🔬 测试 {name}...")

    if hasattr(engine, 'warmup'):
        engine.warmup(rounds=warmup_rounds)

    latencies = []

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = i * 33  # ~30fps timestamp

        t0 = time.perf_counter()
        result = engine.extract_frame(rgb, ts)
        elapsed = (time.perf_counter() - t0) * 1000

        if result is not None:
            latencies.append(elapsed)

        if (i + 1) % 50 == 0:
            print(f"  进度: {i + 1}/{len(frames)}")

    if not latencies:
        return {
            'name': name,
            'first_latency_ms': 0,
            'mean_latency_ms': 0,
            'median_latency_ms': 0,
            'p99_latency_ms': 0,
            'throughput_fps': 0,
            'total_frames': 0,
            'error': '无有效帧',
        }

    lats = np.array(latencies)
    return {
        'name': name,
        'first_latency_ms': round(latencies[0], 2),
        'mean_latency_ms': round(float(lats.mean()), 2),
        'median_latency_ms': round(float(np.median(lats)), 2),
        'p99_latency_ms': round(float(np.percentile(lats, 99)), 2),
        'throughput_fps': round(1000.0 / max(lats.mean(), 1e-6), 1),
        'total_frames': len(latencies),
    }


def get_model_sizes() -> Dict:
    """获取模型体积对比信息。"""
    result = {
        'tflite_size_kb': 0,
        'ir_size_kb': 0,
        'compression_ratio': 0,
        'compression_pct': 0,
        'meets_target': False,
    }

    # TFLite 大小
    task_path = Path('pose_landmarker_lite.task')
    if task_path.exists():
        try:
            with zipfile.ZipFile(task_path, 'r') as zf:
                landmark_names = [n for n in zf.namelist()
                                  if 'landmark' in n.lower() and n.endswith('.tflite')]
                if landmark_names:
                    result['tflite_size_kb'] = round(
                        zf.getinfo(landmark_names[0]).file_size / 1024, 1)
        except Exception:
            pass

    # IR 大小
    ir_bin = Path('src/dance_scoring/models/pose_landmarker.bin')
    if ir_bin.exists():
        result['ir_size_kb'] = round(ir_bin.stat().st_size / 1024, 1)

    if result['tflite_size_kb'] > 0 and result['ir_size_kb'] > 0:
        ratio = result['ir_size_kb'] / result['tflite_size_kb']
        result['compression_ratio'] = round(ratio, 4)
        result['compression_pct'] = round((1 - ratio) * 100, 1)
        result['meets_target'] = ratio <= 0.5

    # 从 meta.json 读取精确数据
    meta_path = Path('src/dance_scoring/models/pose_landmarker_meta.json')
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get('tflite_size_bytes'):
                result['tflite_size_kb'] = round(meta['tflite_size_bytes'] / 1024, 1)
            if meta.get('ir_size_bytes'):
                result['ir_size_kb'] = round(meta['ir_size_bytes'] / 1024, 1)
            if meta.get('compression_ratio'):
                result['compression_ratio'] = meta['compression_ratio']
                result['compression_pct'] = round((1 - meta['compression_ratio']) * 100, 1)
                result['meets_target'] = meta['compression_ratio'] <= 0.5
        except Exception:
            pass

    return result


def print_results(mp_result: Dict, ov_result: Dict, model_info: Dict):
    """输出格式化的性能对比和竞赛指标。"""
    print("\n" + "=" * 60)
    print("   姿态推理性能对比")
    print("=" * 60)

    # 设备信息
    print(f"\n设备: Intel Core Ultra 5 225U")
    print(f"NPU: {'可用' if NPUManager.available() else '不可用 (CPU 回退)'}")
    print(f"测试帧数: {mp_result.get('total_frames', 0)}")

    # 性能表
    print(f"\n{'指标':<20} {'MediaPipe(CPU)':<18} {'OpenVINO(CPU)':<18} {'竞赛要求'}")
    print("-" * 70)
    rows = [
        ("首帧延迟(ms)", "first_latency_ms"),
        ("平均延迟(ms)", "mean_latency_ms"),
        ("中位延迟(ms)", "median_latency_ms"),
        ("P99 延迟(ms)", "p99_latency_ms"),
        ("吞吐量(fps)", "throughput_fps"),
    ]
    for label, key in rows:
        mp_val = mp_result.get(key, '-')
        ov_val = ov_result.get(key, '-')
        target = ""
        if key == "mean_latency_ms":
            target = "≤50"
        elif key == "throughput_fps":
            target = "≥20"
        print(f"{label:<20} {str(mp_val):<18} {str(ov_val):<18} {target}")

    # 模型体积
    print(f"\n{'模型体积对比':}")
    print(f"  原始 TFLite:  {model_info['tflite_size_kb']:.1f} KB "
          f"({model_info['tflite_size_kb']/1024:.1f} MB)")
    print(f"  IR (FP16):    {model_info['ir_size_kb']:.1f} KB "
          f"({model_info['ir_size_kb']/1024:.1f} MB)")
    print(f"  压缩率:       {model_info['compression_pct']:.1f}%")
    if model_info['meets_target']:
        print(f"  竞赛指标:     ✓ 满足 ≥50%")
    else:
        print(f"  竞赛指标:     ⚠ 未达标 (当前 {model_info['compression_pct']:.1f}%)")

    # 竞赛指标总览
    ov_mean = ov_result.get('mean_latency_ms', float('inf'))
    ov_fps = ov_result.get('throughput_fps', 0)

    print(f"\n{'竞赛指标达标情况':}")
    checks = [
        ("推理延迟 ≤50ms", ov_mean <= 50),
        ("帧率 ≥20fps", ov_fps >= 20),
        ("模型体积压缩 ≥50%", model_info['meets_target']),
    ]
    all_pass = True
    for label, passed in checks:
        icon = "✓" if passed else "✗"
        if not passed:
            all_pass = False
        print(f"  [{icon}] {label}")
    print(f"\n  {'✓ 全部达标' if all_pass else '⚠ 部分未达标'}")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="姿态推理性能基准测试 — MediaPipe vs OpenVINO",
    )
    parser.add_argument('video', nargs='?', default=None,
                        help='测试视频路径')
    parser.add_argument('--frames', type=int, default=100,
                        help='测试帧数 (默认: 100)')
    parser.add_argument('--rounds', type=int, default=2,
                        help='预热轮数 (默认: 2)')
    args = parser.parse_args()

    # 加载测试帧
    frames = []
    if args.video and os.path.exists(args.video):
        cap = cv2.VideoCapture(args.video)
        fid = 0
        while len(frames) < args.frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            fid += 1
        cap.release()
        print(f"📹 测试视频: {os.path.basename(args.video)} | "
              f"{len(frames)}帧")
    else:
        # 生成随机假帧
        print(f"⚠ 无测试视频，使用随机帧 ({args.frames}帧)")
        for _ in range(args.frames):
            frames.append(
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            )

    cfg = Config()

    # ---- MediaPipe ----
    from dance_scoring.core.engine import MediaPipeEngine
    mp_engine = MediaPipeEngine(cfg)
    mp_result = benchmark_backend(mp_engine, frames, "MediaPipe (CPU)",
                                  warmup_rounds=args.rounds)

    # ---- OpenVINO ----
    try:
        from dance_scoring.core.engine import OpenVINOEngine
        ov_engine = OpenVINOEngine(cfg)
        ov_result = benchmark_backend(ov_engine, frames, "OpenVINO (CPU)",
                                      warmup_rounds=args.rounds)
    except Exception as e:
        print(f"⚠ OpenVINO 不可用: {e}")
        ov_result = {
            'name': 'OpenVINO (N/A)',
            'first_latency_ms': 0,
            'mean_latency_ms': 0,
            'median_latency_ms': 0,
            'p99_latency_ms': 0,
            'throughput_fps': 0,
            'total_frames': 0,
            'error': str(e),
        }

    # ---- 输出 ----
    model_info = get_model_sizes()
    print_results(mp_result, ov_result, model_info)


if __name__ == "__main__":
    main()

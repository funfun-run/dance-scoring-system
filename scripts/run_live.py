#!/usr/bin/env python3
"""
实时舞蹈跟练 — USB 摄像头 + 姿态比对 + 即时反馈。

用法:
    python scripts/run_live.py -r <reference.mp4>
    python scripts/run_live.py -r <reference.mp4> -c 0 -t 60 --no-display
"""

import sys
import os
import signal
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dance_scoring.core.config import Config, BEATS_PER_SEGMENT, DEFAULT_BPM
from dance_scoring.core.extractor import PoseExtractor, download_model
from dance_scoring.core.frame import PoseFrame
from dance_scoring.core.scorer import Scorer
from dance_scoring.core.correction import generate_correction
from dance_scoring.camera.usb import UsbCamera
from dance_scoring.camera.base import CameraBase


# ============================================================
# 配置
# ============================================================

@dataclass
class LiveConfig:
    """实时跟练模式配置。"""
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    window_size: int = 150        # 滑动窗口帧数
    window_step: int = 30         # 每次对齐步进帧数
    alignment_method: str = "fastdtw"  # 实时模式默认 fastdtw
    score_threshold: float = 50.0
    pass_score: float = 60.0
    correction_threshold: float = 10.0
    bpm: int = DEFAULT_BPM
    no_display: bool = False      # 无 GUI 模式（仅终端输出）


# ============================================================
# 滑动窗口缓冲
# ============================================================

class SlidingWindowBuffer:
    """固定大小的滑动窗口，用于累积用户实时帧。"""

    def __init__(self, max_size: int = 150, step: int = 30):
        self._buffer: List[PoseFrame] = []
        self._max_size = max_size
        self._step = step

    def add(self, frame: PoseFrame) -> None:
        """追加一帧到窗口末尾。"""
        self._buffer.append(frame)

    def is_full(self) -> bool:
        """窗口是否已满 (≥ max_size)。"""
        return len(self._buffer) >= self._max_size

    def get_window(self) -> List[PoseFrame]:
        """返回当前窗口的副本。"""
        return list(self._buffer)

    def slide(self) -> None:
        """移除最老的 step 帧，为下一轮留空间。"""
        if len(self._buffer) > self._step:
            self._buffer = self._buffer[self._step:]
        else:
            self._buffer = []

    def clear(self) -> None:
        """清空缓冲区。"""
        self._buffer = []

    def __len__(self) -> int:
        return len(self._buffer)


# ============================================================
# 实时评分配器
# ============================================================

class LiveScorer:
    """
    实时跟练核心编排器。
    职责：摄像头采集 → 姿态提取 → 滑动窗口 → DTW/fastdtw 对齐 → 打分 → 纠正建议。
    """

    def __init__(self, config: LiveConfig, reference_path: str):
        self.cfg = config
        self.ref_path = reference_path
        self._camera: Optional[CameraBase] = None
        self._extractor: Optional[PoseExtractor] = None
        self._ref_poses: List[PoseFrame] = []
        self._ref_segments: List[Dict] = []   # 参考视频分段信息
        self._buffer = SlidingWindowBuffer(config.window_size, config.window_step)
        self._running = False
        self._scorer_config = Config(score_threshold=config.score_threshold)
        self._current_seg_idx = 0
        self._seg_results: List[Dict] = []    # 各段评分结果累积

    # ---------- 初始化 ----------

    def setup(self) -> bool:
        """初始化摄像头、姿态提取器、加载参考视频。成功返回 True。"""
        # 1. 摄像头
        self._camera = UsbCamera(
            device_id=self.cfg.camera_id,
            resolution=(self.cfg.camera_width, self.cfg.camera_height),
            fps=self.cfg.camera_fps,
        )
        if not self._camera.open():
            print("❌ 无法打开摄像头")
            return False
        print(f"📷 摄像头已打开 | {self.cfg.camera_fps}fps | "
              f"{self.cfg.camera_width}x{self.cfg.camera_height}")

        # 2. 姿态提取器
        download_model()
        self._extractor = PoseExtractor(self._scorer_config)

        # 3. 加载参考视频 → 预提取全部姿态序列
        if not os.path.exists(self.ref_path):
            print(f"❌ 参考视频不存在: {self.ref_path}")
            return False
        print(f"📐 加载参考视频: {os.path.basename(self.ref_path)}")
        self._ref_poses = self._extractor.extract(self.ref_path)
        if len(self._ref_poses) < 10:
            print(f"❌ 参考视频姿态帧数不足: {len(self._ref_poses)}")
            return False

        # 4. 按八拍分段
        spb = 60.0 / self.cfg.bpm
        sps = spb * BEATS_PER_SEGMENT
        frames_per_seg = max(1, int(sps * self._scorer_config.target_fps))
        total_frames = len(self._ref_poses)
        nseg = max(1, (total_frames + frames_per_seg - 1) // frames_per_seg)

        for sid in range(nseg):
            sf = sid * frames_per_seg
            ef = min((sid + 1) * frames_per_seg, total_frames)
            self._ref_segments.append({
                'id': sid + 1,
                'ref_start': sf,
                'ref_end': ef,
                'start_time': sf / self._scorer_config.target_fps,
                'end_time': ef / self._scorer_config.target_fps,
            })

        print(f"📐 参考姿态已加载: {total_frames}帧 ({nseg}段)")
        return True

    # ---------- 主循环 ----------

    def start(self) -> None:
        """启动实时跟练主循环。Ctrl+C 退出。"""
        if self._extractor is None or self._camera is None:
            print("❌ 请先调用 setup()")
            return

        self._running = True
        print("▶️ 开始跟练...\n")

        try:
            while self._running:
                # 1. 读取摄像头帧
                frame = self._camera.read()
                if frame is None:
                    print("⚠️ 摄像头读取失败，等待...")
                    time.sleep(0.1)
                    continue

                # 2. 单帧姿态提取
                pf = self._extract_single_frame(frame)
                if pf is None:
                    # 无人或检测失败，跳过
                    if not self.cfg.no_display:
                        cv2.imshow("Live Dance Practice", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue

                # 3. 追加到滑动窗口
                self._buffer.add(pf)

                # 4. 窗口满 → 对齐 + 打分
                if self._buffer.is_full():
                    seg_result = self._score_window()
                    self._seg_results.append(seg_result)
                    self._print_segment_result(seg_result)
                    self._buffer.slide()

                # 5. 显示帧（可选）
                if not self.cfg.no_display:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(display_frame, "Live Dance Practice",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Live Dance Practice", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._print_summary()

    def stop(self) -> None:
        """停止跟练。"""
        self._running = False
        if self._camera:
            self._camera.close()
        cv2.destroyAllWindows()
        print("⏹ 跟练已停止")

    # ---------- 单帧提取 ----------

    def _extract_single_frame(self, rgb_frame: np.ndarray) -> Optional[PoseFrame]:
        """
        从单帧 RGB 图像提取姿态。
        复用 PoseExtractor 的 MediaPipe 检测器，但不走视频循环。
        """
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts = int(time.time() * 1000) % (2**31)
            res = self._extractor.det.detect_for_video(mp_img, ts)

            if (res.pose_world_landmarks and
                    len(res.pose_world_landmarks) > 0 and
                    len(res.pose_world_landmarks[0]) >= 33):
                kp3d = np.zeros((33, 3), dtype=np.float32)
                cf = np.zeros(33, dtype=np.float32)
                for i, lm in enumerate(res.pose_world_landmarks[0][:33]):
                    kp3d[i] = [lm.x, lm.y, lm.z * 0.3]  # Z 轴加权
                    cf[i] = lm.visibility if hasattr(lm, 'visibility') else 1.0
                # 使用递增的虚拟帧 ID
                fid = len(self._buffer)
                return PoseFrame(fid, kp3d, cf)
            return None
        except Exception:
            return None

    # ---------- 窗口对齐 + 打分 ----------

    def _score_window(self) -> Dict:
        """
        用当前滑动窗口与对应参考段做 DTW/fastdtw 对齐并打分。

        返回: {segment_id, score, joint_deviations, corrections, ...}
        """
        window_poses = self._buffer.get_window()

        # 确定当前对应的参考段
        idx = min(self._current_seg_idx, len(self._ref_segments) - 1)
        seg = self._ref_segments[idx]
        ref_seg_poses = self._ref_poses[seg['ref_start']:seg['ref_end']]

        if len(ref_seg_poses) < 5:
            # 参考段太短，跳到下一段
            self._current_seg_idx = min(self._current_seg_idx + 1,
                                        len(self._ref_segments) - 1)
            return {
                'segment_id': seg['id'],
                'score': 0.0,
                'qualified': '不合格',
                'joint_deviations': {},
                'correction_text': '',
            }

        # 使用 Scorer 的对齐逻辑（复用）
        from dance_scoring.core.scorer import Scorer
        scorer = Scorer(
            self._scorer_config,
            bpm=self.cfg.bpm,
            alignment_method=self.cfg.alignment_method,
        )
        # 临时创建包含窗口+段的小型评分
        overall, segs, low, path = scorer.score(ref_seg_poses, window_poses)

        # 计算各关节平均偏差
        joint_devs = self._calc_joint_deviations(ref_seg_poses, window_poses, path)

        # 生成纠正建议
        seg_obj = type('SegObj', (), {
            'id': seg['id'],
            'score': segs[0]['score'] if segs else 0.0,
            'joint_deviations': joint_devs,
        })()
        corrections = generate_correction(
            [seg_obj],
            top_n=3,
            threshold_deg=self.cfg.correction_threshold,
        )

        result = {
            'segment_id': seg['id'],
            'score': segs[0]['score'] if segs else 0.0,
            'qualified': '合格' if (segs[0]['score'] if segs else 0) >= self.cfg.pass_score else '不合格',
            'joint_deviations': joint_devs,
            'correction_text': corrections.get(seg['id'], ''),
            'path_length': len(path),
        }

        # 推进段索引
        self._current_seg_idx = min(self._current_seg_idx + 1,
                                    len(self._ref_segments) - 1)
        return result

    def _calc_joint_deviations(
        self,
        ref_poses: List[PoseFrame],
        user_poses: List[PoseFrame],
        path: List[tuple],
    ) -> Dict[int, float]:
        """从对齐路径计算各关节的平均角度偏差。"""
        if not path:
            return {}

        from dance_scoring.core.config import ANGLE_JOINTS
        joint_sums: Dict[int, float] = {}
        joint_counts: Dict[int, int] = {}

        for ri, ui in path:
            ref_angles = ref_poses[ri].angles
            user_angles = user_poses[ui].angles
            for j, (a, b, c) in enumerate(ANGLE_JOINTS):
                # 记录每个参与关节的偏差（正 = 用户偏大）
                dev = float(user_angles[j] - ref_angles[j])
                for joint_idx in [a, b, c]:
                    joint_sums[joint_idx] = joint_sums.get(joint_idx, 0.0) + dev
                    joint_counts[joint_idx] = joint_counts.get(joint_idx, 0) + 1

        return {
            j: joint_sums[j] / max(joint_counts[j], 1)
            for j in joint_sums
        }

    # ---------- 输出 ----------

    def _print_segment_result(self, result: Dict) -> None:
        """终端输出单段结果。"""
        sid = result['segment_id']
        score = result['score']
        qualified = result['qualified']
        icon = "✓" if qualified == '合格' else "✗"
        line = f"[段{sid}] 得分:{score:.1f} {icon}"

        if result['correction_text']:
            # 去掉 "第X段：" 前缀避免重复
            text = result['correction_text']
            if text.startswith(f"第{sid}段："):
                text = text[len(f"第{sid}段："):]
            line += f" | {text}"

        print(line)

    def _print_summary(self) -> None:
        """终端输出总评。"""
        if not self._seg_results:
            return

        print("\n" + "=" * 55)

        scores = [r['score'] for r in self._seg_results if r['score'] > 0]
        if not scores:
            print("  无有效评分数据")
            print("=" * 55)
            return

        overall = round(np.mean(scores), 1)
        pass_count = sum(1 for r in self._seg_results
                        if r['qualified'] == '合格')

        if overall >= 90:
            grade = "⭐优秀"
        elif overall >= 78:
            grade = "👍良好"
        elif overall >= 60:
            grade = "📝还行"
        elif overall >= 35:
            grade = "⚠️需改进"
        else:
            grade = "💪需重练"

        print(f"🏆 总评: {grade} | 得分: {overall:.1f}/100")
        print(f"   通过率: {pass_count}/{len(self._seg_results)}")

        # 薄弱部位
        all_devs: Dict[int, float] = {}
        for r in self._seg_results:
            for j, dev in r.get('joint_deviations', {}).items():
                all_devs[j] = max(abs(dev), all_devs.get(j, 0.0))

        if all_devs:
            from dance_scoring.core.correction import JOINT_NAMES_CN
            sorted_joints = sorted(all_devs.items(), key=lambda x: x[1], reverse=True)[:5]
            weak_parts = [
                f"{JOINT_NAMES_CN.get(j, f'关节{j}')}({dev:.1f}°)"
                for j, dev in sorted_joints if dev > self.cfg.correction_threshold
            ]
            if weak_parts:
                print(f"📊 薄弱部位: {', '.join(weak_parts)}")

        print("=" * 55)


# ============================================================
# CLI 入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="实时舞蹈跟练 — 摄像头 + 姿态比对 + 即时反馈",
    )
    parser.add_argument('-r', '--reference', required=True,
                        help='参考视频路径（必须）')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='摄像头设备索引 (默认: 0)')
    parser.add_argument('-b', '--bpm', type=int, default=DEFAULT_BPM,
                        help=f'BPM (默认: {DEFAULT_BPM})')
    parser.add_argument('-t', '--threshold', type=float, default=50.0,
                        help='低分阈值 (默认: 50)')
    parser.add_argument('--window', type=int, default=150,
                        help='滑动窗口大小 (默认: 150帧)')
    parser.add_argument('--window-step', type=int, default=30,
                        help='窗口步进 (默认: 30帧)')
    parser.add_argument('--alignment', choices=['dtw', 'fastdtw'],
                        default='fastdtw',
                        help='对齐算法 (默认: fastdtw)')
    parser.add_argument('--no-display', action='store_true',
                        help='禁用 OpenCV 窗口显示（纯终端模式）')
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"❌ 参考视频不存在: {args.reference}")
        sys.exit(1)

    config = LiveConfig(
        camera_id=args.camera,
        bpm=args.bpm,
        score_threshold=args.threshold,
        window_size=args.window,
        window_step=args.window_step,
        alignment_method=args.alignment,
        no_display=args.no_display,
    )

    scorer = LiveScorer(config, args.reference)
    if not scorer.setup():
        sys.exit(1)

    # 注册信号处理 (SIGINT → 优雅退出)
    def _handle_sigint(signum, frame):
        print("\n⏹ 收到退出信号...")
        scorer.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    scorer.start()
    scorer.stop()


if __name__ == "__main__":
    main()

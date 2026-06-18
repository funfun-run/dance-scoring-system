# gui/live_view.py — 实时跟练界面 (v3 — 双画面版)
#
# 布局: 左上「我的画面」(摄像头) + 左下「参考示范」(当前段循环)
#       右侧 得分 / 纠正 / 薄弱部位
#       底部 控制栏
#
# v3: 加入参考示范画面，用户可以对照练习

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import queue
import time
import os
from typing import Optional, List, Dict

import numpy as np
import cv2

from dance_scoring.gui.logger import log, guard
from dance_scoring.gui.theme import COLORS, FONTS

VIDEO_FILTERS = [("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                 ("所有文件", "*.*")]

# ============================================================
# 骨骼叠加绘制
# ============================================================

class PoseOverlay:
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (27, 31), (28, 30), (28, 32),
        (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (1, 7), (4, 8), (9, 10),
    ]

    @staticmethod
    def draw(frame_bgr, landmarks, weak_joints=None):
        weak_joints = set(weak_joints or [])
        h, w = frame_bgr.shape[:2]
        for a, b in PoseOverlay.POSE_CONNECTIONS:
            pa, pb = landmarks[a], landmarks[b]
            if (0 <= pa[0] < w and 0 <= pa[1] < h and
                0 <= pb[0] < w and 0 <= pb[1] < h):
                cv2.line(frame_bgr, (int(pa[0]), int(pa[1])),
                         (int(pb[0]), int(pb[1])), (255, 255, 255), 1)
        for i, pt in enumerate(landmarks):
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                color = (0, 0, 255) if i in weak_joints else (0, 255, 0)
                cv2.circle(frame_bgr, (int(pt[0]), int(pt[1])), 3, color, -1)
        return frame_bgr


# ============================================================
# 后台工作线程
# ============================================================

class LiveWorker(threading.Thread):

    def __init__(self, ref_path: str, result_queue: queue.Queue,
                 camera_id: int = 0, bpm: int = 120,
                 alignment: str = "fastdtw"):
        super().__init__(daemon=True)
        self.ref_path = ref_path
        self.queue = result_queue
        self.camera_id = camera_id
        self.bpm = bpm
        self.alignment = alignment
        self._cancel = threading.Event()
        self._paused = threading.Event()  # 轮次完成后暂停，等待 GUI 点击继续

    def cancel(self):
        self._cancel.set()
        self._paused.set()  # 取消时也要解除暂停，防止永久阻塞
        try:
            if hasattr(self, 'camera') and self.camera:
                self.camera.close()
        except Exception:
            pass

    def resume(self):
        """GUI 调用，解除暂停继续下一轮练习。"""
        self._paused.set()

    def run(self):
        with guard("LiveWorker.run"):
            try:
                self._setup()
                self._loop()
            except Exception as e:
                import traceback
                log.error(f"LiveWorker 异常: {traceback.format_exc()}")
                self.queue.put({'type': 'error',
                               'message': f"{e}\n{traceback.format_exc()}"})
            finally:
                self._cleanup()

    # ======== 初始化 ========

    def _setup(self):
        with guard("LiveWorker._setup"):
            # 摄像头
            log.info("打开摄像头...")
            from dance_scoring.camera.usb import UsbCamera
            self.camera = UsbCamera(device_id=self.camera_id)
            if not self.camera.open():
                log.error(f"摄像头打开失败 (device_id={self.camera_id})")
                self.queue.put({'type': 'error',
                               'message': f'无法打开摄像头 (device_id={self.camera_id})'})
                raise RuntimeError('摄像头打开失败')

            self.queue.put({'type': 'status', 'message': '📷 摄像头已打开'})
            log.info("摄像头就绪")

            # 姿态提取器
            log.info("下载/加载 MediaPipe 模型...")
            from dance_scoring.core.extractor import PoseExtractor, download_model
            from dance_scoring.core.config import Config
            download_model()
            self.extractor = PoseExtractor(Config())
            self.queue.put({'type': 'status', 'message': '📐 姿态模型已加载'})
            log.info("MediaPipe 模型就绪")

            # 参考视频 → 预提取姿态序列
            self.queue.put({'type': 'status', 'message': '📹 提取参考视频姿态...'})
            log.info(f"提取参考姿态: {self.ref_path}")
            self.ref_poses = self.extractor.extract(self.ref_path)
            nref = len(self.ref_poses)
            log.info(f"参考姿态: {nref} 帧")

            # 按八拍分段
            from dance_scoring.core.config import BEATS_PER_SEGMENT, TARGET_FPS
            spb = 60.0 / self.bpm
            sps = spb * BEATS_PER_SEGMENT
            fps = max(1, int(sps * TARGET_FPS))

            # 缓存参考视频原始帧（在姿态提取之后做，避免同时打开两个 VideoCapture）
            self.queue.put({'type': 'status', 'message': '📹 缓存参考视频帧...'})
            log.info(f"缓存参考视频帧...")
            self._all_raw_frames: list = []
            try:
                cap = cv2.VideoCapture(self.ref_path)
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开参考视频")
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self._all_raw_frames.append(frame)
                    frame_idx += 1
                cap.release()
                log.info(f"参考视频帧缓存完成: {frame_idx} 帧")
            except Exception as e:
                log.warning(f"缓存参考帧失败（不影响跟练）: {e}")
                self._all_raw_frames = []
            raw_total = max(len(self._all_raw_frames), 1)
            ratio = raw_total / max(nref, 1)

        self.ref_segs = []
        for sid in range(max(1, (nref + fps - 1) // fps)):
            ps, pe = sid * fps, min((sid + 1) * fps, nref)
            rs = int(ps * ratio)
            re = max(int(pe * ratio), rs + 1)
            self.ref_segs.append({
                'id': sid + 1,
                'start': ps, 'end': pe,
                'raw_start': rs, 'raw_end': re,
            })

        # 从内存切片加载第一段帧
        self._ref_raw_cache = []
        self._ref_raw_seg_id = 0
        self._load_seg_raw_frames(0)

        self.queue.put({
            'type': 'status',
            'message': f'📐 参考: {nref}姿态帧, {len(self.ref_segs)}段',
        })

        # 滑动窗口（window_step = window_max 确保每段从头开始收集完整的 150 帧，
        # 避免前一段的旧帧混入，导致后续段参考视频只显示极少数帧）
        self.window: List = []
        self.window_max = 150
        self.window_step = self.window_max  # 每段清空窗口，独立收集
        self.current_seg = 0
        self.seg_results: List[Dict] = []
        self.frame_counter = 0
        self._seg_frame_no = 0  # 当前段内帧计数 (不受窗口切片影响)

        # 预导入 C++ 扩展（避免 _loop 延迟导入时与摄像头线程冲突导致段错误）
        from dance_scoring.core.scorer import Scorer
        from dance_scoring.core.config import Config, Z_AXIS_WEIGHT
        from dance_scoring.core.frame import PoseFrame
        import mediapipe as mp
        self._Scorer = Scorer
        self._Config = Config
        self._PoseFrame = PoseFrame
        self._mp = mp
        self._Z_AXIS_WEIGHT = Z_AXIS_WEIGHT

    # ======== 主循环 ========

    def _loop(self):
        log.info("进入实时跟练主循环")
        try:
            self._loop_impl()
        except Exception as e:
            import traceback
            log.error(f"LiveWorker._loop 异常: {traceback.format_exc()}")
            self.queue.put({'type': 'error', 'message': str(e)})

    def _loop_impl(self):
        """实际主循环逻辑。"""
        scorer_cfg = self._Config()
        scorer = self._Scorer(scorer_cfg, bpm=self.bpm, alignment_method=self.alignment)
        self._send_segment_start()

        # 摄像头预热（避免首帧段错误）
        for _ in range(10):
            if self._cancel.is_set():
                return
            try:
                f = self.camera.read()
                if f is not None and hasattr(f, 'shape') and f.shape[0] > 0:
                    break
            except Exception:
                pass
            time.sleep(0.2)

        while not self._cancel.is_set():
            try:
                user_frame = self.camera.read()
            except Exception:
                time.sleep(0.5)
                continue
            if user_frame is None:
                if self._cancel.is_set():
                    break
                time.sleep(0.05)
                continue

            # 单帧姿态提取
            pf = self._extract_frame(user_frame)
            if pf is not None:
                self.window.append(pf)
                self._seg_frame_no += 1

                # 窗口满 → 打分 → 切换下一段
                if len(self.window) >= self.window_max:
                    result = self._score_window(scorer)
                    if result:
                        self.seg_results.append(result)
                        self.current_seg = result.get('segment_id',
                                                      self.current_seg)
                        # 终端日志
                        print(f"[跟练] 段{result['segment_id']} 得分:{result['score']:.1f} "
                              f"{result['qualified']} | {result['correction_text'][:50]}")
                        data = {'type': 'result', 'user_frame': user_frame, **result}
                        self.queue.put(data)

                        if self.current_seg >= len(self.ref_segs):
                            self._send_cycle_complete()
                            # 等待 GUI 点击继续
                            self._paused.wait(timeout=300)  # 最多等5分钟
                            if self._cancel.is_set():
                                break
                            self._paused.clear()
                            self.current_seg = 0
                            self.seg_results = []
                            print(f"[跟练] 全部 {len(self.ref_segs)} 段完成，开始新一轮")
                        self._send_segment_start()
                    self.window = self.window[self.window_step:]

            # 获取当前段的参考帧 (循环播放)
            ref_frame = self._get_current_ref_frame()

            # 推送双画面
            self.queue.put({
                'type': 'dual_frame',
                'user_frame': user_frame,
                'ref_frame': ref_frame,
                'seg_id': self.current_seg + 1,
                'total_segs': len(self.ref_segs),
                'buffer_fill': len(self.window),
                'buffer_max': self.window_max,
            })

    def _extract_frame(self, rgb_frame):
        import mediapipe as mp
        from dance_scoring.core.frame import PoseFrame
        from dance_scoring.core.config import Z_AXIS_WEIGHT
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts = int(time.time() * 1000) % (2**31)
            res = self.extractor.det.detect_for_video(mp_img, ts)
            if (res.pose_world_landmarks and
                    len(res.pose_world_landmarks) > 0 and
                    len(res.pose_world_landmarks[0]) >= 33):
                kp3d = np.zeros((33, 3), dtype=np.float32)
                cf = np.zeros(33, dtype=np.float32)
                for i, lm in enumerate(res.pose_world_landmarks[0][:33]):
                    kp3d[i] = [lm.x, lm.y, lm.z * Z_AXIS_WEIGHT]
                    cf[i] = lm.visibility if hasattr(lm, 'visibility') else 1.0
                self.frame_counter += 1
                return PoseFrame(self.frame_counter, kp3d, cf)
        except Exception:
            pass
        return None

    def _load_seg_raw_frames(self, seg_idx):
        """从内存切片加载指定段的原始视频帧（无 seek，避免关键帧定位问题）。"""
        if seg_idx >= len(self.ref_segs):
            return
        seg = self.ref_segs[seg_idx]
        rs, re = seg['raw_start'], seg['raw_end']
        # 从一次性预读的帧列表中切片
        all_frames = getattr(self, '_all_raw_frames', [])
        rs = max(0, min(rs, len(all_frames)))
        re = max(rs + 1, min(re, len(all_frames)))
        self._ref_raw_cache = all_frames[rs:re]
        self._ref_raw_seg_id = seg_idx

    def _get_current_ref_frame(self):
        """获取当前段当前应显示的参考视频帧 (循环播放)。"""
        if not self.ref_segs:
            return None
        idx = min(self.current_seg, len(self.ref_segs) - 1)
        # 切换段时重新加载
        if idx != self._ref_raw_seg_id:
            self._load_seg_raw_frames(idx)
        if not self._ref_raw_cache:
            return None
        pos = self._seg_frame_no % max(len(self._ref_raw_cache), 1)
        return self._ref_raw_cache[pos]

    def _send_segment_start(self):
        if not self.ref_segs:
            return
        self._seg_frame_no = 0
        idx = min(self.current_seg, len(self.ref_segs) - 1)
        seg = self.ref_segs[idx]
        self.queue.put({
            'type': 'segment_start',
            'seg_id': seg['id'],
            'total_segs': len(self.ref_segs),
        })

    def _send_cycle_complete(self):
        """汇总本轮所有段的得分、缺陷、纠正建议，发送到 GUI 展示。"""
        if not self.seg_results:
            return
        from dance_scoring.core.correction import JOINT_NAMES_CN
        # 平均分
        scores = [r['score'] for r in self.seg_results if r.get('score', 0) > 0]
        avg_score = sum(scores) / max(len(scores), 1)
        # 合格统计
        qualified = sum(1 for r in self.seg_results if r.get('qualified') == '合格')
        total = len(self.seg_results)

        # 汇总关节偏差（使用 Deviation 对象，已过滤可见性+面部）
        all_joint_sums: Dict[int, float] = {}
        all_joint_counts: Dict[int, int] = {}
        all_corrections: List[str] = []
        all_skipped: Dict[str, int] = {}  # 跨段频繁跳过的关节

        for r in self.seg_results:
            # 优先使用 Deviation 对象（更准确），回退旧格式
            deviations = r.get('deviations', [])
            if deviations:
                for d in deviations:
                    all_joint_sums[d.joint_idx] = (
                        all_joint_sums.get(d.joint_idx, 0.0) + abs(d.deviation_deg))
                    all_joint_counts[d.joint_idx] = (
                        all_joint_counts.get(d.joint_idx, 0) + 1)
            else:
                # 兼容旧格式
                devs = r.get('joint_deviations', {})
                for j, d in devs.items():
                    all_joint_sums[j] = all_joint_sums.get(j, 0.0) + abs(d)
                    all_joint_counts[j] = all_joint_counts.get(j, 0) + 1
            corr = r.get('correction_text', '')
            if corr:
                all_corrections.append(corr)
            for name in r.get('skipped_joints', []):
                all_skipped[name] = all_skipped.get(name, 0) + 1

        # 找出偏差最大的 TOP5 关节
        joint_avgs = {j: all_joint_sums[j] / max(all_joint_counts[j], 1)
                      for j in all_joint_sums}
        top_joints = sorted(joint_avgs.items(), key=lambda x: x[1], reverse=True)[:5]
        weakest = [(j, d, JOINT_NAMES_CN.get(j, f'关节{j}')) for j, d in top_joints
                   if d > 3.0]

        # 频繁跳过的关节（≥50% 段都跳过）
        frequent_skipped = [name for name, cnt in all_skipped.items()
                           if cnt >= max(total // 2, 1)]

        # 汇总纠正文本（去重取最相关）
        seen_corr = set()
        unique_corr = []
        for c in all_corrections:
            key = c[:30]
            if key not in seen_corr:
                seen_corr.add(key)
                unique_corr.append(c)

        summary = {
            'type': 'cycle_complete',
            'avg_score': round(avg_score, 1),
            'qualified': qualified,
            'total': total,
            'weakest_joints': weakest,
            'corrections': unique_corr[:3],
            'frequent_skipped': frequent_skipped,
            'seg_results': [{'id': r['segment_id'], 'score': r['score'],
                            'qualified': r['qualified']}
                           for r in self.seg_results],
        }

        # 先发摘要（GUI 立即展示），再发 AI 请求（主线程队列顺序执行）
        self.queue.put(summary)
        self._request_ai_plan(summary, weakest, all_corrections)

        skipped_info = f" 跳过:{frequent_skipped}" if frequent_skipped else ""
        print(f"[跟练] 本轮完成: 均分{avg_score:.1f}, {qualified}/{total}合格, "
              f"薄弱部位: {[w[2] for w in weakest[:3]]}{skipped_info}")

    def _request_ai_plan(self, summary: dict, weakest: list, corrections: list):
        """通过队列将 AI 请求交给主线程执行（避免 LiveWorker 线程中的 OpenVINO/MediaPipe 冲突）。"""
        log.info("请求 AI 训练计划（主线程模式）...")
        self.queue.put({'type': 'status', 'message': '🤖 AI 正在生成训练计划...'})

        ctx_lines = [
            "你是真人舞蹈教练，刚看完学员的一轮跟练。根据数据给训练建议：",
            f"均分{summary['avg_score']:.0f}分，{summary['qualified']}/{summary['total']}段通过。",
        ]
        if weakest:
            parts = ["薄弱部位:"]
            for j, d, name in weakest[:3]:
                parts.append(f"{name}")
            ctx_lines.append(" ".join(parts))
        ctx_lines += [
            "要求：先一句话鼓励，再给2-3条具体练习建议。像聊天一样自然，不超过100字。",
        ]

        system_ctx = "\n".join(ctx_lines)
        user_msg = "请根据以上数据生成训练计划。"

        # 通过队列发送给主线程（主线程加载模型不会和 MediaPipe 冲突）
        self.queue.put({
            'type': '_do_ai_plan',
            'system_ctx': system_ctx,
            'user_msg': user_msg,
        })

    def _score_window(self, scorer):
        if not self.ref_segs:
            return None
        idx = min(self.current_seg, len(self.ref_segs) - 1)
        seg = self.ref_segs[idx]
        ref_seg = self.ref_poses[seg['start']:seg['end']]
        if len(ref_seg) < 5:
            self.current_seg = min(self.current_seg + 1, len(self.ref_segs) - 1)
            return None

        overall, scored_segs, low, path = scorer.score(ref_seg, self.window)

        # 直接使用 Scorer 的输出（已含可见性过滤 + 面部排除 + 关节去重 + 纠正文本）
        scorer_seg = scored_segs[0] if scored_segs else {}
        deviations = scorer_seg.get('deviations', [])
        skipped_joints = scorer_seg.get('skipped_joints', [])
        correction_text = scorer_seg.get('correction_text', '')
        seg_score = scorer_seg.get('score', 0)

        # 保持兼容旧格式的 joint_deviations dict
        devs = {d.joint_idx: d.deviation_deg for d in deviations}

        return {
            'segment_id': seg['id'],
            'score': seg_score,
            'qualified': '合格' if seg_score >= 60 else '不合格',
            'joint_deviations': devs,
            'deviations': deviations,          # List[Deviation]
            'skipped_joints': skipped_joints,  # List[str]
            'correction_text': correction_text,
            'path_length': len(path),
        }

    def _cleanup(self):
        if hasattr(self, 'camera') and self.camera:
            self.camera.close()
        # 释放预加载的全部视频帧内存
        if hasattr(self, '_all_raw_frames'):
            self._all_raw_frames = []


# ============================================================
# 实时跟练 Panel
# ============================================================

class LivePanel(ttk.Frame):

    CANVAS_W = 320
    CANVAS_H = 240

    def __init__(self, master, reference_path: str = ""):
        super().__init__(master)
        self._ref_path = reference_path
        self._running = False
        self._worker: Optional[LiveWorker] = None
        self._queue: queue.Queue = queue.Queue()
        self._ref_seg_count = 0
        self._tk_img_user = None
        self._tk_img_ref = None
        self._last_result = None
        self._container = None
        self._import_frame = None
        self._live_frame = None
        self._import_video = None
        self._build()

    def _build(self):
        self._container = ttk.Frame(self)
        self._container.pack(fill=tk.BOTH, expand=True)

        # ── 导入界面 (默认显示) ──
        self._build_import()
        # ── 跟练界面 (选择视频后才显示) ──
        self._build_live()

        if self._ref_path and os.path.exists(self._ref_path):
            self._show_live()
        else:
            self._show_import()

    def _build_import(self):
        """视频导入界面 — 首次打开时显示。"""
        self._import_frame = ttk.Frame(self._container)
        inner = ttk.Frame(self._import_frame, padding=20)
        inner.pack(expand=True)
        ttk.Label(inner, text="🎬", font=FONTS["icon"]).pack()
        ttk.Label(inner, text="实时跟练", font=FONTS["heading"]).pack(pady=(8, 4))
        ttk.Label(inner, text="选择参考视频后即可开始对照练习",
                 font=FONTS["body"], foreground=COLORS["text_secondary"]
                 ).pack(pady=(0, 12))

        # 大号导入按钮 — 最直观的入口
        self.btn_select_ref = ttk.Button(
            inner,
            text="📁  选择参考视频",
            command=self._sel_ref_file,
            bootstyle="warning",
            style="warning.TButton",
        )
        self.btn_select_ref.pack(pady=4)

        # 选中后显示文件名 + 进入跟练按钮
        self.lbl_selected = ttk.Label(inner, text="",
                                      font=FONTS["body"], foreground=COLORS["success"])
        self.lbl_selected.pack(pady=(8, 0))
        self.btn_go_live = ttk.Button(inner, text="▶  开始跟练",
                                       command=self._show_live,
                                       bootstyle="success", width=20)
        self.btn_go_live.pack(pady=4)
        self.btn_go_live.config(state=tk.DISABLED)

        # 如果已有路径，预填充
        if self._ref_path and os.path.exists(self._ref_path):
            self._on_import_select(self._ref_path)

    def _sel_ref_file(self):
        path = filedialog.askopenfilename(
            title="选择参考视频",
            filetypes=VIDEO_FILTERS)
        if path:
            self._on_import_select(path)

    def _build_live(self):
        """跟练双画面界面 — 选择视频后显示。"""
        self._live_frame = ttk.Frame(self._container)
        self._main = ttk.Frame(self._live_frame)
        self._main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左侧: 双画面
        self._left = ttk.Frame(self._main)
        self._left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(self._left, text="📷 我的画面", font=FONTS["body"],
                 bootstyle="info").pack(anchor=tk.W, pady=(0, 2))
        self.canvas_user = tk.Canvas(self._left, width=self.CANVAS_W,
                                     height=self.CANVAS_H, bg="black",
                                     highlightthickness=1,
                                     highlightbackground=COLORS["bg_input"])
        self.canvas_user.pack(pady=(0, 4))
        ttk.Label(self._left, text="📹 参考示范", font=FONTS["body"],
                 bootstyle="warning").pack(anchor=tk.W, pady=(4, 2))
        self.canvas_ref = tk.Canvas(self._left, width=self.CANVAS_W,
                                    height=self.CANVAS_H, bg="black",
                                    highlightthickness=1,
                                    highlightbackground=COLORS["bg_input"])
        self.canvas_ref.pack()

        # 右侧面板
        self._right = ttk.Frame(self._main, width=240)
        self._right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        self._right.pack_propagate(False)

        # 段进度
        f1 = ttk.Labelframe(self._right, text="📊 进度", padding=8)
        f1.pack(fill=tk.X, pady=(0, 8))
        self.lbl_segment = ttk.Label(f1, text="段: - / -", font=FONTS["body"])
        self.lbl_segment.pack(anchor=tk.W)
        self.lbl_buffer = ttk.Label(f1, text="采集: 0/0 帧", font=("", 9))
        self.lbl_buffer.pack(anchor=tk.W)

        # 得分
        f2 = ttk.Labelframe(self._right, text="🎯 当前段得分", padding=8)
        f2.pack(fill=tk.X, pady=(0, 8))
        self.lbl_score = ttk.Label(f2, text="--.-", font=FONTS["score"])
        self.lbl_score.pack()
        self.progress_bar = ttk.Progressbar(f2, mode='determinate', length=180)
        self.progress_bar.pack(fill=tk.X, pady=(4, 0))

        # 纠正
        f3 = ttk.Labelframe(self._right, text="✏️ 纠正建议", padding=8)
        f3.pack(fill=tk.X, pady=(0, 8))
        self.lbl_correction = ttk.Label(f3, text="等待开始...", wraplength=200)
        self.lbl_correction.pack(anchor=tk.W)

        # 薄弱
        f4 = ttk.Labelframe(self._right, text="💪 薄弱部位", padding=8)
        f4.pack(fill=tk.X, pady=(0, 8))
        self.lbl_weak = ttk.Label(f4, text="-", wraplength=200)
        self.lbl_weak.pack(anchor=tk.W)

        # ── 轮次摘要面板（默认隐藏，一轮完成后覆盖右侧）──
        self._build_summary_panel()

        # 底部控制栏
        ctrl = ttk.Frame(self._live_frame, padding=4)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X)
        self.btn_start = ttk.Button(ctrl, text="▶ 开始", command=self._start,
                                    bootstyle="success", width=8)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        self.btn_stop = ttk.Button(ctrl, text="⏹ 停止", command=self._stop,
                                   bootstyle="danger-outline", width=8)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(ctrl, text="📁 换参考", command=self._show_import,
                   bootstyle="secondary-outline", width=9).pack(side=tk.LEFT, padx=2)

        self.lbl_status = ttk.Label(ctrl, text="就绪 · 请点击 ▶ 开始",
                                   bootstyle="secondary")
        self.lbl_status.pack(side=tk.RIGHT, padx=4)
        self._show_placeholders()

    def _do_ai_plan(self, system_ctx: str, user_msg: str):
        """生成训练计划。只用 3B（1.5B 对长 prompt 太慢，模型导出时未开 KV-cache）。"""
        q = self._queue

        def _call():
            try:
                from LLM.model_manager import load_model, chat_safe, get_current_model_key
                # 3B 有 KV-cache，长 prompt 也快
                if get_current_model_key() != "3b":
                    load_model("3b")
                plan = chat_safe(system_ctx, [], user_msg, max_wait=120)
                if not plan or plan.startswith('抱歉'):
                    plan = '未生成有效计划，请重试。'
                q.put({'type': 'ai_plan', 'plan': plan})
                log.info(f"AI 训练计划完成 ({len(plan)} 字)")
            except Exception as e:
                q.put({'type': 'ai_plan', 'plan': f'⚠️ AI 异常: {e}'})
        threading.Thread(target=_call, daemon=True).start()

    # ── 轮次摘要面板 ──

    def _build_summary_panel(self):
        """构建轮次完成后的摘要面板（覆盖右侧信息区，带滚动条）。"""
        self._summary_frame = ttk.Frame(self._main)

        # Canvas + 滚动条
        self._sum_canvas = tk.Canvas(self._summary_frame, bg=COLORS["bg"],
                                      highlightthickness=0)
        sum_scrollbar = ttk.Scrollbar(self._summary_frame, orient=tk.VERTICAL,
                                       command=self._sum_canvas.yview)
        self._sum_canvas.configure(yscrollcommand=sum_scrollbar.set)

        self._sum_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sum_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 内容 frame 放在 canvas 内部
        self._sum_inner = ttk.Frame(self._sum_canvas, padding=16)
        self._sum_window = self._sum_canvas.create_window(
            (0, 0), window=self._sum_inner, anchor=tk.NW)

        # 让 canvas 宽度跟随 inner frame
        def _on_configure(event):
            self._sum_canvas.itemconfig(self._sum_window, width=event.width)
        self._sum_canvas.bind("<Configure>", _on_configure)

        def _on_inner_configure(event):
            self._sum_canvas.configure(scrollregion=self._sum_canvas.bbox("all"))
        self._sum_inner.bind("<Configure>", _on_inner_configure)

        # 鼠标滚轮滚动 — 局部绑定，不影响全局
        def _on_mousewheel(event):
            if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
                self._sum_canvas.yview_scroll(-1, "units")
            elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
                self._sum_canvas.yview_scroll(1, "units")
        # Linux: Button-4/Button-5 on the canvas itself
        self._sum_canvas.bind("<Button-4>", _on_mousewheel)
        self._sum_canvas.bind("<Button-5>", _on_mousewheel)
        # 将 canvas 的滚动绑定传播给 inner frame 的每个子 widget
        def _bind_scroll(w):
            w.bind("<Button-4>", _on_mousewheel)
            w.bind("<Button-5>", _on_mousewheel)
            for child in w.winfo_children():
                _bind_scroll(child)
        self._sum_inner.bind("<Map>", lambda e: _bind_scroll(self._sum_inner))

        inner = self._sum_inner

        # 标题
        self._sum_title = ttk.Label(inner, text="", font=FONTS["heading"])
        self._sum_title.pack(anchor=tk.W, pady=(0, 4))

        # 总分 + 合格统计
        self._sum_score = ttk.Label(inner, text="", font=("", 36, "bold"))
        self._sum_score.pack(anchor=tk.W, pady=(4, 2))
        self._sum_qual = ttk.Label(inner, text="", font=FONTS["body"])
        self._sum_qual.pack(anchor=tk.W, pady=(0, 12))

        # 各段得分条
        ttk.Label(inner, text="── 各段得分 ──", font=FONTS["body_bold"]
                 ).pack(anchor=tk.W, pady=(0, 4))
        self._sum_bars = ttk.Frame(inner)
        self._sum_bars.pack(fill=tk.X, pady=(0, 12))

        # 薄弱部位
        ttk.Label(inner, text="── 薄弱部位 ──", font=FONTS["body_bold"]
                 ).pack(anchor=tk.W, pady=(0, 4))
        self._sum_weak = ttk.Label(inner, text="", font=FONTS["body"],
                                    wraplength=420)
        self._sum_weak.pack(anchor=tk.W, pady=(0, 12))

        # 关键纠正建议
        ttk.Label(inner, text="── 改进建议 ──", font=FONTS["body_bold"]
                 ).pack(anchor=tk.W, pady=(0, 4))
        self._sum_corr = ttk.Label(inner, text="", font=FONTS["body"],
                                    wraplength=420)
        self._sum_corr.pack(anchor=tk.W, pady=(0, 12))

        # AI 训练计划
        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(4, 8))
        ttk.Label(inner, text="🤖 AI 训练计划", font=FONTS["body_bold"],
                 foreground=COLORS["accent"]).pack(anchor=tk.W, pady=(0, 4))
        self._sum_ai_plan = tk.Label(inner, text="", font=FONTS["body"],
                                      fg="#000000", bg="#0F172A",
                                      wraplength=420, justify=tk.LEFT)
        self._sum_ai_plan.pack(anchor=tk.W, pady=(0, 16))

        # 操作按钮
        btn_row = ttk.Frame(inner)
        btn_row.pack(fill=tk.X)
        self.btn_restart = ttk.Button(btn_row, text="🔄  重新跟练",
                                      command=self._restart_practice,
                                      bootstyle="success", style="success.TButton")
        self.btn_restart.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="🚪  退出跟练", command=self._exit_practice,
                   bootstyle="danger-outline").pack(side=tk.LEFT)

    def _restart_practice(self):
        """重新跟练：停止当前 worker，清空结果，从头开始。"""
        if self._worker:
            self._worker.cancel()
            self._worker = None
        self._hide_summary()
        self._last_result = None
        self._ref_seg_count = 0
        self._show_placeholders()
        self._start()

    def _exit_practice(self):
        """退出跟练：停止 worker，回到导入界面。"""
        if self._worker:
            self._worker.cancel()
            self._worker = None
        self._running = False
        self._hide_summary()
        self._show_placeholders()
        self._show_import()

    def _show_summary(self, data: dict):
        """显示轮次摘要，隐藏左右画面。"""
        self._left.pack_forget()
        self._right.pack_forget()
        # 显示摘要
        self._summary_frame.pack(fill=tk.BOTH, expand=True)

        # 填充数据
        avg = data.get('avg_score', 0)
        qualified = data.get('qualified', 0)
        total = data.get('total', 0)
        weakest = data.get('weakest_joints', [])
        corrections = data.get('corrections', [])
        seg_results = data.get('seg_results', [])

        # 标题
        if avg >= 80:
            emoji, grade = "🌟", "优秀"
        elif avg >= 60:
            emoji, grade = "✅", "良好"
        else:
            emoji, grade = "💪", "需加强"
        self._sum_title.config(
            text=f"{emoji} 本轮练习完成 — {grade}")

        # 总分
        color = COLORS["success"] if avg >= 60 else COLORS["danger"]
        self._sum_score.config(text=f"{avg:.1f} 分", foreground=color)
        self._sum_qual.config(
            text=f"通过: {qualified}/{total} 段  |  "
                 f"合格率: {int(qualified/max(total,1)*100)}%")

        # 各段得分条
        for w in self._sum_bars.winfo_children():
            w.destroy()
        for r in seg_results:
            bar_frame = ttk.Frame(self._sum_bars)
            bar_frame.pack(fill=tk.X, pady=2)
            ttk.Label(bar_frame, text=f"段{r['id']}", font=("", 9),
                     width=4, anchor=tk.W).pack(side=tk.LEFT)
            bar = ttk.Progressbar(bar_frame, mode='determinate', length=160)
            bar.pack(side=tk.LEFT, padx=4)
            bar['value'] = min(r['score'], 100)
            passed = "✅" if r.get('qualified') == '合格' else "❌"
            ttk.Label(bar_frame, text=f"{r['score']:.0f}{passed}",
                     font=("", 9)).pack(side=tk.LEFT, padx=4)

        # 薄弱部位
        if weakest:
            lines = []
            for j, d, name in weakest:
                icon = "🔴" if d > 10 else "🟡"
                lines.append(f"{icon} {name} (偏差 {d:.1f}°)")
            self._sum_weak.config(text="\n".join(lines))
        else:
            self._sum_weak.config(text="无明显薄弱部位 ✨")

        # 纠正建议
        if corrections:
            self._sum_corr.config(
                text="\n".join(f"⚡ {c[:100]}" for c in corrections))
        else:
            self._sum_corr.config(text="全部段表现良好，继续保持！")

        # 频繁跳过的部位（可见帧不足）
        frequent_skipped = data.get('frequent_skipped', [])
        if frequent_skipped:
            self._sum_weak.config(
                text=self._sum_weak.cget("text") +
                f"\n\n⚠️ 未入镜: {'、'.join(frequent_skipped[:5])}")

        # AI 训练计划（初始：生成中）
        self._sum_ai_plan.config(
            text="⏳ AI 正在根据本轮数据生成训练计划...",
            fg="#666666")

        # 更新状态栏
        self.lbl_status.config(
            text=f"🏆 本轮均分 {avg:.1f} · 点击继续开始下一轮",
            bootstyle="success")

    def _hide_summary(self):
        """隐藏摘要，恢复左右画面。"""
        self._summary_frame.pack_forget()
        # 重置 canvas 滚动位置
        try:
            self._sum_canvas.yview_moveto(0)
        except Exception:
            pass
        self._left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        self.lbl_status.config(text="新一轮练习开始...", bootstyle="info")

    # ── 模式切换 ──

    def _show_import(self):
        self._running = False
        if self._worker:
            self._worker.cancel()
            self._worker = None
        if self._live_frame:
            self._live_frame.pack_forget()
        self._import_frame.pack(fill=tk.BOTH, expand=True)
        if self._ref_path and os.path.exists(self._ref_path):
            self._on_import_select(self._ref_path)
        else:
            self._ref_path = ""
            self.lbl_selected.config(text="")
            self.btn_go_live.config(state=tk.DISABLED)

    def _show_live(self):
        if not self._ref_path or not os.path.exists(self._ref_path):
            messagebox.showwarning("提示", "请先选择参考视频")
            return
        self._import_frame.pack_forget()
        self._live_frame.pack(fill=tk.BOTH, expand=True)
        self.lbl_status.config(
            text=f"参考: {os.path.basename(self._ref_path)[:20]} · 请点击 ▶ 开始",
            bootstyle="info")

    def _on_import_select(self, path):
        self._ref_path = path
        self.btn_go_live.config(state=tk.NORMAL)
        self.btn_go_live.config(text=f"▶  开始跟练 — {os.path.basename(path)[:15]}")

    def set_reference(self, path: str):
        self._ref_path = path
        if path and os.path.exists(path):
            self._on_import_select(path)
            self._show_live()

    # ---------- 控制 ----------

    def _start(self):
        with guard("开始跟练"):
            if not self._ref_path or not os.path.exists(self._ref_path):
                messagebox.showwarning("提示", "请先选择参考视频")
                return
            if self._worker is not None and self._worker.is_alive():
                return
            log.info(f"启动跟练: ref={self._ref_path}")
            self._running = True
            self._last_result = None
            self.btn_start.config(state=tk.DISABLED)

            # 跟练 AI 训练计划固定用 3B（1.5B 长 prompt 太慢）
            self.lbl_status.config(text="⏳ 加载 AI 模型...", bootstyle="info")
            # 必须在启动 MediaPipe 前加载 LLM
            def _preload_then_start():
                try:
                    from LLM.model_manager import load_model
                    load_model("3b")
                except Exception as e:
                    log.warning(f"模型预加载失败: {e}")
                self.master.after(0, self._do_start)
            threading.Thread(target=_preload_then_start, daemon=True).start()

    def _do_start(self):
        """LLM 加载完成后在主线程启动 LiveWorker。"""
        self.lbl_status.config(text="初始化摄像头...", bootstyle="info")
        self.update_idletasks()
        self._queue = queue.Queue()
        self._worker = LiveWorker(self._ref_path, self._queue, camera_id=0)
        self._worker.start()
        self.after(100, self._poll)

    def _stop(self):
        with guard("停止跟练"):
            log.info("停止跟练")
            self._running = False
            if self._worker:
                self._worker.cancel()
                self._worker = None
            try:
                self.btn_start.config(state=tk.NORMAL)
            except Exception:
                pass
            try:
                self.lbl_status.config(text="已停止", bootstyle="secondary")
            except Exception:
                pass
            try:
                self._show_placeholders()
            except Exception:
                pass
            try:
                self._hide_summary()
            except Exception:
                pass

    # ---------- 轮询队列 ----------

    def _poll(self):
        if not self._running:
            return
        try:
            while True:
                msg = self._queue.get_nowait()
                t = msg.get('type', '')
                try:
                    if t == 'error':
                        messagebox.showerror("跟练错误", msg.get('message', '未知错误'))
                        self._stop()
                    elif t == 'status':
                        self.lbl_status.config(text=str(msg.get('message', ''))[:40])
                    elif t == 'segment_start':
                        self._ref_seg_count = msg.get('total_segs', self._ref_seg_count)
                        sid = msg.get('seg_id', 0)
                        self.lbl_segment.config(text=f"▶ 段: {sid} / {max(self._ref_seg_count,1)}")
                        self.lbl_score.config(text="--.-", bootstyle="primary")
                        self.lbl_correction.config(text="跟练中...")
                    elif t == 'result':
                        self._last_result = msg
                        self._ref_seg_count = max(self._ref_seg_count, msg.get('segment_id', 0))
                        print(f"[GUI] 收到段{msg.get('segment_id',0)}结果 得分:{msg.get('score',0):.1f}")
                        self._update_display(msg)
                    elif t == 'dual_frame':
                        uf, rf = msg.get('user_frame'), msg.get('ref_frame')
                        if uf is not None:
                            self._draw_canvas(self.canvas_user, uf, 'user', msg)
                        if rf is not None:
                            self._draw_canvas(self.canvas_ref, rf, 'ref', msg)
                        buf = msg.get('buffer_fill', 0)
                        buf_max = max(msg.get('buffer_max', 1), 1)
                        self.lbl_buffer.config(text=f"采集: {buf}/{buf_max} 帧")
                        self.progress_bar['value'] = min(buf / buf_max * 100, 100)
                    elif t == 'summary':
                        # 兼容旧的 summary 消息：仅更新状态栏
                        results = msg.get('seg_results', [])
                        if results:
                            scores = [r['score'] for r in results if r.get('score', 0) > 0]
                            if scores:
                                avg = sum(scores) / len(scores)
                                p = sum(1 for r in results if r.get('qualified') == '合格')
                                self.lbl_status.config(text=f"🏆 均分:{avg:.1f} | 通过:{p}/{len(results)}")
                    elif t == 'cycle_complete':
                        self._show_summary(msg)
                    elif t == 'ai_plan':
                        try:
                            self._sum_ai_plan.config(
                                text=msg.get('plan', ''),
                                fg="#000000")
                            self.lbl_status.config(text="✅ 训练计划已生成", bootstyle="success")
                        except Exception:
                            pass
                    elif t == '_do_ai_plan':
                        # 主线程处理 AI 请求（避免 LiveWorker 线程中 OpenVINO/MediaPipe 冲突）
                        self._do_ai_plan(msg.get('system_ctx', ''),
                                        msg.get('user_msg', ''))
                except Exception:
                    pass  # widget 可能已被销毁
        except queue.Empty:
            pass

        if self._running:
            self.after(67, self._poll)

    def _update_display(self, data: dict):
        score = data.get('score', 0)
        seg_id = data.get('segment_id', 0)
        corr_text = data.get('correction_text', '')
        skipped = data.get('skipped_joints', [])
        score_str = f"{score:.1f}" if score > 0 else "--.-"

        self.lbl_segment.config(text=f"段: {seg_id} / {max(self._ref_seg_count,1)}")
        self.lbl_score.config(text=score_str)
        if corr_text:
            if corr_text.startswith(f"第{seg_id}段："):
                corr_text = corr_text[len(f"第{seg_id}段："):]
            # 截断前先去掉可能很长的跳过提示
            display_text = corr_text[:80]
            self.lbl_correction.config(text=display_text)
        else:
            self.lbl_correction.config(text="跟练中...")

        # 薄弱部位：优先使用 Deviation 对象（已过滤可见性+面部）
        deviations = data.get('deviations', [])
        if deviations:
            from dance_scoring.core.correction import JOINT_NAMES_CN
            sd = sorted(deviations, key=lambda d: abs(d.deviation_deg), reverse=True)[:3]
            self.lbl_weak.config(text="  ".join(
                f"{'🔴' if abs(d.deviation_deg)>10 else '🟡'} {d.joint_name}"
                for d in sd if abs(d.deviation_deg) > 5) or "-")
        else:
            # 兼容旧格式
            joint_devs = data.get('joint_deviations', {})
            if joint_devs:
                from dance_scoring.core.correction import JOINT_NAMES_CN
                sd = sorted(joint_devs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                self.lbl_weak.config(text="  ".join(
                    f"{'🔴' if abs(d)>10 else '🟡'} {JOINT_NAMES_CN.get(j,f'关节{j}')}"
                    for j, d in sd if abs(d) > 5) or "-")
            else:
                self.lbl_weak.config(text="-")

    def _draw_canvas(self, canvas, frame, tag, data):
        import cv2
        from PIL import Image, ImageTk
        try:
            h, w = frame.shape[:2]
            scale = min(self.CANVAS_W / w, self.CANVAS_H / h)
            nw, nh = int(w * scale), int(h * scale)
            if tag == 'user':
                display = frame  # UsbCamera.read() 已返回 RGB
            else:
                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display = cv2.resize(display, (nw, nh))
            img = Image.fromarray(display)
            photo = ImageTk.PhotoImage(img)
            canvas.delete("all")
            x, y = (self.CANVAS_W - nw) // 2, (self.CANVAS_H - nh) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=photo)
            if tag == 'user':
                self._tk_img_user = photo
            else:
                self._tk_img_ref = photo
            label = "📷 我" if tag == 'user' else "📹 参考"
            canvas.create_text(5, 5, anchor=tk.NW, text=label,
                              fill="#eaeaea", font=("", 9, "bold"))
            if tag == 'user' and self._last_result:
                score = self._last_result.get('score', 0)
                if score > 0:
                    color = "#0f9b58" if score >= 60 else "#e94560"
                    canvas.create_text(self.CANVAS_W // 2, 22,
                                      text=f"{score:.1f}", fill=color,
                                      font=("Helvetica", 24, "bold"))
        except Exception:
            pass  # cv2/PIL 处理失败时跳过该帧

    def _show_placeholders(self):
        for c in [self.canvas_user, self.canvas_ref]:
            c.delete("all")
            c.create_rectangle(0, 0, self.CANVAS_W, self.CANVAS_H,
                              fill="black", outline=COLORS["bg_input"])
            c.create_text(self.CANVAS_W // 2, self.CANVAS_H // 2,
                         text="🎬" if c == self.canvas_user else "📹",
                         fill=COLORS["text_muted"], font=("", 28))


# ============================================================
# 实时跟练弹窗
# ============================================================

class LiveApp(ttk.Toplevel):
    def __init__(self, master, reference_path: str = ""):
        super().__init__(master)
        self.title("🎬 实时跟练 — 双画面")
        self.geometry("860x700")
        self.minsize(700, 560)
        self._panel = LivePanel(self, reference_path)
        self._panel.pack(fill=tk.BOTH, expand=True)
        if reference_path:
            self._panel.set_reference(reference_path)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        try:
            self.grab_set()
        except Exception:
            pass
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    def _on_close(self):
        if hasattr(self, '_panel') and self._panel is not None:
            self._panel._stop()
        if hasattr(self, 'extractor') and self.extractor is not None:
            try:
                self.extractor.close()
            except Exception:
                pass
        self.destroy()

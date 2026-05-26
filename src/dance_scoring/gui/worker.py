# gui/worker.py — 后台线程封装，避免阻塞 GUI

import threading
import os

from dance_scoring.video.info import get_video_info
from dance_scoring.video.beat_detector import detect_beats_from_audio, detect_beats_from_motion
from dance_scoring.video.splitter import get_beat_segments, calculate_segments_fixed, extract_slow_segment
from dance_scoring.video.merger import merge_videos
from dance_scoring.core.config import BEATS_PER_SEGMENT, SLOW_SPEED


class Worker:
    """后台任务基类，支持进度回调和完成回调"""

    def __init__(self, on_progress=None, on_done=None):
        self._thread = None
        self._cancel = False
        self.on_progress = on_progress
        self.on_done = on_done

    def start(self):
        self._cancel = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel = True

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        raise NotImplementedError

    def _report(self, percent, msg=""):
        if self.on_progress:
            self.on_progress(int(percent), msg)

    def _finish(self, success, result=None, error=""):
        if self.on_done:
            self.on_done(success, result, error)


class SplitWorker(Worker):
    """视频分割后台任务"""

    def __init__(self, ref_path, bpm, output_dir, on_progress=None, on_done=None):
        super().__init__(on_progress, on_done)
        self.ref_path = ref_path
        self.bpm = bpm
        self.output_dir = output_dir

    def _run(self):
        try:
            self._report(0, "读取视频信息...")
            fps, frames, duration, w, h = get_video_info(self.ref_path)

            if self._cancel:
                return

            segments = None
            used_method = f"固定BPM={self.bpm}"
            final_bpm = self.bpm

            self._report(5, "音频节拍检测...")
            result = detect_beats_from_audio(self.ref_path)
            if result is not None:
                beat_times, tempo = result
                segments = get_beat_segments(beat_times, duration, BEATS_PER_SEGMENT)
                if segments:
                    used_method = f"音频节拍 (BPM={tempo:.1f})"
                    final_bpm = tempo

            if segments is None and not self._cancel:
                self._report(15, "运动检测...")
                result = detect_beats_from_motion(self.ref_path)
                if result is not None:
                    beat_times, tempo = result
                    segments = get_beat_segments(beat_times, duration, BEATS_PER_SEGMENT)
                    if segments:
                        used_method = f"运动检测 (BPM≈{tempo:.0f})"
                        final_bpm = tempo

            if segments is None:
                segments = calculate_segments_fixed(duration, self.bpm)
                final_bpm = self.bpm

            if self._cancel:
                return

            self._report(30, f"生成{len(segments)}段慢动作...")
            os.makedirs(self.output_dir, exist_ok=True)
            all_clips = []

            for idx, seg in enumerate(segments):
                if self._cancel:
                    return
                sid = seg['id']
                out_path = os.path.join(self.output_dir, f"ref_seg_{sid:02d}_slow.mp4")
                extract_slow_segment(self.ref_path, seg['start'], seg['end'], out_path)
                all_clips.append(out_path)
                pct = 30 + int(50 * (idx + 1) / len(segments))
                self._report(pct, f"第{sid}段...")

            self._report(85, "合并片段...")
            merged = os.path.join(self.output_dir, "all_segments_merged.mp4")
            merge_videos(all_clips, merged)

            self._report(100, "分割完成")
            self._finish(True, {
                'segments': segments,
                'clips': all_clips,
                'merged': merged,
                'method': used_method,
                'bpm': final_bpm,
                'duration': duration,
                'fps': fps,
                'width': w,
                'height': h,
            })

        except Exception as e:
            self._finish(False, None, str(e))


class ScoreWorker(Worker):
    """评分后台任务"""

    def __init__(self, ref_path, user_path, bpm, threshold, segments_dir,
                 on_progress=None, on_done=None):
        super().__init__(on_progress, on_done)
        self.ref_path = ref_path
        self.user_path = user_path
        self.bpm = bpm
        self.threshold = threshold
        self.segments_dir = segments_dir

    def _run(self):
        try:
            from dance_scoring.core.config import Config
            from dance_scoring.core.extractor import PoseExtractor, download_model
            from dance_scoring.core.scorer import Scorer

            download_model()

            total_steps = 3
            step_weight = {0: 0, 1: 33, 2: 66}

            def extract_progress(step_name, step_idx):
                def cb(pct):
                    base = step_weight[step_idx]
                    scaled = base + int(pct * (33 if step_idx < 2 else 34) / 100)
                    self._report(scaled, step_name)
                return cb

            if self._cancel:
                return

            self._report(0, "提取参考视频姿态...")
            cfg = Config(score_threshold=self.threshold)
            ref = PoseExtractor(cfg).extract(
                self.ref_path,
                progress_callback=extract_progress("提取参考视频姿态...", 0)
            )

            if self._cancel:
                return

            self._report(33, "提取用户视频姿态...")
            user = PoseExtractor(cfg).extract(
                self.user_path,
                progress_callback=extract_progress("提取用户视频姿态...", 1)
            )

            if self._cancel:
                return

            self._report(66, "DTW对齐+评分...")
            scorer = Scorer(cfg, bpm=self.bpm)

            def score_progress(pct, msg):
                if self._cancel:
                    return
                base = 66
                self._report(base + int(pct * 34 / 100), msg)

            overall, segs, low, path = scorer.score(ref, user, progress_callback=score_progress)

            if self._cancel:
                return

            self._report(90, "输出练习视频...")
            from dance_scoring.core.segments import extract_clips_from_segments
            files = extract_clips_from_segments(segs, self.segments_dir, cfg=cfg)

            result = {
                'overall': overall,
                'segs': segs,
                'low': low,
                'path': path,
                'files': files,
                'ref_frames': len(ref),
                'user_frames': len(user),
                'path_len': len(path),
            }

            self._report(100, "评分完成")
            self._finish(True, result)

        except Exception as e:
            self._finish(False, None, str(e))

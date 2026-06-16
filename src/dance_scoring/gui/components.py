# gui/components.py — 可复用 GUI 组件 (ttkbootstrap 主题)

import tkinter as tk
from tkinter import messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from typing import Optional
import cv2
from PIL import Image, ImageTk
import os
import subprocess

from dance_scoring.gui.theme import COLORS, FONTS

def _open_dir(path: str):
    """跨平台打开目录。"""
    if os.path.isdir(path):
        if os.name == 'nt':
            os.startfile(path)
        else:
            subprocess.Popen(['xdg-open', path])
    else:
        os.makedirs(path, exist_ok=True)
        if os.name == 'nt':
            os.startfile(path)
        else:
            subprocess.Popen(['xdg-open', path])


class VideoPreview(tk.Frame):
    """视频预览面板：显示缩略图 + 视频信息 + 播放控制"""

    PREVIEW_W = 320
    PREVIEW_H = 240

    def __init__(self, parent, label="Video"):
        super().__init__(parent, bd=1, relief=tk.SUNKEN)
        self.label_text = label
        self.video_path = None
        self._cap = None
        self._playing = False
        self._after_id = None
        self._build()

    def _build(self):
        header = tk.Label(self, text=self.label_text, font=("", 11, "bold"))
        header.pack(pady=(4, 0))

        self.canvas = tk.Canvas(self, width=self.PREVIEW_W, height=self.PREVIEW_H,
                                bg="black", highlightthickness=0)
        self.canvas.pack(padx=8, pady=4)

        self.lbl_name = tk.Label(self, text="未导入", fg="gray")
        self.lbl_name.pack()
        self.lbl_info = tk.Label(self, text="", fg="gray")
        self.lbl_info.pack()

        self.btn_play = tk.Button(self, text="▶ 预览播放", command=self._toggle_play,
                                  state=tk.DISABLED)
        self.btn_play.pack(pady=4)

    def load(self, path):
        self.stop()
        self.video_path = path
        try:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise ValueError("无法打开视频")
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total / fps if fps > 0 else 0

            name = os.path.basename(path)
            if len(name) > 28:
                name = name[:25] + "..."
            self.lbl_name.config(text=name, fg="black")
            self.lbl_info.config(
                text=f"时长: {duration:.1f}s | {w}×{h} | {fps:.0f}fps", fg="gray")

            ret, frame = self._cap.read()
            if ret:
                self._show_frame(frame)
            self.btn_play.config(state=tk.NORMAL)
        except Exception as e:
            self.video_path = None
            self._cap = None
            self.lbl_name.config(text="加载失败", fg="red")
            self.lbl_info.config(text=str(e), fg="red")
            self.btn_play.config(state=tk.DISABLED)

    def _show_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_rgb, (nw, nh))
        img = Image.fromarray(resized)
        self._tk_img = ImageTk.PhotoImage(img)
        x = (self.PREVIEW_W - nw) // 2
        y = (self.PREVIEW_H - nh) // 2
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor=tk.NW, image=self._tk_img)

    def _toggle_play(self):
        if self._playing:
            self.stop()
        else:
            self._play()

    def _play(self):
        if self._cap is None:
            return
        self._playing = True
        self.btn_play.config(text="⬛ 停止播放")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 15
        self._play_delay = int(1000 / min(fps, 25))

        if self._cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._play_frame()

    def _play_frame(self):
        if not self._playing:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        if ret:
            self._show_frame(frame)
        self._after_id = self.after(self._play_delay, self._play_frame)

    def stop(self):
        self._playing = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.btn_play.config(text="▶ 预览播放")

    def destroy(self):
        self.stop()
        if self._cap:
            self._cap.release()
        super().destroy()


class ProgressDialog(tk.Toplevel):
    """进度条弹窗"""

    def __init__(self, parent, title="处理中"):
        super().__init__(parent)
        self.title(title)
        self.geometry("380x120")
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        self.var_pct = tk.StringVar(value="0%")
        self.var_msg = tk.StringVar(value="准备...")

        tk.Label(self, textvariable=self.var_msg, font=("", 10)).pack(pady=(16, 8))
        self.progress = ttk.Progressbar(self, mode='determinate', length=320)
        self.progress.pack(pady=4)
        tk.Label(self, textvariable=self.var_pct, font=("", 9)).pack()

        self.grab_set()
        self.update()

    def update_progress(self, percent, msg=""):
        self.var_pct.set(f"{percent}%")
        self.var_msg.set(msg)
        self.progress['value'] = percent
        self.update_idletasks()


class ScoreResultDialog(tk.Toplevel):
    """评分结果展示窗口"""

    def __init__(self, parent, result):
        super().__init__(parent)
        self.title("评分结果")
        self.resizable(False, False)
        self.transient(parent)

        self.result = result
        self._build()

    def _build(self):
        r = self.result
        main = tk.Frame(self, padx=16, pady=12)
        main.pack()

        overall = r['overall']
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

        ttk.Label(main, text=f"{overall:.1f}",
                 font=FONTS["score"], bootstyle="primary").pack()
        ttk.Label(main, text=f"总评: {grade}",
                 font=("", 14, "bold"), bootstyle="primary").pack(pady=(0, 4))

        info = f"参考帧数: {r['ref_frames']}  |  用户帧数: {r['user_frames']}  |  DTW对齐: {r['path_len']}对"
        tk.Label(main, text=info, fg="gray").pack(pady=(0, 12))

        cols = [("段号", 5), ("时间", 15), ("得分", 8), ("判定", 10)]
        hdr = tk.Frame(main)
        hdr.pack(fill=tk.X)
        for text, w in cols:
            tk.Label(hdr, text=text, width=w, font=("", 9, "bold"),
                     anchor=tk.W, bd=1, relief=tk.RIDGE, padx=3).pack(side=tk.LEFT)

        pass_score = 60.0
        fail_count = 0
        for seg in r['segs']:
            row = tk.Frame(main)
            row.pack(fill=tk.X)
            t = f"{seg['start_time']:.1f}s-{seg['end_time']:.1f}s"
            passed = seg['score'] >= pass_score
            if not passed:
                fail_count += 1
            q = "✅合格" if passed else "❌不合格"
            fg_color = "green" if passed else "red"

            for text, w in [(str(seg['id']), 5), (t, 15), (f"{seg['score']:.1f}", 8), (q, 10)]:
                tk.Label(row, text=text, width=w, anchor=tk.W,
                         fg=fg_color, bd=1, relief=tk.RIDGE, padx=3).pack(side=tk.LEFT)

        tk.Frame(main, height=8).pack()
        if fail_count > 0:
            tk.Label(main, text=f"❌ {fail_count}/{len(r['segs'])}段不合格，已输出练习视频",
                     fg="red").pack(pady=(8, 4))
        else:
            tk.Label(main, text="🎉 全部合格！", fg="green").pack(pady=(8, 4))

        btn_frame = tk.Frame(main)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="打开练习视频目录",
                  command=lambda: _open_dir("output/low_score_clips")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="打开分段目录",
                  command=lambda: _open_dir("output/segments")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="关闭", command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _open_dir_deprecated(self, relative_path):
        _open_dir(os.path.join(os.getcwd(), relative_path))


class SegmentListDialog(tk.Toplevel):
    """分割结果列表窗口"""

    def __init__(self, parent, result, ref_path):
        super().__init__(parent)
        self.title("分割结果")
        self.geometry("500x420")
        self.resizable(False, False)
        self.transient(parent)

        self.result = result
        self.ref_path = ref_path
        self._playing_cap = None
        self._playing = False
        self._build()

    def _build(self):
        main = tk.Frame(self, padx=12, pady=8)
        main.pack(fill=tk.BOTH, expand=True)

        r = self.result
        tk.Label(main, text=f"分段方式: {r['method']}  |  {len(r['segments'])}段",
                 font=("", 10)).pack(anchor=tk.W, pady=(0, 4))

        list_frame = tk.Frame(main)
        list_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        hdr = tk.Frame(list_frame)
        hdr.pack(fill=tk.X)
        for text, w in [("段号", 5), ("时间段", 14), ("文件名", 26)]:
            tk.Label(hdr, text=text, width=w, font=("", 9, "bold"),
                     anchor=tk.W, bd=1, relief=tk.RIDGE, padx=3).pack(side=tk.LEFT)

        self.listbox = tk.Listbox(list_frame, width=52, height=12, font=("Consolas", 9))
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=2)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)

        for seg in r['segments']:
            sid = seg['id']
            name = f"ref_seg_{sid:02d}_slow.mp4"
            self.listbox.insert(tk.END, f"  第{sid:2d}段  {seg['start']:6.2f}s - {seg['end']:6.2f}s    {name}")

        right = tk.Frame(main, width=200)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        self.play_canvas = tk.Canvas(right, width=160, height=120, bg="black")
        self.play_canvas.pack(pady=4)

        self.btn_play_seg = tk.Button(right, text="▶ 播放选中段",
                                      command=self._toggle_seg_play, width=16)
        self.btn_play_seg.pack(pady=2)

        self.lbl_seg_info = tk.Label(right, text="点击列表选择分段", fg="gray", wraplength=180)
        self.lbl_seg_info.pack(pady=2)

        tk.Button(right, text="用系统播放器打开",
                  command=self._open_external, width=16).pack(pady=2)

        tk.Button(main, text="关闭", command=self._on_close).pack(pady=(8, 0))

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        seg = self.result['segments'][idx]
        self.selected_seg = seg
        self.lbl_seg_info.config(
            text=f"第{seg['id']}段\n{seg['start']:.2f}s - {seg['end']:.2f}s")
        self._stop_seg_play()
        self._show_seg_frame(seg)

    def _show_seg_frame(self, seg):
        cap = cv2.VideoCapture(self.ref_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sf = int(seg['start'] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2.resize(frame_rgb, (160, 120)))
            self._tk_img = ImageTk.PhotoImage(img)
            self.play_canvas.delete("all")
            self.play_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    def _toggle_seg_play(self):
        if self._playing:
            self._stop_seg_play()
            return
        if not hasattr(self, 'selected_seg'):
            return
        self._playing = True
        self.btn_play_seg.config(text="⬛ 停止")
        self._play_seg_loop()

    def _play_seg_loop(self):
        if not self._playing:
            return
        seg = self.selected_seg
        if self._playing_cap is None:
            self._playing_cap = cv2.VideoCapture(self.ref_path)
            fps = self._playing_cap.get(cv2.CAP_PROP_FPS)
            sf = int(seg['start'] * fps)
            ef = int(seg['end'] * fps)
            self._playing_cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
            self._seg_ef = ef

        if self._playing_cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._seg_ef:
            self._playing_cap.set(cv2.CAP_PROP_POS_FRAMES,
                                  int(seg['start'] * self._playing_cap.get(cv2.CAP_PROP_FPS)))

        ret, frame = self._playing_cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2.resize(frame_rgb, (160, 120)))
            self._tk_img = ImageTk.PhotoImage(img)
            self.play_canvas.delete("all")
            self.play_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        self._seg_after = self.after(80, self._play_seg_loop)

    def _stop_seg_play(self):
        self._playing = False
        if hasattr(self, '_seg_after'):
            self.after_cancel(self._seg_after)
        if self._playing_cap:
            self._playing_cap.release()
            self._playing_cap = None
        self.btn_play_seg.config(text="▶ 播放选中段")

    def _open_external(self):
        if not hasattr(self, 'selected_seg'):
            return
        seg = self.selected_seg
        clip = os.path.join("output", "segments", f"ref_seg_{seg['id']:02d}_slow.mp4")
        if os.path.exists(clip):
            os.startfile(clip)
        else:
            messagebox.showinfo("提示", f"文件不存在: {clip}")

    def _on_close(self):
        self._stop_seg_play()
        self.destroy()


# ============================================================
# HUD 运动风新增组件
# ============================================================

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

class VideoImporter(ttk.Frame):
    """视频导入组件 — 点击选择 + 最近文件列表。"""

    DROP_H = 120

    def __init__(self, master, label="视频", on_select=None):
        super().__init__(master)
        self._label = label
        self._callback = on_select
        self._selected: Optional[str] = None
        self._build()

    def _build(self):
        l = ttk.Label(self, text=f"📹 {self._label}", font=FONTS["body_bold"])
        l.pack(anchor=tk.W, pady=(0, 4))

        self.drop_frame = tk.Frame(self, bg=COLORS["card"], height=self.DROP_H,
                                   highlightthickness=1,
                                   highlightbackground=COLORS["border"])
        self.drop_frame.pack(fill=tk.X)
        self.drop_frame.pack_propagate(False)

        self.drop_label = tk.Label(
            self.drop_frame, text="📹\n点击选择视频文件",
            bg=COLORS["card"], fg=COLORS["text_muted"],
            font=FONTS["body"], justify=tk.CENTER, cursor="hand2")
        self.drop_label.pack(expand=True)

        for w in [self.drop_frame, self.drop_label]:
            w.bind("<Button-1>", self._on_click)

        self.info_label = tk.Label(self, text="", bg=COLORS["bg"],
                                   fg=COLORS["text_secondary"], font=FONTS["small"],
                                   anchor=tk.W)

        # 最近列表
        ttk.Label(self, text="── 最近视频 ──", font=FONTS["small"],
                 foreground=COLORS["text_muted"]).pack(anchor=tk.W, pady=(6, 2))
        self.recent_frame = ttk.Frame(self)
        self.recent_frame.pack(fill=tk.X)
        try:
            self._populate_recent()
        except Exception:
            ttk.Label(self.recent_frame, text="  (扫描视频目录失败)",
                     font=FONTS["small"], foreground=COLORS["text_muted"]).pack()

    def _on_click(self, event=None):
        path = filedialog.askopenfilename(
            title=f"选择{self._label}",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                       ("所有文件", "*.*")])
        if path:
            self.set_file(path)

    def _populate_recent(self):
        import cv2
        candidates = []
        for d in ["videos", os.path.expanduser("~/Videos"),
                  os.path.expanduser("~/Video")]:
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.lower().endswith(VIDEO_EXTS):
                        candidates.append(os.path.join(d, fn))
        seen = set()
        for p in candidates[:8]:
            if p not in seen:
                seen.add(p)
                try:
                    cap = cv2.VideoCapture(p)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    dur = frames / fps if fps > 0 else 0
                    cap.release()
                    dur_s = f"{int(dur)}s"
                    fn = os.path.basename(p)[:25]
                    text = f"  📹 {fn}  {dur_s}  {frames}帧"
                except Exception:
                    text = f"  📹 {os.path.basename(p)[:30]}"

                row = ttk.Label(self.recent_frame, text=text, font=FONTS["small"],
                               anchor=tk.W, cursor="hand2")
                row.pack(fill=tk.X, pady=1)
                row.bind("<Button-1>", lambda e, path=p: self.set_file(path))

        browse_btn = ttk.Label(self.recent_frame, text="  📂 浏览其他文件...",
                              font=FONTS["small"], foreground=COLORS["accent"],
                              cursor="hand2", anchor=tk.W)
        browse_btn.pack(fill=tk.X, pady=(2, 0))
        browse_btn.bind("<Button-1>", self._on_click)

    def set_file(self, path: str):
        import cv2
        self._selected = path
        fn = os.path.basename(path)[:35]
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                info = fn
            else:
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                dur = frames / fps if fps > 0 else 0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                info = f"{fn}\n{frames}帧 · {dur:.1f}s · {w}×{h} · {fps:.0f}fps"
        except Exception:
            info = fn

        self.drop_label.config(text=f"📹\n{info}", fg=COLORS["text"])
        self.drop_frame.config(highlightbackground=COLORS["success"],
                               highlightthickness=2)
        self.info_label.config(text=f"✅ 已选: {fn}")
        self.info_label.pack(fill=tk.X, pady=(2, 0))

        if self._callback:
            self._callback(path)

    def get_file(self) -> Optional[str]:
        return self._selected

    def clear(self):
        self._selected = None
        self.drop_label.config(text="📹\n点击选择视频文件",
                              fg=COLORS["text_muted"])
        self.drop_frame.config(highlightbackground=COLORS["border"],
                               highlightthickness=1)
        self.info_label.pack_forget()


class ScoreDisplay(ttk.Frame):
    """大号得分显示 — 颜色按阈值自动切换。"""

    def __init__(self, master, size="medium"):
        super().__init__(master)
        font = FONTS["score_medium"] if size == "medium" else FONTS["score_large"]
        self.lbl_score = ttk.Label(self, text="--.-", font=font, bootstyle="primary")
        self.lbl_score.pack()
        self.lbl_grade = ttk.Label(self, text="", font=FONTS["body"], bootstyle="secondary")
        self.lbl_grade.pack()

    def set(self, score: float, threshold: float = 60.0):
        if score <= 0:
            self.lbl_score.config(text="--.-", bootstyle="primary")
            self.lbl_grade.config(text="")
            return
        self.lbl_score.config(text=f"{score:.1f}")
        if score >= threshold:
            self.lbl_score.config(bootstyle="success")
            grade = "⭐ 优秀" if score >= 85 else "👍 合格"
            self.lbl_grade.config(text=grade, bootstyle="success")
        else:
            self.lbl_score.config(bootstyle="danger")
            grade = "⚠️ 需改进" if score >= 40 else "💪 需重练"
            self.lbl_grade.config(text=grade, bootstyle="danger")


class SegmentBar(tk.Frame):
    """段得分水平条 — 绿/红色按分数填充，可点击。"""

    BAR_H = 32

    def __init__(self, master, seg_id: int, score: float, start_time: float,
                 end_time: float, threshold: float = 60.0, on_click=None):
        super().__init__(master, bg=COLORS["bg"])
        self.seg_id = seg_id
        self._threshold = threshold
        self._on_click = on_click
        self._build(score, start_time, end_time)

    def _build(self, score, start_time, end_time):
        passed = score >= self._threshold
        fill_color = COLORS["success"] if passed else COLORS["danger"]
        fill_pct = min(100, max(2, score))
        cw = 160

        # 段号
        l1 = tk.Label(self, text=f"段{self.seg_id}", font=FONTS["body_bold"],
                     bg=COLORS["bg"], fg=COLORS["text"], width=4, anchor=tk.W)
        l1.pack(side=tk.LEFT)

        # Canvas 条
        cv = tk.Canvas(self, width=cw, height=self.BAR_H,
                      bg=COLORS["input_bg"], highlightthickness=0)
        cv.pack(side=tk.LEFT, padx=4)
        cv.create_rectangle(0, 0, int(cw * fill_pct / 100), self.BAR_H,
                           fill=fill_color, outline="")
        cv.create_text(cw // 2, self.BAR_H // 2, text=f"{score:.1f}",
                      fill=COLORS["text"], font=FONTS["body_bold"])

        # 图标
        icon = tk.Label(self, text="✅" if passed else "❌", font=("", 14),
                       bg=COLORS["bg"])
        icon.pack(side=tk.LEFT, padx=4)

        # 时间
        tl = tk.Label(self, text=f"{start_time:.1f}s-{end_time:.1f}s",
                     font=FONTS["small"], bg=COLORS["bg"], fg=COLORS["text_muted"])
        tl.pack(side=tk.LEFT, padx=4)

        # 点击事件
        for w in [self, l1, cv, icon, tl]:
            w.bind("<Button-1>", lambda e, sid=self.seg_id:
                   self._on_click and self._on_click(sid))

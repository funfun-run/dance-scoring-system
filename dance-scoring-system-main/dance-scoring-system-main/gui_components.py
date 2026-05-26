# gui_components.py - 可复用GUI组件

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os


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
        # 标题
        header = tk.Label(self, text=self.label_text, font=("", 11, "bold"))
        header.pack(pady=(4, 0))

        # 预览画布
        self.canvas = tk.Canvas(self, width=self.PREVIEW_W, height=self.PREVIEW_H,
                                bg="black", highlightthickness=0)
        self.canvas.pack(padx=8, pady=4)

        # 信息标签
        self.lbl_name = tk.Label(self, text="未导入", fg="gray")
        self.lbl_name.pack()
        self.lbl_info = tk.Label(self, text="", fg="gray")
        self.lbl_info.pack()

        # 播放按钮
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

            # 显示第一帧
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

        # 总评
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

        tk.Label(main, text=f"总评: {overall:.1f}/100  {grade}",
                 font=("", 14, "bold")).pack(pady=(0, 4))

        info = f"参考帧数: {r['ref_frames']}  |  用户帧数: {r['user_frames']}  |  DTW对齐: {r['path_len']}对"
        tk.Label(main, text=info, fg="gray").pack(pady=(0, 12))

        # 表格标题
        cols = [("段号", 5), ("时间", 15), ("得分", 8), ("判定", 10)]
        hdr = tk.Frame(main)
        hdr.pack(fill=tk.X)
        for text, w in cols:
            tk.Label(hdr, text=text, width=w, font=("", 9, "bold"),
                     anchor=tk.W, bd=1, relief=tk.RIDGE, padx=3).pack(side=tk.LEFT)

        # 数据行
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

        # 底部信息
        tk.Frame(main, height=8).pack()
        if fail_count > 0:
            tk.Label(main, text=f"❌ {fail_count}/{len(r['segs'])}段不合格，已输出练习视频",
                     fg="red").pack(pady=(8, 4))
        else:
            tk.Label(main, text="🎉 全部合格！", fg="green").pack(pady=(8, 4))

        btn_frame = tk.Frame(main)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="打开练习视频目录",
                  command=lambda: self._open_dir("output/low_score_clips")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="打开分段目录",
                  command=lambda: self._open_dir("output/segments")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="关闭", command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _open_dir(self, relative_path):
        path = os.path.join(os.path.dirname(__file__), relative_path)
        if os.path.isdir(path):
            os.startfile(path)
        else:
            messagebox.showinfo("提示", f"目录不存在: {path}")


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

        # 列表 → 滚动区域
        list_frame = tk.Frame(main)
        list_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # 表头
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

        # 播放区
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
        # 显示第一帧
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

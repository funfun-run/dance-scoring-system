# gui_main.py - 舞蹈评分系统 GUI 主程序

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

# 确保当前目录在搜索路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui_components import VideoPreview, ProgressDialog, ScoreResultDialog, SegmentListDialog
from gui_worker import SplitWorker, ScoreWorker
from split_8beats import get_video_info

VIDEO_FILTERS = [("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                 ("所有文件", "*.*")]


class MainApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("舞蹈评分系统 v6.2")
        self.root.geometry("960x640")
        self.root.minsize(860, 560)

        self.ref_path = None
        self.user_path = None
        self.split_result = None
        self.score_result = None
        self._worker = None

        self._build()
        self._center_window()

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"+{x}+{y}")

    def _build(self):
        # 主容器
        main = tk.Frame(self.root, padx=8, pady=8)
        main.pack(fill=tk.BOTH, expand=True)

        # 标题
        title = tk.Label(main, text="舞蹈评分系统 v6.2", font=("", 16, "bold"))
        title.pack(pady=(0, 8))

        # 中间三栏
        body = tk.Frame(main)
        body.pack(fill=tk.BOTH, expand=True)

        # 左栏：参考视频
        self.preview_ref = VideoPreview(body, label="参考视频")
        self.preview_ref.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))

        # 中栏：用户视频
        self.preview_user = VideoPreview(body, label="用户视频")
        self.preview_user.pack(side=tk.LEFT, fill=tk.Y, padx=4)

        # 右栏：操作面板
        panel = tk.LabelFrame(body, text="操作面板", padx=10, pady=8)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0), ipadx=6)

        # 视频导入
        tk.Label(panel, text="视频导入", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))
        tk.Button(panel, text="导入参考视频", command=self._import_ref,
                  width=18, height=2).pack(pady=2)
        tk.Button(panel, text="导入用户视频", command=self._import_user,
                  width=18, height=2).pack(pady=2)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        # 参数设置
        tk.Label(panel, text="参数设置", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))

        f1 = tk.Frame(panel)
        f1.pack(fill=tk.X, pady=2)
        tk.Label(f1, text="BPM:").pack(side=tk.LEFT)
        self.var_bpm = tk.StringVar(value="120")
        tk.Entry(f1, textvariable=self.var_bpm, width=8).pack(side=tk.RIGHT)

        f2 = tk.Frame(panel)
        f2.pack(fill=tk.X, pady=2)
        tk.Label(f2, text="阈值:").pack(side=tk.LEFT)
        self.var_threshold = tk.StringVar(value="50")
        tk.Entry(f2, textvariable=self.var_threshold, width=8).pack(side=tk.RIGHT)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        # 操作按钮
        tk.Label(panel, text="执行操作", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.btn_split = tk.Button(panel, text="分割参考视频", command=self._do_split,
                                   width=18, height=2, bg="#e8f0fe")
        self.btn_split.pack(pady=2)
        self.btn_score = tk.Button(panel, text="开始评分", command=self._do_score,
                                   width=18, height=2, bg="#e8ffe8")
        self.btn_score.pack(pady=2)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        tk.Button(panel, text="查看上次评分结果", command=self._show_last_result,
                  width=18).pack(pady=2)
        tk.Button(panel, text="打开练习视频目录", command=lambda: self._open_dir("output/low_score_clips"),
                  width=18).pack(pady=2)
        tk.Button(panel, text="打开分段目录", command=lambda: self._open_dir("output/segments"),
                  width=18).pack(pady=2)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪 — 请导入参考视频和用户视频")
        status = tk.Label(self.root, textvariable=self.status_var, bd=1,
                          relief=tk.SUNKEN, anchor=tk.W, padx=8, pady=3)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    # ── 视频导入 ──

    def _import_ref(self):
        path = filedialog.askopenfilename(title="选择参考视频", filetypes=VIDEO_FILTERS)
        if path:
            self.ref_path = path
            self.preview_ref.load(path)
            self._set_status(f"参考视频: {os.path.basename(path)}")

    def _import_user(self):
        path = filedialog.askopenfilename(title="选择用户视频", filetypes=VIDEO_FILTERS)
        if path:
            self.user_path = path
            self.preview_user.load(path)
            self._set_status(f"用户视频: {os.path.basename(path)}")

    # ── 分割 ──

    def _do_split(self):
        if not self.ref_path:
            messagebox.showwarning("提示", "请先导入参考视频")
            return
        try:
            bpm = int(self.var_bpm.get())
        except ValueError:
            messagebox.showwarning("提示", "BPM 请输入整数")
            return

        self._disable_buttons()
        self._set_status("正在分割参考视频...")

        progress = ProgressDialog(self.root, "分割参考视频")

        def on_progress(pct, msg):
            progress.update_progress(pct, msg)

        def on_done(success, result, error):
            progress.destroy()
            self._enable_buttons()
            if success:
                self.split_result = result
                self._set_status(f"分割完成 — {len(result['segments'])}段 | {result['method']}")
                SegmentListDialog(self.root, result, self.ref_path)
            else:
                messagebox.showerror("分割失败", error)
                self._set_status("分割失败")

        self._worker = SplitWorker(
            self.ref_path, bpm, "output/segments", on_progress, on_done
        )
        self._worker.start()

    # ── 评分 ──

    def _do_score(self):
        if not self.ref_path:
            messagebox.showwarning("提示", "请先导入参考视频")
            return
        if not self.user_path:
            messagebox.showwarning("提示", "请先导入用户视频")
            return
        try:
            bpm = int(self.var_bpm.get())
        except ValueError:
            messagebox.showwarning("提示", "BPM 请输入整数")
            return
        try:
            threshold = float(self.var_threshold.get())
        except ValueError:
            messagebox.showwarning("提示", "阈值请输入数字")
            return

        self._disable_buttons()
        self._set_status("正在评分...")

        progress = ProgressDialog(self.root, "舞蹈评分进行中")

        def on_progress(pct, msg):
            progress.update_progress(pct, msg)

        def on_done(success, result, error):
            progress.destroy()
            self._enable_buttons()
            if success:
                self.score_result = result
                self._set_status(
                    f"总评: {result['overall']:.1f}/100 | "
                    f"{len([s for s in result['segs'] if s['score']<60])}/{len(result['segs'])}段不合格"
                )
                ScoreResultDialog(self.root, result)
            else:
                messagebox.showerror("评分失败", error)
                self._set_status("评分失败")

        self._worker = ScoreWorker(
            self.ref_path, self.user_path, bpm, threshold, "output/segments",
            on_progress, on_done
        )
        self._worker.start()

    # ── 辅助 ──

    def _show_last_result(self):
        if self.score_result:
            ScoreResultDialog(self.root, self.score_result)
        else:
            messagebox.showinfo("提示", "还没有评分结果，请先运行评分")

    def _open_dir(self, relative_path):
        path = os.path.join(os.path.dirname(__file__), relative_path)
        if os.path.isdir(path):
            os.startfile(path)
        else:
            os.makedirs(path, exist_ok=True)
            os.startfile(path)

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _disable_buttons(self):
        self.btn_split.config(state=tk.DISABLED)
        self.btn_score.config(state=tk.DISABLED)

    def _enable_buttons(self):
        self.btn_split.config(state=tk.NORMAL)
        self.btn_score.config(state=tk.NORMAL)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MainApp()
    app.run()

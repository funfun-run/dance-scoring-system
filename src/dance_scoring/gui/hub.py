# gui/hub.py — 主 Hub 窗口 (HUD 运动风 v3)

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import os

from dance_scoring.gui.theme import COLORS, FONTS
from dance_scoring.gui.logger import log, guard, safe_after, safe_thread
from dance_scoring.gui.panels import (
    ScoringPanel, ReviewPanel, SplitPanel,
    SettingsPanel, NPUPanel, ModelPanel, PerfPanel,
)

_CARD_SPECS = [
    {"key": "live",   "icon": "🎬", "title": "实时跟练",
     "desc": "摄像头对照参考视频练习\n实时打分 + 纠正建议",
     "btn": "进入跟练", "accent": True},
    {"key": "score",  "icon": "📊", "title": "离线评分",
     "desc": "导入两段视频\n分段对比打分",
     "btn": "导入视频", "accent": False},
    {"key": "review", "icon": "📁", "title": "练习回顾",
     "desc": "低分片段慢放\n纠正建议详解",
     "btn": "查看回顾", "accent": False},
    {"key": "split",  "icon": "✂️", "title": "视频分割",
     "desc": "参考视频按八拍切段\n导出练习片段",
     "btn": "开始分割", "accent": False},
]

_TOOLBAR = [
    {"key": "settings", "icon": "⚙️", "label": "设置"},
    {"key": "npu",      "icon": "🖥️", "label": "NPU"},
    {"key": "model",    "icon": "🔧", "label": "模型"},
    {"key": "perf",     "icon": "📈", "label": "性能"},
]

_PANEL_MAP = {
    "score":    ScoringPanel,
    "review":   ReviewPanel,
    "split":    SplitPanel,
    "settings": SettingsPanel,
    "npu":      NPUPanel,
    "model":    ModelPanel,
    "perf":     PerfPanel,
}


class Hub(ttk.Frame):

    def __init__(self, master):
        super().__init__(master)
        self._current_panel = None
        self.last_score_result: dict = {}
        self.selected_model: str = "3b"
        with guard("Hub._build"):
            self._build()
        log.info("Hub 初始化完成")

    def _build(self):
        # ── 顶部标题栏 ──
        top = ttk.Frame(self, padding=(20, 16, 20, 0))
        top.pack(fill=tk.X)
        ttk.Label(top, text="🕺  舞蹈评分系统",
                 font=FONTS["title"]).pack(side=tk.LEFT)

        # 模型选择器（右上）— 醒目的可点击标签
        model_frame = ttk.Frame(top)
        model_frame.pack(side=tk.RIGHT, padx=(8, 0))
        self.lbl_model = ttk.Label(model_frame,
                                    font=FONTS["body_bold"],
                                    foreground=COLORS["accent"],
                                    cursor="hand2")
        self.lbl_model.pack(side=tk.LEFT)
        self.lbl_model.bind("<Button-1>", self._on_model_toggle)
        self._refresh_model_list()

        # NPU 指示灯
        npu_frame = ttk.Frame(top)
        npu_frame.pack(side=tk.RIGHT, padx=(8, 0))
        self.lbl_npu = ttk.Label(npu_frame, text="○ NPU", font=FONTS["small"])
        self.lbl_npu.pack(side=tk.RIGHT)
        self.after(30000, self._update_npu_dot)

        # ── 内容区 ──
        self.content = ttk.Frame(self)
        self.content.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)
        self._show_cards()

        # ── 底部工具栏 ──
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=20, pady=(0, 12))
        for item in _TOOLBAR:
            ttk.Button(
                bar, text=f"{item['icon']}  {item['label']}",
                command=lambda k=item['key']: self._show_panel(k),
                bootstyle="secondary-outline",
            ).pack(side=tk.LEFT, padx=3)

        # ── 状态栏 ──
        status = ttk.Frame(self)
        status.pack(fill=tk.X, padx=20, pady=(0, 8))
        self.var_status = tk.StringVar(value="🟢 就绪")
        self.var_summary = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.var_status,
                 font=FONTS["small"]).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.var_summary,
                 font=FONTS["small"]).pack(side=tk.RIGHT)

    # ======== 卡片视图 ========

    def _show_cards(self):
        with guard("显示主卡片"):
            self._clear_content()
            self._current_panel = None
            grid = ttk.Frame(self.content)
            grid.pack(expand=True)
            for i, spec in enumerate(_CARD_SPECS):
                row, col = i // 2, i % 2
                card = self._make_card(grid, spec)
                card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            grid.grid_columnconfigure(0, weight=1)
            grid.grid_columnconfigure(1, weight=1)
            grid.grid_rowconfigure(0, weight=1)
            grid.grid_rowconfigure(1, weight=1)

    def _make_card(self, parent, spec):
        accent = spec.get("accent", False)
        boot = "warning" if accent else "secondary"

        frame = ttk.Frame(parent, padding=14)
        ttk.Label(frame, text=spec["icon"], font=FONTS["icon"]).pack(anchor=tk.W)
        ttk.Label(frame, text=spec["title"], font=FONTS["heading"]
                 ).pack(anchor=tk.W, pady=(4, 2))
        ttk.Label(frame, text=spec["desc"], font=FONTS["body"],
                 foreground=COLORS["text_secondary"], justify=tk.LEFT
                 ).pack(anchor=tk.W, pady=(0, 10))
        ttk.Button(
            frame, text=spec["btn"],
            command=lambda k=spec["key"]: self._on_card_click(k),
            bootstyle=boot,
        ).pack(anchor=tk.W)
        return frame

    def _on_card_click(self, key):
        with guard(f"卡片点击: {key}"):
            if key == "live":
                self._open_live()
            else:
                self._show_panel(key)

    def _open_live(self):
        with guard("打开实时跟练"):
            try:
                from dance_scoring.gui.live_view import LiveApp
                LiveApp(self.winfo_toplevel())
            except Exception as e:
                log.error(f"实时跟练启动失败: {e}")
                from tkinter import messagebox
                messagebox.showerror("实时跟练", f"启动失败: {e}")

    # ======== 面板切换 ========

    def _show_panel(self, key):
        with guard(f"切换面板: {key}"):
            self._clear_content()
            self._current_panel = key

            nav = ttk.Frame(self.content)
            nav.pack(fill=tk.X, pady=(0, 4))
            ttk.Button(nav, text="← 返回", command=self._show_cards,
                      bootstyle="link").pack(side=tk.LEFT)

            cls = _PANEL_MAP.get(key)
            if cls:
                panel = cls(self.content)
                panel.pack(fill=tk.BOTH, expand=True)
                log.debug(f"面板 {key} 创建完成")

    def _clear_content(self):
        try:
            for w in self.content.winfo_children():
                w.destroy()
        except Exception as e:
            log.warning(f"清除内容区异常: {e}")

    # ======== 模型选择 ========

    def _refresh_model_list(self):
        """刷新模型列表（纯 Python 检查，不 import C++ 库）。"""
        import os as _os
        from importlib.util import find_spec
        models = []

        ov_pkg = find_spec("optimum.intel") is not None
        ov_dir = False
        if ov_pkg:
            ov_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "LLM", "qwen2.5-1.5b-ov")
            ov_dir = _os.path.isdir(ov_path)
        # 1.5B (optimum-intel) 和 MediaPipe 底层冲突，暂禁用
        models.append({"key": "1.5b", "name": "1.5B (不可用)", "available": False})

        llama_pkg = find_spec("llama_cpp") is not None
        gguf_file = False
        if llama_pkg:
            gguf_path = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "LLM", "qwen2.5-3b-instruct-q4_k_m.gguf")
            gguf_file = _os.path.isfile(gguf_path)
        models.append({"key": "3b", "name": "3B", "available": llama_pkg and gguf_file})

        self._model_list = models
        self._update_model_display()

    def _on_model_toggle(self, event=None):
        """点击模型标签切换到下一个可用模型。"""
        available = [m for m in self._model_list if m['available']]
        if not available:
            return
        # 找到当前选中的索引，切换到下一个
        current_keys = [m['key'] for m in available]
        try:
            idx = current_keys.index(self.selected_model)
            next_idx = (idx + 1) % len(available)
        except ValueError:
            next_idx = 0
        new_model = available[next_idx]
        old_model = self.selected_model
        self.selected_model = new_model['key']
        self._update_model_display()
        log.info(f"模型切换: {old_model} → {new_model['key']}")
        self.set_status(f"🟢 模型: {new_model['name']}")

        if old_model != new_model['key']:
            safe_thread("model_unload", self._unload_old_model, m=new_model)

    def _unload_old_model(self, m):
        """后台卸载旧模型。"""
        with guard("卸载旧模型"):
            from LLM.model_manager import unload_model
            unload_model()
            self.master.after(0, lambda: self.set_status(
                f"🟢 模型: {m['name']} — 下次评分时自动加载"))

    def _update_model_display(self):
        """更新模型标签显示。"""
        try:
            m = next((x for x in self._model_list if x['key'] == self.selected_model), None)
            if m:
                color = COLORS["text_muted"] if not m['available'] else COLORS["accent"]
                speed = "快 ~2s" if m['key'] == '1.5b' else '准 ~8s'
                self.lbl_model.config(
                    text=f"🤖 {m['name']} ({speed}) | 点击切换",
                    foreground=color)
        except Exception:
            pass

    # ======== NPU 指示灯 ========

    def _update_npu_dot(self):
        try:
            from dance_scoring.platform.npu import NPUManager
            avail = NPUManager.available()
            self.lbl_npu.config(
                text=f"● NPU" if avail else "○ NPU",
                foreground=COLORS["success"] if avail else COLORS["text_muted"],
            )
        except Exception as e:
            log.debug(f"NPU 检测失败: {e}")
            try:
                self.lbl_npu.config(text="○ NPU")
            except Exception:
                pass
        # 30 秒后重试 — 使用安全包装
        safe_after(self, "NPU检测", 30000, self._update_npu_dot)

    # ======== 数据共享 ========

    def set_score_result(self, overall: float, segs: list, threshold: float,
                         ref_path: str = "", user_path: str = "",
                         corrections: dict = None, joint_devs: dict = None):
        with guard("保存评分结果"):
            self.last_score_result = {
                'overall': overall, 'segs': list(segs),
                'threshold': threshold,
                'ref_path': ref_path, 'user_path': user_path,
                'corrections': corrections or {},
                'joint_devs': joint_devs or {},
            }
            log.debug(f"评分结果已保存: overall={overall:.1f}")

    def set_status(self, msg):
        try:
            self.var_status.set(msg)
        except Exception:
            pass

    def set_summary(self, msg):
        try:
            self.var_summary.set(msg)
        except Exception:
            pass

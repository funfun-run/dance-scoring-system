# gui/panels.py — 功能面板 (评分 / 回顾 / 分割 / 设置 / NPU / 模型 / 性能)

import tkinter as tk
from tkinter import messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import os
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, List, Dict

from dance_scoring.gui.theme import COLORS, FONTS
from dance_scoring.gui.logger import log, guard, safe_after, safe_thread
from dance_scoring.gui.components import VideoImporter, ScoreDisplay, SegmentBar


def _check_llm_available() -> bool:
    """检查 LLMProvider 子类是否可用（仅检查文件存在，不加载模型）。"""
    try:
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        my_qwen = os.path.join(project_root, "LLM", "my_qwen.py")
        return os.path.exists(my_qwen)
    except Exception:
        return False


# ============================================================
# 评分面板
# ============================================================

class ScoringPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._ref_path: Optional[str] = None
        self._user_path: Optional[str] = None
        self._build()

    def _build(self):
        # 双视频导入
        imp = ttk.Frame(self)
        imp.pack(fill=tk.X, pady=(0, 8))
        left = ttk.Frame(imp)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.ref_import = VideoImporter(left, "参考视频", on_select=lambda p: setattr(self, '_ref_path', p))
        self.ref_import.pack(fill=tk.X)
        right = ttk.Frame(imp)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.user_import = VideoImporter(right, "用户视频", on_select=lambda p: setattr(self, '_user_path', p))
        self.user_import.pack(fill=tk.X)

        # 参数
        params = ttk.Frame(self)
        params.pack(fill=tk.X, pady=4)
        ttk.Label(params, text="BPM:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.var_bpm = tk.StringVar(value="120")
        ttk.Entry(params, textvariable=self.var_bpm, width=6).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(params, text="合格线:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.var_threshold = tk.StringVar(value="60")
        ttk.Entry(params, textvariable=self.var_threshold, width=6).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(params, text="算法:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.var_algo = tk.StringVar(value="dtw")
        ttk.Combobox(params, textvariable=self.var_algo, values=["dtw", "fastdtw"],
                    state="readonly", width=8).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(params, text="纠正:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.var_correction = tk.StringVar(value="rule")
        # 检查 LLM 是否可用
        llm_available = _check_llm_available()
        corr_values = ["rule", "llm"] if llm_available else ["rule"]
        corr_state = "readonly" if llm_available else tk.DISABLED
        self.cmb_correction = ttk.Combobox(
            params, textvariable=self.var_correction, values=corr_values,
            state=corr_state, width=6)
        self.cmb_correction.pack(side=tk.LEFT, padx=(2, 0))

        # 开始按钮
        self.btn_start = ttk.Button(self, text="▶  开始评分", command=self._do_score,
                                     bootstyle="warning", style="warning.TButton")
        self.btn_start.pack(fill=tk.X, pady=8)

        # 进度
        self.progress = ttk.Progressbar(self, mode='determinate', length=300)
        self.lbl_progress = ttk.Label(self, text="", font=FONTS["small"],
                                      foreground=COLORS["text_muted"])

        # 结果区
        self.result_frame = ttk.Frame(self)
        self.score_display = ScoreDisplay(self.result_frame, "medium")

        self.seg_list = ttk.Frame(self.result_frame)

    def _do_score(self):
        if getattr(self, '_scoring', False):
            return
        if not self._ref_path:
            messagebox.showwarning("提示", "请先选择参考视频"); return
        if not self._user_path:
            messagebox.showwarning("提示", "请先选择用户视频"); return
        try:
            bpm = int(self.var_bpm.get())
            threshold = float(self.var_threshold.get())
        except ValueError:
            messagebox.showwarning("提示", "BPM/阈值请输入数字"); return

        # ⚠️ 必须在主线程读取 Tkinter 变量
        ref_path = self._ref_path
        user_path = self._user_path
        algo = self.var_algo.get()
        correction_backend = self.var_correction.get()

        self._scoring = True
        self.progress.pack(fill=tk.X, pady=4)
        self.lbl_progress.pack()
        self.result_frame.pack_forget()
        self.btn_start.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.lbl_progress.config(text="加载模型...")

        log.info(f"开始评分: ref={os.path.basename(ref_path)} user={os.path.basename(user_path)} "
                 f"algo={algo} correction={correction_backend}")

        def run():
            with guard("评分流水线"):
                def _progress(pct, msg):
                    try:
                        self.master.after(0, lambda: self.progress.config(value=pct))
                        self.master.after(0, lambda: self.lbl_progress.config(text=msg))
                    except Exception:
                        pass

                _progress(10, "加载模型...")
                from dance_scoring.core.extractor import PoseExtractor, download_model
                from dance_scoring.core.scorer import Scorer
                from dance_scoring.core.config import Config

                with guard("下载模型"):
                    download_model()
                cfg = Config(score_threshold=threshold)

                _progress(15, "提取参考视频姿态...")
                with guard("提取参考姿态"):
                    ref = PoseExtractor(cfg).extract(ref_path)
                log.debug(f"参考姿态: {len(ref)} 帧")

                _progress(45, "提取用户视频姿态...")
                with guard("提取用户姿态"):
                    user = PoseExtractor(cfg).extract(user_path)
                log.debug(f"用户姿态: {len(user)} 帧")

                # 强制 GC 释放 MediaPipe C++ 对象，避免和 optimum-intel 冲突
                import gc; gc.collect()

                _progress(70, "DTW对齐+打分...")
                with guard("创建纠正提供者"):
                    try:
                        from dance_scoring.core.correction_provider import create_correction_provider
                        hub = self._find_hub()
                        model = getattr(hub, 'selected_model', '3b') if hub else '3b'
                        corr = create_correction_provider(correction_backend, model=model)
                        log.info(f"纠正后端: {corr.provider_name}")
                    except Exception as e:
                        log.warning(f"LLM 纠正不可用，回退规则引擎: {e}")
                        corr = None

                scorer = Scorer(cfg, bpm=bpm, alignment_method=algo)
                overall, segs, low, path = scorer.score(
                    ref, user, correction_provider=corr,
                )
                log.info(f"评分完成: overall={overall:.1f}")

                _progress(95, "保存结果...")
                self._scoring = False

                def _done():
                    with guard("显示评分结果"):
                        if not self.winfo_exists():
                            return
                        self.progress.pack_forget()
                        self.lbl_progress.pack_forget()
                        self.btn_start.config(state=tk.NORMAL)
                        self.lbl_progress.config(
                            text=f"✅ 评分完成！总评 {overall:.1f}/100  "
                                 f"{sum(1 for s in segs if s['score']>=threshold)}/"
                                 f"{len(segs)}段合格",
                            font=FONTS["body_bold"])
                        self.lbl_progress.pack()

                        corrections = {}
                        joint_devs = {}
                        for s in segs:
                            if s.get('correction_text'):
                                corrections[s['id']] = s['correction_text']
                            if s.get('deviations'):
                                joint_devs[s['id']] = s['deviations']

                        for c in self.winfo_toplevel().winfo_children():
                            if hasattr(c, 'set_score_result'):
                                c.set_score_result(
                                    overall, segs, threshold,
                                    ref_path=ref_path or "",
                                    user_path=user_path or "",
                                    corrections=corrections,
                                    joint_devs=joint_devs,
                                )
                                c.set_status(f"✅ 评分完成 {overall:.1f}分")
                                break

                self.master.after(0, _done)

        safe_thread("scoring", run)

    def _find_hub(self):
        try:
            root = self.winfo_toplevel()
            for c in root.winfo_children():
                if hasattr(c, 'last_score_result'):
                    return c
        except Exception:
            pass
        return None

    def _go_review(self, segs, threshold):
        pass  # 由 Hub 控制面板切换

    def _reset(self):
        self.result_frame.pack_forget()
        for w in self.seg_list.winfo_children():
            w.destroy()


# ============================================================
# 回顾面板
# ============================================================

class ReviewPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._hub = None
        self._seg_widgets = []
        # 对话状态
        self._chat_history = []       # [(role, text), ...]
        self._chat_llm = None         # 懒加载的 LLMProvider 引用
        self._chat_sending = False    # 防止重复发送
        self._build()

    def _build(self):
        ttk.Label(self, text="📁 练习回顾", font=FONTS["heading"],
                 foreground=COLORS["text"]).pack(anchor=tk.W, pady=(0, 12))

        # 内容区：左列表 + 右详情
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        self.seg_list_frame = ttk.Frame(body, width=200)
        self.seg_list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.seg_list_frame.pack_propagate(False)

        self.detail_frame = ttk.Frame(body)
        self.detail_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 默认空状态
        self._show_empty()

        # 尝试加载已有结果
        self.after(200, self._try_load)

    def _find_hub(self):
        """向上查找 Hub 实例。"""
        try:
            root = self.winfo_toplevel()
            for c in root.winfo_children():
                if hasattr(c, 'last_score_result'):
                    return c
        except Exception:
            pass
        return None

    def _try_load(self, retry=0):
        with guard(f"加载评分结果 (第{retry+1}次)"):
            self._hub = self._find_hub()
            data = None
            if self._hub is not None:
                data = getattr(self._hub, 'last_score_result', None)
            if data and data.get('segs'):
                self._load_result(data)
            elif retry < 5:
                safe_after(self, "重试加载", 500, self._try_load, retry + 1)

    def _load_result(self, data: dict):
        """加载评分结果数据。"""
        with guard("渲染回顾面板"):
            for w in self.seg_list_frame.winfo_children():
                w.destroy()
            for w in self.detail_frame.winfo_children():
                w.destroy()
            if hasattr(self, '_chat_frame') and self._chat_frame is not None:
                try:
                    self._chat_frame.destroy()
                except Exception:
                    pass

        segs = data.get('segs', [])
        threshold = data.get('threshold', 60)
        overall = data.get('overall', 0)

        # 标题
        ttk.Label(self.seg_list_frame, text=f"总评 {overall:.1f}",
                 font=FONTS["heading"], bootstyle="primary").pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(self.seg_list_frame,
                 text=f"{sum(1 for s in segs if s['score']>=threshold)}/{len(segs)}段合格",
                 font=FONTS["body"], foreground=COLORS["text_secondary"]
                 ).pack(anchor=tk.W, pady=(0, 8))

        # 段列表
        for seg in segs:
            passed = seg['score'] >= threshold
            text = f"{'✅' if passed else '❌'} 段{seg['id']}  {seg['score']:.1f}"
            fg = COLORS["success"] if passed else COLORS["danger"]
            lbl = ttk.Label(self.seg_list_frame, text=text, font=FONTS["body"],
                          foreground=fg, cursor="hand2")
            lbl.pack(fill=tk.X, pady=2, anchor=tk.W)
            lbl.bind("<Button-1>",
                    lambda e, s=seg: self._show_seg_detail(s, threshold))
            self._seg_widgets.append(lbl)

        # 默认显示第一段
        if segs:
            self._show_seg_detail(segs[0], threshold)

        # 底部按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_frame, text="📂 练习视频目录",
                  command=lambda: _open_output_dir("output/low_score_clips"),
                  bootstyle="secondary-outline").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📂 分段目录",
                  command=lambda: _open_output_dir("output/segments"),
                  bootstyle="secondary-outline").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🔄 刷新",
                  command=self._try_load,
                  bootstyle="secondary-outline").pack(side=tk.LEFT, padx=2)

        # ── AI 舞蹈教练对话窗口 ──
        self._build_chat(data)

    def _show_seg_detail(self, seg: dict, threshold: float):
        """显示某段的详细信息：得分、时间、纠正建议、练习视频入口。"""
        for w in self.detail_frame.winfo_children():
            w.destroy()

        passed = seg['score'] >= threshold
        color = COLORS["success"] if passed else COLORS["danger"]
        icon = "✅" if passed else "❌"

        # 段标题
        ttk.Label(self.detail_frame,
                 text=f"{icon} 第 {seg['id']} 段",
                 font=FONTS["heading"]).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(self.detail_frame,
                 text=f"得分: {seg['score']:.1f}  |  "
                      f"时间: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s",
                 font=FONTS["body"],
                 foreground=color).pack(anchor=tk.W, pady=(0, 12))

        # 纠正建议
        ttk.Label(self.detail_frame, text="── 纠正建议 ──",
                 font=FONTS["body_bold"]).pack(anchor=tk.W, pady=(0, 6))

        corr_text = seg.get('correction_text', '')
        if not corr_text:
            # 从 Hub 数据生成纠正建议
            hub_data = self._hub.last_score_result if self._hub else {}
            corrections = hub_data.get('corrections', {})
            corr_text = corrections.get(seg['id'], '')

        if corr_text:
            if corr_text.startswith(f"第{seg['id']}段："):
                corr_text = corr_text[len(f"第{seg['id']}段："):]
            for line in corr_text.split('；'):
                line = line.strip()
                if line:
                    ttk.Label(self.detail_frame, text=f"⚡ {line}",
                             font=FONTS["body"], wraplength=400,
                             foreground=COLORS["warning"]).pack(anchor=tk.W, pady=2)
        else:
            ttk.Label(self.detail_frame, text="该段达标，无特别建议 ✨",
                     font=FONTS["body"],
                     foreground=COLORS["text_secondary"]).pack(anchor=tk.W)

        # 练习视频
        ttk.Separator(self.detail_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        ttk.Label(self.detail_frame, text="── 练习视频 ──",
                 font=FONTS["body_bold"]).pack(anchor=tk.W, pady=(0, 6))

        clip_name = f"practice_seg{seg['id']:02d}_score{seg['score']:.0f}_slow.mp4"
        clip_path = os.path.join(os.getcwd(), "output", "low_score_clips", clip_name)
        seg_path = os.path.join(os.getcwd(), "output", "segments",
                               f"ref_seg_{seg['id']:02d}_slow.mp4")

        found_paths = []
        if os.path.exists(clip_path):
            found_paths.append(("低分练习视频 (慢动作 0.8x)", clip_path))
        if os.path.exists(seg_path):
            found_paths.append(("分段慢动作视频 (0.8x)", seg_path))

        if found_paths:
            for label, p in found_paths:
                fsize = os.path.getsize(p) / 1024
                frm = ttk.Frame(self.detail_frame)
                frm.pack(fill=tk.X, anchor=tk.W, pady=2)
                ttk.Label(frm, text=f"🎬 {label}",
                         font=FONTS["body"]).pack(side=tk.LEFT)
                ttk.Label(frm, text=f"({fsize:.0f}KB)",
                         font=FONTS["small"],
                         foreground=COLORS["text_muted"]).pack(side=tk.LEFT, padx=4)
                ttk.Button(frm, text="▶",
                          command=lambda p=p: _play_video(p),
                          bootstyle="success-outline", width=3
                          ).pack(side=tk.LEFT, padx=4)
        else:
            ttk.Label(self.detail_frame,
                     text="💡 暂无练习视频\n不合格段的视频会在评分时自动生成",
                     font=FONTS["small"],
                     foreground=COLORS["text_muted"]).pack(anchor=tk.W)

        ttk.Button(self.detail_frame, text="📂 打开目录",
                  command=lambda: _open_output_dir("output/low_score_clips"),
                  bootstyle="secondary-outline").pack(anchor=tk.W, pady=(4, 0))

    # ── AI 舞蹈教练对话 ──

    def _build_chat(self, data: dict):
        """在回顾面板底部构建 AI 对话窗口。"""
        # 清除旧对话历史和子进程
        self._chat_history = []
        if hasattr(self, '_chat_proc') and self._chat_proc and self._chat_proc.poll() is None:
            try:
                self._chat_proc.stdin.close()
                self._chat_proc.wait(timeout=3)
            except Exception:
                self._chat_proc.kill()
        self._chat_proc = None

        # 所有聊天 widget 放入独立 frame，便于清理
        self._chat_frame = ttk.Frame(self)
        self._chat_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # 分隔线
        ttk.Separator(self._chat_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 8))

        # 标题
        chat_header = ttk.Frame(self._chat_frame)
        chat_header.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(chat_header, text="🤖 AI 舞蹈教练",
                 font=FONTS["body_bold"],
                 foreground=COLORS["accent"]).pack(side=tk.LEFT)
        ttk.Label(chat_header, text="向 AI 提问，获取个性化改进建议",
                 font=FONTS["small"],
                 foreground=COLORS["text_muted"]).pack(side=tk.LEFT, padx=8)

        # 对话显示区
        chat_display_frame = ttk.Frame(self._chat_frame)
        chat_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self._chat_text = tk.Text(
            chat_display_frame,
            height=10,
            bg=COLORS["input_bg"],
            fg=COLORS["text"],
            font=FONTS["body"],
            relief=tk.FLAT,
            borderwidth=6,
            wrap=tk.WORD,
            state=tk.DISABLED,
            insertbackground=COLORS["text"],
        )
        chat_scroll = ttk.Scrollbar(
            chat_display_frame, orient=tk.VERTICAL,
            command=self._chat_text.yview,
        )
        self._chat_text.configure(yscrollcommand=chat_scroll.set)
        self._chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chat_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 配置对话文本样式
        self._chat_text.tag_configure(
            "ai", foreground=COLORS["accent"], font=FONTS["body"],
        )
        self._chat_text.tag_configure(
            "user", foreground=COLORS["success"], font=FONTS["body"],
        )
        self._chat_text.tag_configure(
            "system", foreground=COLORS["text_muted"], font=FONTS["small"],
        )

        # 输入区
        input_frame = ttk.Frame(self._chat_frame)
        input_frame.pack(fill=tk.X)

        self._chat_input = ttk.Entry(input_frame, font=FONTS["body"])
        self._chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._chat_input.bind("<Return>", self._on_chat_key)
        self._chat_input.focus_set()

        self._chat_send_btn = ttk.Button(
            input_frame, text="发送",
            command=self._send_chat,
            bootstyle="warning",
        )
        self._chat_send_btn.pack(side=tk.RIGHT)

        # 存储评分数据供对话上下文使用
        self._chat_data = data

        # 显示欢迎消息
        overall = data.get('overall', 0)
        segs = data.get('segs', [])
        fail_count = sum(1 for s in segs if s.get('score', 0) < 60)
        self._append_chat("system",
            f"当前评分结果：总分 {overall:.1f}，{len(segs)} 段中 {fail_count} 段不合格。"
            f"你可以问 AI 关于某个动作的改进方法，或如何针对薄弱部位练习。"
        )

        # 延迟加载：AI 模型在用户首次提问时才加载
        self._chat_llm = False  # False = 未加载, True = 已加载
        self._append_chat("system", "输入问题后 AI 将自动加载模型（首次需约 8 秒）")

    def _on_chat_key(self, event):
        """Enter 键发送消息。"""
        self._send_chat()
        return "break"

    def _send_chat(self):
        """发送用户消息并获取 AI 回复。"""
        if self._chat_sending:
            return
        user_text = self._chat_input.get().strip()
        if not user_text:
            return

        self._chat_sending = True
        self._chat_input.delete(0, tk.END)
        self._chat_send_btn.config(state=tk.DISABLED)
        self._append_chat("user", user_text)
        log.debug(f"对话消息: {user_text[:50]}...")

        is_first = (self._chat_llm is False)
        if is_first:
            self._append_chat("system", "⏳ 首次加载模型约需8秒...")

        chat_data = self._chat_data
        chat_history = [(r, t) for r, t in self._chat_history[-6:]]
        chat_history.append(("user", user_text))
        system_ctx = self._build_chat_system_context(chat_data)

        def _call_ai():
            with guard("AI对话"):
                import subprocess as _sp, tempfile as _tf, json as _json, sys as _sys
                reply = ""
                try:
                    data = _json.dumps({'s': system_ctx, 'h': chat_history[:-1], 'u': user_text},
                                      ensure_ascii=False)
                    with _tf.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                        f.write(data); inp = f.name
                    script = (
                        'import sys,json\nsys.path.insert(0,"src")\n'
                        f'with open({repr(inp)}) as f: d=json.load(f)\n'
                        'from LLM.my_qwen import Qwen3BProvider\n'
                        'llm=Qwen3BProvider()\n'
                        'print(llm.chat(d["s"],d.get("h",[]),d["u"]))\n'
                    )
                    r = _sp.run([_sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=120)
                    import os; os.unlink(inp)
                    # 取最后一行非空内容作为回复（过滤加载日志）
                    lines = [l for l in r.stdout.strip().split('\n') if l.strip()
                            and not l.startswith('llama_') and '加载' not in l and '✅' not in l]
                    reply = lines[-1].strip() if lines else ''
                except Exception:
                    pass
                if not reply:
                    reply = "抱歉，请再说一次。"
                for prefix in ["教练：", "教练:", "好的，", "好的 "]:
                    if reply.startswith(prefix):
                        reply = reply[len(prefix):].strip()
                self.master.after(0, lambda r=reply: self._append_chat("ai", r))
            def _on_done():
                self._chat_sending = False
                self._chat_llm = True
                try:
                    if hasattr(self, '_chat_send_btn') and self._chat_send_btn.winfo_exists():
                        self._chat_send_btn.config(state=tk.NORMAL)
                except Exception:
                    pass
            self.master.after(0, _on_done)

        safe_thread("chat_call", _call_ai)

    def _build_chat_system_context(self, data: dict) -> str:
        """构建对话的系统上下文（含薄弱部位记忆）。"""
        segs = data.get('segs', [])
        overall = data.get('overall', 0)
        fail_segs = [s for s in segs if s.get('score', 0) < 60]

        # 收集薄弱部位和各段得分
        weak_parts = {}
        for s in fail_segs:
            for d in (s.get('deviations', []) or [])[:5]:
                name = d.joint_name
                if name not in weak_parts or abs(d.deviation_deg) > abs(weak_parts[name]):
                    weak_parts[name] = d.deviation_deg
        # 按偏差排序
        weak_sorted = sorted(weak_parts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        weak_list = "、".join(f"{n}" for n, _ in weak_sorted) if weak_sorted else "无明显薄弱"

        seg_summary = " ".join(
            f"段{s['id']}{s['score']:.0f}分" for s in segs
        )

        return (
            f"你是真人舞蹈教练，正在和学员一对一聊天指导。\n"
            f"学员测评：总分{overall:.0f}，{len(fail_segs)}段需改进。{seg_summary}。\n"
            f"最需关注：{weak_list}。\n"
            f"规则：用口语化中文，亲切自然像朋友聊天。每次只给1条建议。\n"
            f"如果学员问具体部位，针对那个部位回答。不要列举数值。不超过50字。"
        )

    def _append_chat(self, role: str, text: str):
        """向对话显示区追加一条消息。"""
        try:
            self._chat_text.config(state=tk.NORMAL)
            if role == "user":
                tag, prefix = "user", "🧑 "
            elif role == "ai":
                tag, prefix = "ai", "🤖 "
            else:
                tag, prefix = "system", ""
            self._chat_text.insert(tk.END, f"{prefix}{text}\n\n", tag)
            self._chat_text.see(tk.END)
            self._chat_text.config(state=tk.DISABLED)
            if role in ("user", "ai"):
                self._chat_history.append((role, text))
        except Exception as e:
            log.warning(f"追加对话失败: {e}")

    def _show_empty(self):
        ttk.Label(self.detail_frame, text="📁 练习回顾", font=FONTS["heading"],
                 foreground=COLORS["text"]).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(self.detail_frame,
                 text="请先完成一次评分\n\n"
                      "评分后此处会显示：\n"
                      "• 各段得分与纠正建议\n"
                      "• 低分片段慢动作练习视频\n"
                      "• 薄弱部位分析\n\n"
                      "点击下方的「🔄 刷新」加载已有结果",
                 font=FONTS["body"], foreground=COLORS["text_muted"],
                 justify=tk.LEFT, wraplength=400).pack(pady=20)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="📂 练习视频目录",
                  command=lambda: _open_output_dir("output/low_score_clips"),
                  bootstyle="secondary-outline").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="📂 分段目录",
                  command=lambda: _open_output_dir("output/segments"),
                  bootstyle="secondary-outline").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="🔄 加载评分结果",
                  command=self._try_load,
                  bootstyle="secondary-outline").pack(side=tk.LEFT, padx=2)


# ============================================================
# 分割面板
# ============================================================

class SplitPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._build()

    def _build(self):
        self.import_video = VideoImporter(self, "参考视频")
        self.import_video.pack(fill=tk.X, pady=(0, 8))

        params = ttk.Frame(self)
        params.pack(fill=tk.X, pady=4)
        ttk.Label(params, text="BPM:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.var_bpm = tk.StringVar(value="120")
        ttk.Entry(params, textvariable=self.var_bpm, width=6).pack(side=tk.LEFT, padx=(2, 0))

        self.btn_split = ttk.Button(self, text="✂️  开始分割", command=self._do_split,
                                     bootstyle="warning")
        self.btn_split.pack(fill=tk.X, pady=8)

        self.result_frame = ttk.Frame(self)
        self.lbl_result = ttk.Label(self.result_frame, text="", font=FONTS["body"],
                                     foreground=COLORS["text_secondary"])

    def _do_split(self):
        if getattr(self, '_splitting', False):
            return
        path = self.import_video.get_file()
        if not path:
            messagebox.showwarning("提示", "请先选择参考视频"); return
        try:
            bpm = int(self.var_bpm.get())
        except ValueError:
            messagebox.showwarning("提示", "BPM 请输入整数"); return

        self._splitting = True
        self.btn_split.config(state=tk.DISABLED)
        self.result_frame.pack(fill=tk.X, pady=8)
        self.lbl_result.pack()
        self.lbl_result.config(text="分割中...")

        from dance_scoring.gui.worker import SplitWorker
        log.info(f"开始分割: {path} BPM={bpm}")

        def _on_progress(pct, msg):
            try:
                self.lbl_result.config(text=msg)
            except Exception:
                pass

        def _on_done(success, result, error):
            self._splitting = False
            try:
                self.btn_split.config(state=tk.NORMAL)
            except Exception:
                pass
            if success:
                n = len(result.get('segments', []))
                self.lbl_result.config(
                    text=f"✅ 分割完成: {n}段\n方式: {result.get('method','')}\n"
                         f"BPM: {result.get('bpm',0):.0f}\n输出: output/segments/")
                ttk.Button(
                    self.result_frame, text="📂 打开分段目录",
                    command=lambda: _open_output_dir("output/segments"),
                    bootstyle="secondary-outline").pack(pady=4)
                log.info(f"分割完成: {n}段, {result.get('method','')}")
            else:
                log.error(f"分割失败: {error}")
                messagebox.showerror("分割失败", str(error)[:300])

        worker = SplitWorker(
            path, bpm, "output/segments",
            on_progress=_on_progress, on_done=_on_done,
        )
        worker.start()


# ============================================================
# 设置面板
# ============================================================

class SettingsPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._build()

    def _build(self):
        ttk.Label(self, text="⚙️ 系统设置", font=FONTS["heading"],
                 foreground=COLORS["text"]).pack(anchor=tk.W, pady=(0, 12))

        rows = [
            ("推理后端", ["auto", "openvino", "mediapipe"], "auto"),
            ("摄像头", ["/dev/video0", "/dev/video1"], "/dev/video0"),
            ("对齐算法", ["dtw", "fastdtw"], "dtw"),
            ("BPM 默认", None, "120"),
            ("合格线", None, "60"),
            ("采集窗口(帧)", None, "150"),
        ]
        self.vars = {}

        for label, choices, default in rows:
            f = ttk.Frame(self)
            f.pack(fill=tk.X, pady=3)
            ttk.Label(f, text=label, font=FONTS["body"], width=16,
                     anchor=tk.W).pack(side=tk.LEFT)
            if choices:
                var = tk.StringVar(value=default)
                ttk.Combobox(f, textvariable=var, values=choices,
                            state="readonly", width=14).pack(side=tk.RIGHT)
                self.vars[label] = var
            else:
                var = tk.StringVar(value=default)
                ttk.Entry(f, textvariable=var, width=12).pack(side=tk.RIGHT)
                self.vars[label] = var


# ============================================================
# NPU 面板
# ============================================================

class NPUPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._build()

    def _build(self):
        ttk.Label(self, text="🖥️ NPU 状态", font=FONTS["heading"],
                 foreground=COLORS["text"]).pack(anchor=tk.W, pady=(0, 12))

        self.info_frame = ttk.Frame(self)
        self.info_frame.pack(fill=tk.BOTH, expand=True)
        self._refresh()

    def _refresh(self):
        for w in self.info_frame.winfo_children():
            w.destroy()

        try:
            from dance_scoring.platform.npu import NPUManager
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            device = NPUManager.best_device()
            cpu_info = core.get_property("CPU", "FULL_DEVICE_NAME")

            info = [
                ("设备", device),
                ("CPU", cpu_info[:60]),
                ("可用设备", ", ".join(devices)),
                ("OpenVINO", ov.__version__),
            ]

            avail = NPUManager.available()
            status = "✅ NPU 可用" if avail else "⚠️ NPU 不可用 (CPU 回退)"
            status_color = COLORS["success"] if avail else COLORS["warning"]
            ttk.Label(self.info_frame, text=status, font=FONTS["body_bold"],
                     foreground=status_color).pack(anchor=tk.W, pady=(0, 6))

            for k, v in info:
                f = ttk.Frame(self.info_frame)
                f.pack(fill=tk.X, pady=2)
                ttk.Label(f, text=k, font=FONTS["body"], foreground=COLORS["text_muted"],
                         width=12, anchor=tk.W).pack(side=tk.LEFT)
                ttk.Label(f, text=v, font=FONTS["body"]).pack(side=tk.LEFT)

            ttk.Button(self.info_frame, text="🔄 刷新", command=self._refresh,
                      bootstyle="secondary-outline").pack(pady=(8, 0))

        except Exception as e:
            ttk.Label(self.info_frame, text=f"错误: {e}", font=FONTS["body"],
                     foreground=COLORS["danger"]).pack()


# ============================================================
# 模型面板
# ============================================================

class ModelPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._build()

    def _build(self):
        ttk.Label(self, text="🔧 模型管理", font=FONTS["heading"],
                 foreground=COLORS["text"]).pack(anchor=tk.W, pady=(0, 12))

        self._show_status()
        self._show_actions()

    def _show_status(self):
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, pady=4)

        ir_dir = Path("src/dance_scoring/models")
        xml = ir_dir / "pose_landmarker.xml"
        meta = ir_dir / "pose_landmarker_meta.json"

        if xml.exists():
            ttk.Label(status_frame, text="✅ IR 模型已转换", font=FONTS["body_bold"],
                     foreground=COLORS["success"]).pack(anchor=tk.W)

            if meta.exists():
                with open(meta) as f:
                    m = json.load(f)
                ttk.Label(status_frame,
                         text=f"精度: {m.get('precision','?')} │ "
                              f"原始: {m.get('tflite_size_bytes',0)/1024:.0f}KB │ "
                              f"IR: {m.get('ir_size_bytes',0)/1024:.0f}KB │ "
                              f"压缩: {(1-m.get('compression_ratio',1))*100:.1f}%",
                         font=FONTS["small"],
                         foreground=COLORS["text_secondary"]).pack(anchor=tk.W, pady=(2, 0))
        else:
            ttk.Label(status_frame, text="⚠️ 未转换 IR 模型",
                     font=FONTS["body_bold"],
                     foreground=COLORS["warning"]).pack(anchor=tk.W)

    def _show_actions(self):
        actions = ttk.Frame(self)
        actions.pack(fill=tk.X, pady=12)

        ttk.Label(actions, text="精度:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.var_precision = tk.StringVar(value="FP16")
        ttk.Combobox(actions, textvariable=self.var_precision,
                    values=["FP16", "FP32", "INT8"],
                    state="readonly", width=6).pack(side=tk.LEFT, padx=4)

        ttk.Button(actions, text="🔄 重新转换", command=self._convert,
                  bootstyle="secondary-outline").pack(side=tk.LEFT, padx=4)

        self.lbl_conv_status = ttk.Label(self, text="", font=FONTS["small"])
        self.lbl_conv_status.pack(anchor=tk.W)

    def _convert(self):
        if getattr(self, '_converting', False):
            return
        precision = self.var_precision.get()
        self._converting = True
        self.lbl_conv_status.config(text=f"转换中 ({precision})...", foreground=COLORS["warning"])

        def run():
            import subprocess, sys
            try:
                r = subprocess.run(
                    [sys.executable, "scripts/convert_model.py",
                     "--precision", precision],
                    capture_output=True, text=True, timeout=120)
                if r.returncode == 0:
                    self._converting = False
                    self.master.after(0, lambda: self.lbl_conv_status.config(
                        text="✅ 转换完成", foreground=COLORS["success"]))
                    self.master.after(0, self._show_status)
                else:
                    self._converting = False
                    self.master.after(0, lambda: self.lbl_conv_status.config(
                        text=f"❌ 转换失败: {r.stderr[-120:]}",
                        foreground=COLORS["danger"]))
            except Exception as e:
                self._converting = False
                self.master.after(0, lambda: self.lbl_conv_status.config(
                    text=f"❌ {e}", foreground=COLORS["danger"]))

        threading.Thread(target=run, daemon=True).start()


# ============================================================
# 性能面板
# ============================================================

class PerfPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._build()

    def _build(self):
        ttk.Label(self, text="📈 性能测试", font=FONTS["heading"],
                 foreground=COLORS["text"]).pack(anchor=tk.W, pady=(0, 12))

        # 测试视频选择
        f1 = ttk.Frame(self)
        f1.pack(fill=tk.X, pady=4)
        ttk.Label(f1, text="测试视频:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.lbl_video = ttk.Label(f1, text="(使用随机帧)", font=FONTS["small"],
                                   foreground=COLORS["text_muted"])
        self.lbl_video.pack(side=tk.LEFT, padx=4)
        ttk.Button(f1, text="选择", command=self._sel_video,
                  bootstyle="secondary-outline").pack(side=tk.RIGHT)

        # 参数
        f2 = ttk.Frame(self)
        f2.pack(fill=tk.X, pady=4)
        ttk.Label(f2, text="帧数:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.var_frames = tk.StringVar(value="100")
        ttk.Entry(f2, textvariable=self.var_frames, width=6).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(f2, text="预热:", font=FONTS["body"]).pack(side=tk.LEFT)
        self.var_warmup = tk.StringVar(value="2")
        ttk.Entry(f2, textvariable=self.var_warmup, width=4).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Button(self, text="▶  开始测试", command=self._run_test,
                  bootstyle="warning").pack(fill=tk.X, pady=8)

        self.result_text = tk.Text(self, height=10, bg=COLORS["input_bg"],
                                   fg=COLORS["text"], font=FONTS["mono"],
                                   relief=tk.FLAT, borderwidth=4,
                                   insertbackground=COLORS["text"])
        self.result_text.pack(fill=tk.BOTH, expand=True)

        self._video_path: Optional[str] = None

    def _sel_video(self):
        path = filedialog.askopenfilename(
            title="选择测试视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        if path:
            self._video_path = path
            self.lbl_video.config(text=os.path.basename(path)[:30])

    def _run_test(self):
        if getattr(self, '_testing', False):
            return
        try:
            frames = int(self.var_frames.get())
            warmup = int(self.var_warmup.get())
        except ValueError:
            messagebox.showwarning("提示", "帧数/预热请输入整数"); return

        self._testing = True
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "测试中...\n")

        def run():
            import subprocess, sys
            cmd = [sys.executable, "scripts/benchmark.py",
                   "--frames", str(frames), "--rounds", str(warmup)]
            if self._video_path:
                cmd.append(self._video_path)
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                output = r.stdout if r.returncode == 0 else r.stderr
                self.master.after(0, lambda: self.result_text.delete("1.0", tk.END))
                self.master.after(0, lambda: self.result_text.insert(tk.END, output[-3000:]))
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda: self.result_text.insert(tk.END, f"错误: {err_msg}"))
            finally:
                self._testing = False

        threading.Thread(target=run, daemon=True).start()


# ============================================================
# 工具
# ============================================================

def _open_output_dir(path: str):
    """打开输出目录。"""
    try:
        full = os.path.join(os.getcwd(), path)
        os.makedirs(full, exist_ok=True)
        if os.name == 'nt':
            os.startfile(full)
        else:
            subprocess.Popen(['xdg-open', full])
    except Exception as e:
        log.warning(f"打开目录失败 [{path}]: {e}")
        try:
            messagebox.showwarning("提示", f"无法打开目录: {e}")
        except Exception:
            pass


def _play_video(path: str):
    """用系统播放器打开视频文件。"""
    try:
        if not os.path.exists(path):
            messagebox.showinfo("提示", f"文件不存在:\n{os.path.basename(path)}")
            return
        for player in ['vlc', 'mpv', 'ffplay', 'totem', 'xdg-open']:
            try:
                r = subprocess.run(['which', player], capture_output=True, text=True)
                if r.returncode == 0:
                    subprocess.Popen([player, path],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
                    return
            except Exception:
                continue
        subprocess.Popen(['xdg-open', os.path.dirname(path)])
    except Exception as e:
        log.warning(f"播放视频失败 [{path}]: {e}")
        try:
            messagebox.showwarning("提示", f"无法播放视频: {e}")
        except Exception:
            pass

# gui/app.py — 舞蹈评分系统 GUI 入口 (HUD 运动风 v3)
#
# 启动: python3 src/dance_scoring/gui/app.py

import tkinter as tk
import ttkbootstrap as ttk
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 日志系统 — 最早初始化
from dance_scoring.gui.logger import log, guard
log.info("=" * 50)
log.info("GUI 启动")
log.info("=" * 50)

from dance_scoring.gui.hub import Hub


def _setup_tk_exception_handler(root):
    """全局 Tkinter 异常捕获 — 防止未处理异常导致 GUI 闪退。"""
    # 保存原始处理器
    original_handler = root.report_callback_exception

    def _handler(exc_type, exc_val, exc_tb):
        import traceback
        log.error(
            f"Tkinter 未捕获异常:\n"
            f"{''.join(traceback.format_exception(exc_type, exc_val, exc_tb))}"
        )
        # 也调用原始处理器（打印到 stderr）
        if original_handler:
            original_handler(exc_type, exc_val, exc_tb)

    root.report_callback_exception = _handler


def main():
    with guard("GUI 启动"):
        # ttkbootstrap 默认暗色主题
        root = ttk.Window(
            title="🕺 舞蹈评分系统",
            size=(1100, 680),
        )
        root.minsize(960, 600)

        # 全局异常捕获
        _setup_tk_exception_handler(root)

        # 主 Hub
        hub = Hub(root)
        hub.pack(fill=tk.BOTH, expand=True)

        # 居中
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

        log.info("GUI 主窗口就绪，进入事件循环")
        root.mainloop()
        log.info("GUI 正常退出")


if __name__ == "__main__":
    with guard("main"):
        main()

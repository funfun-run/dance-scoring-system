# gui/app.py — 舞蹈评分系统 GUI 入口 (HUD 运动风 v3)
#
# 启动: python3 src/dance_scoring/gui/app.py

import tkinter as tk
import ttkbootstrap as ttk
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dance_scoring.gui.hub import Hub


def main():
    # ttkbootstrap 默认暗色主题 (不指定 themename 避免与自定义样式冲突)
    root = ttk.Window(
        title="🕺 舞蹈评分系统",
        size=(1100, 680),
    )
    root.minsize(960, 600)

    hub = Hub(root)
    hub.pack(fill=tk.BOTH, expand=True)

    # 居中
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    root.mainloop()


if __name__ == "__main__":
    main()

# gui/theme.py — HUD 运动风色板 + 字体 (ui-ux-pro-max: Fitness/Gym + OLED Dark)
# Space Grotesk is the target font; falls back to system sans-serif on Linux.

# ============================================================
# 色板
# ============================================================

COLORS = {
    "bg":               "#0F172A",   # 主背景
    "bg_light":         "#1A2332",   # 卡片 hover 背景
    "card":             "#1E293B",   # 卡片 / 面板
    "card_border":      "rgba(255, 255, 255, 0.06)",  # 卡片边框
    "accent":           "#F97316",   # 橙色 — 主要操作、跟练卡片边框
    "accent_glow":      "rgba(249, 115, 22, 0.25)",   # 橙色发光
    "success":          "#22C55E",   # 通过绿
    "warning":          "#F5A623",   # 纠正 / 警告黄
    "danger":           "#EF4444",   # 失败红
    "text":             "#F8FAFC",   # 主文字 (近白)
    "text_secondary":   "#94A3B8",   # 次要文字 (冷灰)
    "text_muted":       "#64748B",   # 禁用 / 占位
    "border":           "#334155",   # 分割线
    "input_bg":         "#0C1522",   # 输入框背景
    "bg_input":         "#1E293B",   # canvas 边框 / 旧名兼容
    "npu_on":           "#22C55E",   # NPU 可用绿点
    "npu_off":          "#64748B",   # NPU 不可用灰点
}

# ============================================================
# 字体 (目标: Space Grotesk, 回退: 系统 sans-serif)
# ============================================================

# tkinter 字体元组 (family, size, weight_style)
# Space Grotesk 可用时: ("Space Grotesk", 72, "bold")
# 回退: ("DejaVu Sans", ...)  — Ubuntu 预装

_FONT = "DejaVu Sans"  # Ubuntu 预装，模块加载时无需 tk root

def _detect_font():
    """Tk root 创建后调用，检测最佳可用字体。"""
    global _FONT
    try:
        import tkinter.font as tkfont
        import tkinter as tk
        root = tk._get_default_root()
        if root is None:
            root = tk.Tk()
            root.withdraw()
        available = set(tkfont.families())
        for candidate in ["Space Grotesk", "DejaVu Sans", "Ubuntu", "Helvetica"]:
            if candidate in available:
                _FONT = candidate
                return
    except Exception:
        pass

FONTS = {
    "score":        (_FONT, 72, "bold"),     # 跟练实时得分 (常用别名)
    "score_large":  (_FONT, 72, "bold"),
    "score_medium": (_FONT, 48, "bold"),     # 结果得分
    "heading":      (_FONT, 18, "bold"),     # 卡片标题
    "body":         (_FONT, 11, "normal"),   # 正文
    "body_bold":    (_FONT, 11, "bold"),     # 强调正文
    "mono":         ("Courier", 10, "normal"), # 文件信息
    "small":        (_FONT, 9, "normal"),    # 辅助信息
    "title":        (_FONT, 24, "bold"),     # Hub 主标题
    "icon":         (_FONT, 48, "normal"),   # 卡片图标
}

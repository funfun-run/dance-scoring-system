# gui/logger.py — GUI 全局日志系统
#
# 自动记录到文件 + 控制台输出。
# 用法:
#   from dance_scoring.gui.logger import log, guard, crash_guard
#   log.info("message")
#   with guard("操作名称"): ...
#   @crash_guard("函数名") 装饰回调函数

import logging
import sys
import traceback
import functools
import os
from pathlib import Path
from datetime import datetime

# 日志目录
LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"gui_{datetime.now().strftime('%Y%m%d')}.log"

# —— 配置 ——
_logger = logging.getLogger("dance_gui")
_logger.setLevel(logging.DEBUG)

# 文件 handler — 详细日志，每次写入立即 flush（防止段错误丢失）
class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_fh = _FlushFileHandler(str(LOG_FILE), encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%H:%M:%S"
))
_logger.addHandler(_fh)

# 控制台 handler — INFO+ 实时可见
_ch = logging.StreamHandler(sys.stderr)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter(
    "[%(levelname)s] %(message)s"
))
_logger.addHandler(_ch)

log = _logger


# ============================================================
# 工具函数
# ============================================================

def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


class guard:
    """
    上下文管理器 — 包裹操作，异常自动记录日志。

    用法:
        with guard("加载模型"):
            model = load_model("3b")
    """
    def __init__(self, operation: str, reraise: bool = False):
        self._op = operation
        self._reraise = reraise

    def __enter__(self):
        log.debug(f"开始: {self._op}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            log.debug(f"完成: {self._op}")
        else:
            log.error(f"失败 [{self._op}]: {_format_exc(exc_val)}")
        return not self._reraise


def crash_guard(op_name: str, fallback=None):
    """
    装饰器 — 包裹回调函数，即使内部崩溃也不影响主循环。
    fallback: 异常时的返回值。

    用法:
        @crash_guard("开始评分")
        def _do_score(self): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(f"崩溃 [{op_name}]: {_format_exc(e)}")
                return fallback
        return wrapper
    return decorator


def safe_call(op_name: str, func, *args, **kwargs):
    """
    安全调用任意函数，异常自动记录。

    返回:
        成功时返回函数结果，失败时返回 None。
    """
    try:
        log.debug(f"调用: {op_name}")
        result = func(*args, **kwargs)
        log.debug(f"完成: {op_name}")
        return result
    except Exception as e:
        log.error(f"异常 [{op_name}]: {_format_exc(e)}")
        return None


def safe_after(widget, op_name: str, delay_ms: int, callback, *args):
    """
    Tkinter after() 的安全包装 — callback 异常不会导致主循环崩溃。
    """
    def _wrapped():
        try:
            callback(*args)
        except Exception as e:
            log.error(f"崩溃 [after:{op_name}]: {_format_exc(e)}")

    return widget.after(delay_ms, _wrapped)


def safe_thread(op_name: str, target, daemon: bool = True, **kwargs):
    """
    启动安全后台线程 — target 异常自动记录。
    """
    import threading

    def _wrapped():
        try:
            target(**kwargs)
        except Exception as e:
            log.error(f"崩溃 [thread:{op_name}]: {_format_exc(e)}")

    t = threading.Thread(target=_wrapped, daemon=daemon)
    t.start()
    return t

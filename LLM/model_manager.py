# LLM/model_manager.py — 全局模型生命周期管理器
#
# 职责：
#   1. 保证同一时间只有一个模型驻留内存
#   2. 切换模型时自动卸载旧模型
#   3. 加载/卸载/推理全部加锁，防止并发冲突
#   4. 所有 LLM 调用带超时和异常兜底

import threading
import time
from pathlib import Path

# 全局锁 — 加载、卸载、推理全部走这里
_gate = threading.Lock()

# 当前活跃的模型引用
_current_llm = None
_current_model_key = None  # "1.5b" | "3b"

# 模型加载配置
MODELS = {
    "1.5b": {
        "name": "Qwen2.5-1.5B (OpenVINO)",
        "cls_path": "LLM.my_qwen.Qwen15BProvider",
        "memory_mb": 1500,
    },
    "3b": {
        "name": "Qwen2.5-3B (llama.cpp)",
        "cls_path": "LLM.my_qwen.Qwen3BProvider",
        "memory_mb": 2500,
    },
}

# 超时配置（秒）
INFERENCE_TIMEOUT = 30      # 单次推理超时
CHAT_TIMEOUT = 60           # 对话推理超时（3B 可能较慢）


def _import_class(cls_path: str):
    """懒加载 Provider 类。"""
    import importlib
    mod_path, cls_name = cls_path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def get_current_model_key():
    """返回当前加载的模型 key，未加载返回 None。"""
    return _current_model_key


def load_model(model_key: str, force_reload: bool = False):
    """
    加载指定模型。自动卸载旧模型。线程安全。

    返回:
        LLMProvider 实例

    异常:
        RuntimeError — 模型不可用
    """
    global _current_llm, _current_model_key

    if model_key not in MODELS:
        raise ValueError(f"未知模型: {model_key}，可选值: {list(MODELS.keys())}")

    # 如果已经是目标模型且不强制重载，直接返回
    if _current_model_key == model_key and not force_reload and _current_llm is not None:
        return _current_llm

    with _gate:
        # 双重检查
        if _current_model_key == model_key and not force_reload and _current_llm is not None:
            return _current_llm

        # 卸载旧模型
        _unload_locked()

        # 加载新模型
        cfg = MODELS[model_key]
        print(f"  [ModelManager] 加载 {cfg['name']} (约 {cfg['memory_mb']}MB) ...")

        try:
            cls = _import_class(cfg["cls_path"])
            llm = cls()
            _current_llm = llm
            _current_model_key = model_key
            print(f"  [ModelManager] ✅ {cfg['name']} 就绪")
            return llm
        except Exception as e:
            _current_llm = None
            _current_model_key = None
            raise RuntimeError(f"加载 {cfg['name']} 失败: {e}") from e


def unload_model():
    """手动卸载当前模型，释放内存。"""
    global _current_llm, _current_model_key
    with _gate:
        _unload_locked()


def _unload_locked():
    """内部：卸载当前模型（调用方必须持有 _gate）。"""
    global _current_llm, _current_model_key

    if _current_llm is None:
        return

    key = _current_model_key
    print(f"  [ModelManager] 卸载 {MODELS.get(key, {}).get('name', key)} ...")

    # 调用 Provider 的 unload（如果存在）
    try:
        if hasattr(_current_llm, '_pipe'):
            del _current_llm._pipe
        if hasattr(_current_llm, '_model'):
            del _current_llm._model
        if hasattr(_current_llm, '_tok'):
            del _current_llm._tok
        if hasattr(_current_llm, '_llm'):
            del _current_llm._llm
    except Exception:
        pass

    _current_llm = None
    _current_model_key = None

    # 重置 Provider 单例状态
    try:
        from LLM.my_qwen import _ov_instance, _llama_instance
        import LLM.my_qwen as mq
        mq._ov_instance = None
        mq._llama_instance = None
    except Exception:
        pass

    # 给 GC 一点时间
    import gc
    gc.collect()
    print(f"  [ModelManager] ✅ 已卸载，内存释放")


def infer_safe(prompt: str, max_wait: float = INFERENCE_TIMEOUT) -> str:
    """
    安全推理 — 带超时、异常兜底。

    返回:
        模型输出文本。失败时返回空字符串。
    """
    if _current_llm is None:
        return ""

    result = [""]
    error = [None]

    def _run():
        try:
            result[0] = _current_llm._call_llm(prompt)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=max_wait)

    if thread.is_alive():
        print(f"  [ModelManager] ⚠️ 推理超时 ({max_wait}s)，丢弃本次调用")
        return ""

    if error[0]:
        print(f"  [ModelManager] ⚠️ 推理异常: {error[0]}")
        return ""

    return result[0]


def chat_safe(system_ctx: str, history: list, user_msg: str,
              max_wait: float = CHAT_TIMEOUT) -> str:
    """
    安全对话 — 带超时、异常兜底。

    返回:
        模型回复文本。失败时返回错误提示。
    """
    if _current_llm is None:
        return "AI 模型未加载，请检查模型配置。"
    if not hasattr(_current_llm, 'chat'):
        return "当前模型不支持多轮对话。"

    result = [""]
    error = [None]

    def _run():
        try:
            result[0] = _current_llm.chat(system_ctx, history, user_msg)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=max_wait)

    if thread.is_alive():
        print(f"  [ModelManager] ⚠️ 对话超时 ({max_wait}s)")
        return "抱歉，AI 响应超时。请换一种方式提问试试。"

    if error[0]:
        print(f"  [ModelManager] ⚠️ 对话异常: {error[0]}")
        return "抱歉，AI 暂时无法响应。请稍后重试。"

    text = result[0].strip()
    if not text:
        return "抱歉，AI 未生成有效回复。请再试一次。"
    return text

# LLM/my_qwen.py — Qwen2.5 双模型 Provider (1.5B OpenVINO + 3B GGUF)
#
# Qwen15BProvider: openvino_genai.LLMPipeline → 速度快 (~2s)，适合评分纠正
# Qwen3BProvider:  llama.cpp GGUF → 回答精准 (~8s)，适合对话
#
# 两个模型各自单例，按需加载。

import os
import threading
from pathlib import Path

from .provider import LLMProvider

# Qwen2.5 统一使用 ChatML 对话模板
CHATML_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"
CHATML_USER = "<|im_start|>user\n{content}<|im_end|>\n"
CHATML_ASSISTANT = "<|im_start|>assistant\n"

# 依赖检测 — optimum.intel（模型为此后端导出）
try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    HAS_OV = True
except ImportError:
    HAS_OV = False

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

# ============================================================
# 工具函数
# ============================================================

def _model_dir():
    return Path(__file__).resolve().parent


def _find_ov_model():
    """查找 OpenVINO IR 模型目录。"""
    default = _model_dir() / "qwen2.5-1.5b-ov"
    if default.is_dir():
        return str(default)
    alt = Path("/home/intel/桌面/intel_competition/LLM/qwen2.5-1.5b-ov")
    if alt.is_dir():
        return str(alt)
    return str(default)


def _find_gguf_model():
    """查找 GGUF 模型文件。"""
    for p in [
        _model_dir() / "qwen2.5-3b-instruct-q4_k_m.gguf",
        Path("/home/intel/桌面/intel_competition/LLM/qwen2.5-3b-instruct-q4_k_m.gguf"),
    ]:
        if p.is_file():
            return str(p)
    return str(_model_dir() / "qwen2.5-3b-instruct-q4_k_m.gguf")


def list_available_models():
    """返回可用的模型列表 [{key, name, description, available}]。"""
    models = [
        {
            "key": "1.5b",
            "name": "Qwen2.5 1.5B",
            "description": "OpenVINO — 推理快 ~2s",
            "available": HAS_OV and os.path.isdir(_find_ov_model()),
        },
        {
            "key": "3b",
            "name": "Qwen2.5 3B",
            "description": "llama.cpp — 回答精准 ~8s",
            "available": HAS_LLAMA and os.path.isfile(_find_gguf_model()),
        },
    ]
    return models


# ============================================================
# 1.5B — OpenVINO GenAI 单例
# ============================================================

_ov_instance = None
_ov_lock = threading.Lock()


class Qwen15BProvider(LLMProvider):
    """Qwen2.5 1.5B — optimum.intel.OVModelForCausalLM 推理。速度快，适合评分纠正。"""

    def __new__(cls, *args, **kwargs):
        global _ov_instance
        if _ov_instance is None:
            with _ov_lock:
                if _ov_instance is None:
                    _ov_instance = super().__new__(cls)
                    _ov_instance._init_done = False
        return _ov_instance

    def __init__(self, model_dir=None, device="CPU", max_tokens=80):
        if self._init_done:
            return
        if not HAS_OV:
            raise ImportError("需要 optimum-intel: pip install optimum-intel")
        if model_dir is None:
            model_dir = _find_ov_model()
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        print(f"  加载 1.5B 模型: {model_dir} ...")
        self._model = OVModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            ov_config={"PERFORMANCE_HINT": "LATENCY"},
            use_cache=False, compile=False,
        )
        self._tok = AutoTokenizer.from_pretrained(model_dir)
        self._max_tokens = max_tokens
        self._init_done = True
        print(f"  ✅ 1.5B 模型就绪")

    def _generate(self, messages: list, max_new: int) -> str:
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        with _ov_lock:
            output_ids = self._model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=self._tok.eos_token_id,
            )
        new_ids = output_ids[0, input_len:]
        return self._tok.decode(new_ids, skip_special_tokens=True).strip()

    def _call_llm(self, prompt: str) -> str:
        return self._generate([
            {"role": "system", "content": "你是专业舞蹈教练，用中文简短回答。"},
            {"role": "user", "content": prompt},
        ], self._max_tokens)

    def chat(self, system_context: str, history: list, user_msg: str) -> str:
        msgs = [{"role": "system", "content": system_context}]
        for role, text in history[-6:]:
            msgs.append({"role": "assistant" if role == "ai" else "user", "content": text})
        msgs.append({"role": "user", "content": user_msg})
        return self._generate(msgs, 120)

    @property
    def provider_name(self) -> str:
        return "Qwen2.5-1.5B (OpenVINO)"


# ============================================================
# 3B — llama.cpp GGUF 单例
# ============================================================

_llama_instance = None
_llama_lock = threading.Lock()


class Qwen3BProvider(LLMProvider):
    """Qwen2.5 3B — llama.cpp GGUF 推理。回答更精准，适合 AI 对话。"""

    def __new__(cls, *args, **kwargs):
        global _llama_instance
        if _llama_instance is None:
            with _llama_lock:
                if _llama_instance is None:
                    _llama_instance = super().__new__(cls)
                    _llama_instance._init_done = False
        return _llama_instance

    def __init__(self, model_path=None, n_ctx=2048, max_tokens=80):
        if self._init_done:
            return
        if not HAS_LLAMA:
            raise ImportError("需要 llama-cpp-python: pip install llama-cpp-python")
        if model_path is None:
            model_path = _find_gguf_model()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GGUF 模型不存在: {model_path}")

        print(f"  加载 3B 模型: {os.path.basename(model_path)} ...")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=4,
            verbose=False,
        )
        self._max_tokens = max_tokens
        self._init_done = True
        print(f"  ✅ 3B 模型就绪")

    def _call_llm(self, prompt: str) -> str:
        chat = (
            CHATML_SYSTEM.format(content="你是专业舞蹈教练，用中文简短回答。")
            + CHATML_USER.format(content=prompt)
            + CHATML_ASSISTANT
        )
        with _llama_lock:
            output = self._llm(chat, max_tokens=self._max_tokens, stop=["<|im_end|>", "<|im_start|>"])
        return output["choices"][0]["text"].strip()

    def chat(self, system_context: str, history: list, user_msg: str) -> str:
        parts = [CHATML_SYSTEM.format(content=system_context)]
        for role, text in history[-6:]:
            if role == "user":
                parts.append(CHATML_USER.format(content=text))
            else:
                parts.append(f"<|im_start|>assistant\n{text}<|im_end|>\n")
        parts.append(CHATML_USER.format(content=user_msg))
        parts.append(CHATML_ASSISTANT)

        with _llama_lock:
            output = self._llm(
                "".join(parts),
                max_tokens=120,
                stop=["<|im_end|>", "<|im_start|>"],
            )
        return output["choices"][0]["text"].strip()

    @property
    def provider_name(self) -> str:
        return "Qwen2.5-3B (llama.cpp)"


# 兼容别名
MyQwenProvider = Qwen15BProvider

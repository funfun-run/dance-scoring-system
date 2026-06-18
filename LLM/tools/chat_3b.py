#!/usr/bin/env python3
"""Qwen2.5-3B GGUF 对话测试"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from llama_cpp import Llama

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dance-scoring-system", "LLM", "qwen2.5-3b-instruct-q4_k_m.gguf")

print("加载模型...", end="", flush=True)
llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=8, verbose=False)
print(" 就绪!\n")

history: list[dict] = []

while True:
    try:
        user = input("🧑 你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 再见!")
        break
    if not user:
        continue
    if user.lower() in ("quit", "exit"):
        break
    if user.lower() == "reset":
        history.clear()
        print("🔄 已重置\n")
        continue

    history.append({"role": "user", "content": user})
    parts = ["<|im_start|>system\n你是Qwen，一个有用的人工智能助手，用中文回答。\n<|im_end|>\n"]
    for m in history:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    prompt = "".join(parts)

    output = llm(prompt, max_tokens=512, temperature=0.7, top_p=0.9,
                 stop=["<|im_end|>", "<|im_start|>"])
    reply = output["choices"][0]["text"].strip()
    history.append({"role": "assistant", "content": reply})
    print(f"🤖 AI: {reply}\n")

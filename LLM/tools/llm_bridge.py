#!/usr/bin/env python3
"""
LLM Bridge — 纯 stdlib HTTP 桥接服务
在 LLM 的 venv 中运行，加载 OpenVINO 模型 (optimum.intel)，接收 /chat 请求
启动: source ov_npu_env/bin/activate && python llm_bridge.py [--port 8765]
"""

import sys
import os
import json
import signal
import argparse
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# 模型路径（指向 dance-scoring-system/LLM/ 中的模型）
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "dance-scoring-system", "LLM", "qwen2.5-1.5b-ov")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRIDGE] %(message)s")
logger = logging.getLogger("llm-bridge")

# ─── 模型加载 ───────────────────────────────────────────────────

try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
except ImportError:
    logger.error("optimum-intel / transformers 未安装！请在 LLM 的 venv 中运行本脚本")
    sys.exit(1)

model = None
tokenizer = None
model_lock = threading.Lock()


def load_model(device: str = "CPU"):
    global model, tokenizer
    logger.info(f"加载模型 (optimum): {MODEL_PATH} (device={device})")
    model = OVModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        ov_config={"PERFORMANCE_HINT": "LATENCY"},
        use_cache=False,
        compile=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    logger.info("模型加载完成")


def generate(prompt: str) -> str:
    """线程安全的推理调用 — 仅返回生成的新 tokens。"""
    with model_lock:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = output_ids[0, input_len:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─── HTTP Handler ──────────────────────────────────────────────

class BridgeHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        logger.debug(f"HTTP {args}")

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": "qwen2.5-1.5b-instruct"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/chat":
            self._json(404, {"error": "only /chat"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._json(400, {"error": "empty body"})
            return

        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        prompt = data.get("prompt", "")
        if not prompt:
            self._json(400, {"error": "missing prompt"})
            return

        try:
            reply = generate(prompt)
            self._json(200, {"reply": reply})
        except Exception as e:
            logger.error(f"推理失败: {e}")
            self._json(500, {"error": str(e)})

    def _json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ─── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Bridge Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--device", type=str, default="CPU")
    args = parser.parse_args()

    load_model(args.device)

    server = HTTPServer((args.host, args.port), BridgeHandler)

    def shutdown(sig, frame):
        logger.info("关闭服务...")
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(f"Bridge 就绪 → http://{args.host}:{args.port}")
    logger.info(f"健康检查: curl http://{args.host}:{args.port}/health")
    server.serve_forever()


if __name__ == "__main__":
    main()

# LLM 接入指南

本目录用于接入本地 AI 模型（Qwen2.5 1.5B），为舞蹈评分系统生成更自然的纠正建议。

## 快速开始

只需 2 步即可接入你的本地模型：

### 1. 创建 `LLM/my_qwen.py`

```python
# LLM/my_qwen.py
from LLM.provider import LLMProvider

class MyQwenProvider(LLMProvider):
    """Qwen2.5 1.5B 本地推理提供者。"""

    def __init__(self):
        # ===== 在这里加载你的模型 =====
        # 示例 (OpenVINO):
        #   import openvino_genai as ov_genai
        #   self.pipe = ov_genai.LLMPipeline("path/to/qwen25-1.5b-int4", "CPU")
        #   self.config = ov_genai.GenerationConfig(max_new_tokens=80)
        #
        # 示例 (transformers):
        #   from transformers import AutoModelForCausalLM, AutoTokenizer
        #   self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        #   self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        #
        # 示例 (llama.cpp):
        #   from llama_cpp import Llama
        #   self.llm = Llama("qwen2.5-1.5b-q4.gguf")
        pass

    def _call_llm(self, prompt: str) -> str:
        """调用本地模型，返回原始文本。"""
        # ===== 在这里实现推理调用 =====
        # 示例 (OpenVINO):
        #   return self.pipe.generate(prompt, self.config)
        #
        # 示例 (transformers):
        #   inputs = self.tokenizer(prompt, return_tensors="pt")
        #   outputs = self.model.generate(**inputs, max_new_tokens=80)
        #   return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #
        # 示例 (llama.cpp):
        #   output = self.llm(prompt, max_tokens=80)
        #   return output["choices"][0]["text"]
        raise NotImplementedError("请实现 _call_llm() 方法")

    @property
    def provider_name(self) -> str:
        return "Qwen2.5-1.5B"
```

### 2. 使用 LLM 纠正

```bash
# CLI
python scripts/score.py -r videos/reference.mp4 -u videos/user.mp4 --correction llm

# GUI — 在评分面板中选择纠正后端 "llm"
python src/dance_scoring/gui/app.py
```

## 接口契约

### 输入（由基类自动构建）

基类 `LLMProvider._build_prompt()` 会自动将结构化数据拼接为以下格式：

```
你是一个专业的舞蹈教练。根据学员的关节角度偏差数据...

总分: 72.5 | BPM: 120
第3段: 得分55.0 | 不合格
偏差最大的关节:
- 右肘: +12.5° (手臂弯曲过度)
- 左膝: -8.3° (下蹲不够)
纠正建议:
```

### 输出（由基类自动解析）

`_parse_response()` 会按以下优先级解析模型输出：
1. JSON: `{"correction": "..."}`
2. 行式: 提取第一行非空文本
3. 兜底: 返回原始文本前 80 字符

你的模型只需要返回纯文本，不需要严格的 JSON。

## 模块结构

```
LLM/
├── __init__.py      # 导出 LLMProvider
├── provider.py      # LLMProvider 基类 — Prompt 构建 + 响应解析
├── prompts.py       # Prompt 模板 — 可根据模型微调
├── my_qwen.py       # ← 你创建的文件
└── README.md        # 本文件
```

## 模型要求

| 项目 | 建议 |
|------|------|
| 模型 | Qwen2.5 1.5B Instruct（或任意中文对话模型） |
| 量化 | INT4 推荐（~1.5GB 内存） |
| 输出长度 | max_tokens=80 即可（纠正建议 ≤ 50 字） |
| 推理延迟 | 每次调用 < 2s 可接受（非实时路径） |

## 故障排除

**Q: `--correction llm` 报 "未找到可用的 LLMProvider 子类"？**
A: 请确认 `LLM/my_qwen.py` 存在且定义了 `MyQwenProvider(LLMProvider)` 类。

**Q: 模型输出乱码或格式错误？**
A: `_parse_response()` 有三级 fallback。检查模型是否正确处理中文。可以在 `prompts.py` 中调整 `SYSTEM_PROMPT`。

**Q: 合格段也调用了 LLM？**
A: 不会。`qualified=True` 的段会跳过 LLM 调用，直接返回空字符串。

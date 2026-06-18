# LLM/ — 本地 AI 模型接入接口
#
# 使用方式:
#   1. 创建 LLM/my_qwen.py，继承 LLMProvider，实现 _call_llm()
#   2. 在 create_correction_provider("llm") 中注册你的 Provider 类
#   3. 启动评分即可自动使用 LLM 生成纠正建议

from .provider import LLMProvider

__all__ = ["LLMProvider"]

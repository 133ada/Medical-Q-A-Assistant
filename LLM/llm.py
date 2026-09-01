"""
Memory/llm.py
=============
Memory 模块专用 LLM 工厂（Summary / Memory 提取共用）。
与 Medical_agent_graph.py 中 _make_llm 等价，但独立于此，避免循环依赖。
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from App.Core import config


def make_llm(temperature: float = 0.0):
    """创建 Memory 模块使用的 DeepSeek LLM 实例。"""
    return init_chat_model(
        model=config.CONFIG_DEEPSEEK["model"],
        model_provider="deepseek",
        base_url=config.CONFIG_DEEPSEEK["base_url"],
        api_key=config.CONFIG_DEEPSEEK["api_key"],
        temperature=temperature,
    )


def make_embedding_llm(model: str, openai_api_key: str, openai_api_base: str, dimensions: int):
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        dimensions=dimensions,
        chunk_size=10,
        check_embedding_ctx_length=False,
    )

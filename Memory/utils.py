"""
Memory/Utils.py
===============
Memory 模块共享的小工具：token 计数、JSON 安全解析、时间序列化。
不依赖 LangGraph / LangChain 业务对象，可独立测试。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from App.Core.config import TOKEN_ENCODER_NAME

logger = logging.getLogger(__name__)

# tiktoken 编码器缓存（编码器创建较慢，全局复用）
_encoder_cache: dict[str, Any] = {}


def count_tokens(text: str, encoder_name: str = TOKEN_ENCODER_NAME) -> int:
    """
    估算文本的 token 数量。

    DeepSeek 无官方公开 tokenizer，这里用 tiktoken 的 cl100k_base 近似估算
    （文档第七节：Summary 触发基于"历史上下文累计 Token 数量"）。
    """
    if not text:
        return 0
    try:
        enc = _encoder_cache.get(encoder_name)
        if enc is None:
            try:
                enc = __import__("tiktoken").encoding_for_model(encoder_name)
            except Exception:
                enc = __import__("tiktoken").get_encoding("cl100k_base")
            _encoder_cache[encoder_name] = enc
        return len(enc.encode(text))
    except Exception as exc:  # tokenizer 不可用时按字符数粗略估算，保证不阻塞主流程
        logger.warning("tiktoken 不可用，按字符数/4 估算 token：%s", exc)
        return max(1, len(text) // 4)


def safe_json_loads(text: str, fallback: Any = None) -> Any:
    """安全解析 JSON 字符串；解析失败返回 fallback（不抛异常）。"""
    if not text:
        return fallback
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            return json.loads(text.strip())
        except Exception:
            return fallback


def extract_json_object(text: str) -> Any:
    """
    从 LLM 输出中提取 JSON 对象/数组。

    按顺序尝试：
      1. 直接 json.loads
      2. 剥离 ```json ... ``` 代码块
      3. 提取第一个 { ... } 或 [ ... ] 块
    解析失败返回 None。
    """
    if not text:
        return None

    text = text.strip()
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _json_candidates(text: str):
    yield text
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        yield m.group(1).strip()
    m = re.search(r"[\{\[][\s\S]*?[\}\]]", text)
    if m:
        yield m.group(0)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_to_dt(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

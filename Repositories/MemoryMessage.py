"""
=================
短期记忆消息模型（Pydantic）。

设计要点（CoverLayWindow.md 第四节）：
  · 短期记忆层不直接依赖 LangChain 内部对象，统一使用 MemoryMessage。
  · 提供 MemoryMessage ⇄ LangChain Message 的转换方法，仅在需要时使用。
  · Redis 层存储为 JSON 字符串，字段自洽、可独立反序列化。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Union
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MemoryMessage(BaseModel):
    """短期记忆中的一条消息。role 取值 user / assistant / system。"""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=_utcnow)

    # ── 序列化 ──────────────────────────────────────────────────────────────

    def to_redis_json(self) -> str:
        """序列化为 Redis 中保存的 JSON 字符串。"""
        return self.model_dump_json()

    @classmethod
    def from_redis_json(cls, raw: str) -> "MemoryMessage":
        """从 Redis 中的 JSON 字符串反序列化。"""
        return cls.model_validate_json(raw)

    # ── LangChain 互转 ───────────────────────────────────────────────────────

    def to_langchain(self):
        """转为 LangChain HumanMessage / AIMessage / SystemMessage。"""
        mapping = {
            "user": HumanMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
        }
        return mapping[self.role](content=self.content)

    @classmethod
    def from_langchain(cls, message) -> "MemoryMessage":
        """从 LangChain Message 转为 MemoryMessage。"""
        if isinstance(message, HumanMessage):
            role: MessageRole = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            # ToolMessage 等未知类型，保守视为 user 输入
            role = "user"
        return cls(role=role, content=str(message.content), timestamp=_utcnow())

    @classmethod
    def user(cls, content: str) -> "MemoryMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "MemoryMessage":
        return cls(role="assistant", content=content)


# 供 FastAPI / 调用方作为"可写消息"的便捷类型
AnyMessageLike = Union[MemoryMessage, dict, str]

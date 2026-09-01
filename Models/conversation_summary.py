"""Export the Memory-owned conversation-summary ORM entity."""
from __future__ import annotations

"""
长期记忆相关 SQLAlchemy 表模型（CoverLayWindow.md 第十四节）。

两张表：
  · conversation_summary —— 每个会话最新的 Conversation Summary（长期 Summary）
  · long_term_memory    —— 四类长期记忆：preference / medication / health_context / behavior

字段遵循文档约定：
  · long_term_memory.status     : candidate / confirmed / active / inactive / expired / deleted
  · long_term_memory.source     : user / assistant_inference / system / medical_record
  · long_term_memory.confidence : 0.0 ~ 1.0
  · memory_value 以 JSON 字符串保存结构化信息（与数据库后端无关，PG/SQLite 通用）
"""

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from App.Models.base import Base

MEMORY_TYPES: tuple = ("preference", "medication", "health_context", "behavior")

MEMORY_STATUS: tuple = (
    "candidate", "confirmed", "active", "inactive", "expired", "deleted",
)

MEMORY_SOURCES: tuple = ("user", "assistant_inference", "system", "medical_record")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationSummary(Base):
    """每个 (user_id, conversation_id) 最新的对话总结。"""

    __tablename__ = "conversation_summary"
    __table_args__ = (
        UniqueConstraint("user_id", "conversation_id", name="uq_summary_user_conv"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # 调试/优化用：触发 summary 时的历史 token 数与 summary 自身 token 数
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary_token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "version": self.version,
            "token_count": self.token_count,
            "summary_token_count": self.summary_token_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

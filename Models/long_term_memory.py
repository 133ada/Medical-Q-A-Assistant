"""Export the Memory-owned long-term-Memory ORM entity."""
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


class LongTermMemory(Base):
    """一条长期记忆。memory_key + memory_value 构成记忆内容，类型见 MEMORY_TYPES。"""

    __tablename__ = "long_term_memory"
    __table_args__ = (
        UniqueConstraint("user_id", "memory_type", "memory_key", name="uq_ltm_user_type_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    memory_key: Mapped[str] = mapped_column(String(255), nullable=False)
    memory_value: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="candidate", index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "memory_key": self.memory_key,
            "memory_value": self.memory_value,
            "status": self.status,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }

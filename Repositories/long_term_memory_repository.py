"""Compatibility façade for Memory long-term persistence."""
from __future__ import annotations

"""
repository/sql_memory_repository.py
===================================
长期记忆 SQL 仓库（CoverLayWindow.md 第十五～十六节）。

职责：
  · Conversation Summary 的读取 / 新建 / 更新（upsert，按 user_id+conversation_id）
  · 长期记忆的读取 / 新增 / 更新 / 失效 / 去重（按 user_id+memory_type+memory_key）

只做持久化 CRUD，不含 LLM / Agent 调度逻辑。
数据库不可用时方法抛异常，由 MemoryManager 统一降级。
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from App.Memory.database import session_scope
from App.Models import ConversationSummary, LongTermMemory
from App.Memory.utils import safe_json_loads

logger = logging.getLogger(__name__)

# 默认参与检索/上下文构建的状态（已失效/已删除的不进入）
ACTIVE_STATUSES = ("active", "confirmed", "candidate")


class SQLMemoryRepository:
    """
    长期记忆 SQL 仓库。

    并发控制原则：

    1. 写操作必须处于事务中。
    2. 对已有记录执行 SELECT FOR UPDATE。
    3. 查询和后续 UPDATE 必须使用同一个 SQLAlchemy Session。
    4. 数据库唯一约束负责兜底，避免并发 INSERT 产生重复数据。
    """

    def __init__(self, session_factory: Optional[sessionmaker] = None):
        self.session_factory = session_factory

    # =========================================================================
    # Conversation Summary
    # =========================================================================

    def get_summary(self, user_id: str, conversation_id: str) -> Optional[dict]:
        """读取指定会话的最新 Summary；不存在返回 None。"""
        with session_scope(self.session_factory) as session:
            row = session.execute(
                select(ConversationSummary).where(
                    ConversationSummary.user_id == user_id,
                    ConversationSummary.conversation_id == conversation_id,
                )
            ).scalar_one_or_none()
            return row.to_dict() if row else None

    def upsert_summary(
            self,
            user_id: str,
            conversation_id: str,
            summary: str,
            version: Optional[int] = None,
            token_count: Optional[int] = None,
            summary_token_count: Optional[int] = None,
    ) -> dict:
        """按 (user_id, conversation_id) 更新或新建 Summary。"""
        with session_scope(self.session_factory) as session:
            row = session.execute(
                select(ConversationSummary)
                .where(
                    ConversationSummary.user_id == user_id,
                    ConversationSummary.conversation_id == conversation_id,
                ).with_for_update()
            ).scalar_one_or_none()

            if row is None:
                row = ConversationSummary(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    summary=summary,
                    version=version or 1,
                    token_count=token_count,
                    summary_token_count=summary_token_count,
                )
                session.add(row)
            else:
                row.summary = summary
                if version is not None:
                    row.version = version
                else:
                    row.version += 1
                if token_count is not None:
                    row.token_count = token_count
                if summary_token_count is not None:
                    row.summary_token_count = summary_token_count
            session.flush()
            return row.to_dict()

    # =========================================================================
    # Long-term Memory
    # =========================================================================

    def list_memories(
            self,
            user_id: str,
            memory_type: Optional[str] = None,
            statuses: Optional[tuple] = ACTIVE_STATUSES,
            limit: Optional[int] = None,
    ) -> List[dict]:
        """列出用户的长期记忆；默认只取 active/confirmed/candidate 状态。"""
        with session_scope(self.session_factory) as session:
            stmt = select(LongTermMemory).where(LongTermMemory.user_id == user_id)
            if memory_type:
                stmt = stmt.where(LongTermMemory.memory_type == memory_type)
            if statuses:
                stmt = stmt.where(LongTermMemory.status.in_(statuses))
            stmt = stmt.order_by(LongTermMemory.updated_at.desc())
            if limit:
                stmt = stmt.limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [r.to_dict() for r in rows]

    def find_memory(self, user_id: str, memory_type: str, memory_key: str) -> Optional[dict]:
        """按 (user_id, memory_type, memory_key) 查一条记忆（含已失效的）。"""
        with session_scope(self.session_factory) as session:
            row = session.execute(
                select(LongTermMemory).where(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.memory_type == memory_type,
                    LongTermMemory.memory_key == memory_key,
                )
            ).scalar_one_or_none()
            return row.to_dict() if row else None

    def upsert_memory(
            self,
            user_id: str,
            memory_type: str,
            memory_key: str,
            memory_value: str,
            status: str,
            confidence: float,
            source: str,
            expires_at: Optional[datetime] = None,
    ) -> dict:
        """
        新增或更新一条长期记忆（去重键：user_id + memory_type + memory_key）。

        · medication 更新：命中已有记录则直接覆盖（用户最新信息优先，文档第二十节）
        · behavior 聚合：命中已有记录时 frequency + 1
        · 已失效（deleted）的记录复用：恢复并更新为最新状态
        """
        with session_scope(self.session_factory) as session:
            row = session.execute(
                select(LongTermMemory).where(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.memory_type == memory_type,
                    LongTermMemory.memory_key == memory_key,
                )
            ).scalar_one_or_none()

            if row is None:
                row = LongTermMemory(
                    user_id=user_id,
                    memory_type=memory_type,
                    memory_key=memory_key,
                    memory_value=memory_value,
                    status=status,
                    confidence=confidence,
                    source=source,
                    expires_at=expires_at,
                )
                session.add(row)
            else:
                if memory_type == "behavior":
                    # 行为记忆：聚合频次，而不是覆盖
                    existing_value = safe_json_loads(row.memory_value, {}) or {}
                    new_value = safe_json_loads(memory_value, {}) or {}
                    freq = int(existing_value.get("frequency", 0)) + 1
                    merged = dict(new_value)
                    merged["frequency"] = freq
                    merged["topic"] = memory_key
                    row.memory_value = json.dumps(merged, ensure_ascii=False)
                else:
                    row.memory_value = memory_value
                row.status = status
                row.confidence = confidence
                row.source = source
                row.expires_at = expires_at
            session.flush()
            return row.to_dict()

    def set_status(
            self, user_id: str, memory_type: str, memory_key: str, status: str
    ) -> Optional[dict]:
        """将指定记忆置为某状态（例如 inactive / expired / deleted）。"""
        with session_scope(self.session_factory) as session:
            row = session.execute(
                select(LongTermMemory).where(
                    LongTermMemory.user_id == user_id,
                    LongTermMemory.memory_type == memory_type,
                    LongTermMemory.memory_key == memory_key,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            row.status = status
            session.flush()
            return row.to_dict()

    def delete_memory(self, memory_id: int) -> bool:
        """物理删除一条记忆（一般用 set_status 置为 deleted 替代）。"""
        with session_scope(self.session_factory) as session:
            row = session.get(LongTermMemory, memory_id)
            if row is None:
                return False
            session.delete(row)
            return True

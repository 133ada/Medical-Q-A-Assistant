"""
repository/redis_memory_repository.py
=====================================
短期记忆仓库（CoverLayWindow.md 第三节）。

· Redis Key : Memory:short:{user_id}:{conversation_id}（不同用户/会话互相隔离）
· 存储结构  : Redis List（消息队列语义），元素为 MemoryMessage 的 JSON 字符串
· 条数限制  : 最多 SHORT_TERM_MAX_MESSAGES 条，写入时 ltrim 保留最新 N 条
· TTL       : SHORT_TERM_TTL_SECONDS 秒，每次写入刷新（滑动过期）

本模块只做 Redis CRUD，不含任何 LLM / Agent 逻辑。
Redis 不可用时抛异常，由 MemoryManager 统一降级（文档第二十八节）。

附带 InMemoryShortTermRepository —— 相同接口的内存实现，
用于 Redis 不可用时的兜底（fallback_to_memory=True）与单元测试。
"""

from __future__ import annotations
import time
import logging
from collections import deque
from typing import Deque, List, Optional

import redis as redis_lib

from App.Core.config import (
    REDIS_URL,
    SHORT_TERM_FALLBACK_TO_MEMORY,
    SHORT_TERM_MAX_MESSAGES,
    SHORT_TERM_TTL_SECONDS,
    SHORT_TERM_USE_REDIS,
)
from App.Repositories import MemoryMessage

logger = logging.getLogger(__name__)

# 默认 Key 前缀（第三节建议）
KEY_PREFIX = "Memory:short"


class RedisMemoryRepository:
    """
    短期记忆 Redis CRUD。

    Parameters
    ----------
    client       : 传入的 redis.Redis 客户端；不传则按 REDIS_URL 自动创建。
    key_prefix   : Redis Key 前缀。
    max_messages : 最多保留的消息条数（默认 20）。
    ttl_seconds  : 短期记忆 TTL 秒数（默认 86400 = 1 天）。
    """

    def __init__(
            self,
            client: Optional[redis_lib.Redis] = None,
            key_prefix: str = KEY_PREFIX,
            max_messages: int = SHORT_TERM_MAX_MESSAGES,
            ttl_seconds: int = SHORT_TERM_TTL_SECONDS,
    ):
        self._redis = client or redis_lib.from_url(
            REDIS_URL, decode_responses=True, socket_connect_timeout=3, socket_timeout=3
        )
        self.key_prefix = key_prefix
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

    def _key(self, user_id: str, conversation_id: str) -> str:
        return f"{self.key_prefix}:{user_id}:{conversation_id}"

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def save_message(self, user_id: str, conversation_id: str, message: MemoryMessage) -> None:
        """
        追加一条消息：rpush → 截断到最新 max_messages 条 → 刷新 TTL。
        """
        key = self._key(user_id, conversation_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, message.to_redis_json())
        pipe.ltrim(key, -self.max_messages, -1)  # 只保留最新 max_messages 条
        pipe.expire(key, self.ttl_seconds)  # 滑动过期
        pipe.execute()

    def save_messages(
            self, user_id: str, conversation_id: str, messages: List[MemoryMessage]
    ) -> None:
        """批量保存（例如一次保存 User + Assistant 两条）。"""
        for msg in messages:
            self.save_message(user_id, conversation_id, msg)

    # ── 读取 ──────────────────────────────────────────────────────────────────

    def get_messages(
            self, user_id: str, conversation_id: str, limit: Optional[int] = None
    ) -> List[MemoryMessage]:
        """
        读取最近的消息（默认全部，limit 时只取最新 limit 条）。

        空列表 = 没有短期记忆（首次对话）。
        """
        key = self._key(user_id, conversation_id)
        if limit is not None and limit > 0:
            raw_list = self._redis.lrange(key, -limit, -1)
        else:
            raw_list = self._redis.lrange(key, 0, -1)
        return [MemoryMessage.from_redis_json(r) for r in raw_list]

    # ── 其他 ──────────────────────────────────────────────────────────────────

    def count(self, user_id: str, conversation_id: str) -> int:
        return int(self._redis.llen(self._key(user_id, conversation_id)))

    def clear(self, user_id: str, conversation_id: str) -> None:
        self._redis.delete(self._key(user_id, conversation_id))

    def ttl(self, user_id: str, conversation_id: str) -> int:
        return int(self._redis.ttl(self._key(user_id, conversation_id)))

    @classmethod
    def ping_redis(cls, url: str = REDIS_URL) -> bool:
        """探测 Redis 是否可用。"""
        try:
            client = redis_lib.from_url(
                url, decode_responses=True, socket_connect_timeout=3, socket_timeout=3
            )
            return bool(client.ping())
        except Exception:
            return False


class InMemoryShortTermRepository:
    """
    与 RedisMemoryRepository 相同接口的内存实现（兜底 / 测试）。

    · 用 collections.deque(maxlen=max_messages) 天然实现"最多 N 条，丢弃最旧"
    · 记录写入时间戳，模拟 TTL（超过 ttl_seconds 未写则视为已过期清空）
    """

    def __init__(
            self,
            max_messages: int = SHORT_TERM_MAX_MESSAGES,
            ttl_seconds: int = SHORT_TERM_TTL_SECONDS,
    ):
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        # key -> deque[MemoryMessage]
        self._store: dict[str, Deque[MemoryMessage]] = {}
        # key -> 最近一次写入时间（epoch 秒，用于模拟 TTL）
        self._last_write: dict[str, float] = {}

    def _key(self, user_id: str, conversation_id: str) -> str:
        return f"{user_id}:{conversation_id}"

    def _is_expired(self, key: str) -> bool:
        last = self._last_write.get(key)
        if last is None:
            return True
        return _now() - last > self.ttl_seconds

    def save_message(self, user_id: str, conversation_id: str, message: MemoryMessage) -> None:
        key = self._key(user_id, conversation_id)
        dq = self._store.setdefault(key, deque(maxlen=self.max_messages))
        dq.append(message)
        self._last_write[key] = _now()

    def save_messages(
            self, user_id: str, conversation_id: str, messages: List[MemoryMessage]
    ) -> None:
        for msg in messages:
            self.save_message(user_id, conversation_id, msg)

    def get_messages(
            self, user_id: str, conversation_id: str, limit: Optional[int] = None
    ) -> List[MemoryMessage]:
        key = self._key(user_id, conversation_id)
        if self._is_expired(key):
            self.clear(user_id, conversation_id)
            return []
        dq = self._store.get(key, deque())
        msgs = list(dq)
        if limit is not None and limit > 0:
            msgs = msgs[-limit:]
        return msgs

    def count(self, user_id: str, conversation_id: str) -> int:
        key = self._key(user_id, conversation_id)
        if self._is_expired(key):
            self.clear(user_id, conversation_id)
            return 0
        return len(self._store.get(key, deque()))

    def clear(self, user_id: str, conversation_id: str) -> None:
        key = self._key(user_id, conversation_id)
        self._store.pop(key, None)
        self._last_write.pop(key, None)

    def ttl(self, user_id: str, conversation_id: str) -> int:
        key = self._key(user_id, conversation_id)
        if self._is_expired(key):
            return -2  # 与 Redis 语义一致：-2 表示 key 不存在
        remaining = self.ttl_seconds - (_now() - self._last_write.get(key, 0))
        return int(remaining)


def _now() -> float:
    return time.time()


def create_short_term_store() -> Optional[RedisMemoryRepository]:
    """
    创建短期记忆存储实例。

    策略（文档第二十八节：Redis 不可用不能拖垮问答）：
      · SHORT_TERM_USE_REDIS=True 且 Redis 可用   → RedisMemoryRepository
      · Redis 不可用且 SHORT_TERM_FALLBACK_TO_MEMORY=True → 内存兜底（记 warning）
      · Redis 不可用且不允许兜底                 → 返回 None（Manager 降级为空上下文）
    """
    if not SHORT_TERM_USE_REDIS:
        if SHORT_TERM_FALLBACK_TO_MEMORY:
            logger.warning("短期记忆：已禁用 Redis，使用内存兜底存储")
            return InMemoryShortTermRepository()
        return None

    if RedisMemoryRepository.ping_redis(REDIS_URL):
        logger.info("短期记忆：连接 Redis 成功（%s）", REDIS_URL.split("@")[-1])
        return RedisMemoryRepository()

    if SHORT_TERM_FALLBACK_TO_MEMORY:
        logger.warning(
            "短期记忆：无法连接 Redis（%s），已降级为内存存储。"
            "如需真实 Redis，请启动 Redis 服务器或检查 REDIS_URL。",
            REDIS_URL.split("@")[-1],
        )
        return InMemoryShortTermRepository()

    logger.warning("短期记忆：无法连接 Redis 且未启用内存兜底，短期记忆不可用")
    return None

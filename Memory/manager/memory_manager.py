"""
manager/memory_manager.py
=========================
MemoryManager —— 统一协调短期记忆 / Summary / 长期记忆（CoverLayWindow.md 第二十五节）。

对外方法（供 FastAPI 与 LangGraph 集成层调用）：
  · build_memory_context()     构建本轮 Memory Context（读）
  · save_message()             保存单条短期消息
  · update_after_turn()        每轮对话结束后的完整更新：存消息→判断 Summary→提取长期记忆（写）
  · trigger_summary_if_needed() Token 阈值触发 Summary

  Memory 是增强模块，任何一步失败都不能让主问答链路崩溃 ——
  读失败 → 返回空上下文；写失败 → 记 warning 并继续。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from App.Core.config import (
    LONG_TERM_MAX_IN_CONTEXT,
    SUMMARY_KEEP_RECENT_MESSAGES,
    SUMMARY_TOKEN_THRESHOLD,
)
from App.Memory.extractor import MemoryExtractor
from App.Repositories.MemoryMessage import MemoryMessage
from App.Repositories.long_term_memory_repository import SQLMemoryRepository
from App.Memory.retriever import MemoryRetriever
from App.Memory.summarizer import ConversationSummarizer
from App.Memory.utils import count_tokens

logger = logging.getLogger(__name__)

# 用户/会话缺失时的兜底标识
DEFAULT_USER_ID = "default"


def _empty_context(user_id: str, conversation_id: str) -> dict:
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "recent_messages": [],
        "conversation_summary": "",
        "long_term_memories": [],
        "retrieved_count": 0,
    }


def format_memory_context(memory_context: dict) -> str:
    """
    把 Memory Context 渲染成给 LLM 看的文本（供 LangGraph Answer 节点 Prompt 使用）。
    敏感信息不落日志，但可进入 Prompt 辅助回答。
    """
    parts: List[str] = []

    recent = memory_context.get("recent_messages") or []
    if recent:
        lines = []
        for m in recent:
            speaker = {"user": "用户", "assistant": "助手", "system": "系统"}.get(
                m.get("role", ""), "未知"
            )
            lines.append(f"{speaker}: {m.get('content', '')}")
        parts.append("【近期对话】\n" + "\n".join(lines))
    else:
        parts.append("【近期对话】\n（暂无近期对话）")

    summary = memory_context.get("conversation_summary") or ""
    parts.append(
        f"【历史对话摘要】\n{summary}" if summary else "【历史对话摘要】\n（暂无历史摘要）"
    )

    long_mem = memory_context.get("long_term_memories") or []
    if long_mem:
        lines = []
        for m in long_mem:
            try:
                value = json.loads(m.get("memory_value") or "{}")
                detail = ", ".join(
                    f"{k}={v}" for k, v in value.items() if v not in (None, "")
                )
            except (json.JSONDecodeError, TypeError):
                detail = str(m.get("memory_value", ""))
            lines.append(
                f"- [{m.get('memory_type')}] {m.get('memory_key')}"
                f"（status={m.get('status')}, 置信度={m.get('confidence')}）"
                + (f"：{detail}" if detail else "")
            )
        parts.append("【该用户相关的长期记忆】\n" + "\n".join(lines))
    else:
        parts.append("【该用户相关的长期记忆】\n（暂无）")

    return "\n\n".join(parts)


class MemoryManager:
    """Memory 模块统一协调器。所有组件可注入，便于测试。"""

    def __init__(
            self,
            short_store=None,
            sql_repo: Optional[SQLMemoryRepository] = None,
            summarizer: Optional[ConversationSummarizer] = None,
            extractor: Optional[MemoryExtractor] = None,
            retriever: Optional[MemoryRetriever] = None,
    ):
        # 短期记忆存储：RedisMemoryRepository / InMemoryShortTermRepository / None
        self.short_store = short_store
        self.sql_repo = sql_repo if sql_repo is not None else SQLMemoryRepository()
        self.summarizer = summarizer if summarizer is not None else ConversationSummarizer()
        self.extractor = extractor if extractor is not None else MemoryExtractor()
        self.retriever = retriever if retriever is not None else MemoryRetriever()

    # =========================================================================
    # 读取：构建 Memory Context
    # =========================================================================

    def build_memory_context(
            self, user_id: Optional[str], conversation_id: Optional[str], question: str
    ) -> dict:
        """
        每轮提问前构建 Memory Context。

        Returns
        -------
        dict:
            recent_messages     短期记忆（最近 20 条）
            conversation_summary 当前会话的长期 Summary
            long_term_memories  按 Query 相关性召回的用户长期记忆
        """
        # Anonymous requests must not read a shared fallback user's Memory.
        if not user_id:
            return _empty_context("", conversation_id or "")
        context = _empty_context(user_id, conversation_id or "")

        if not conversation_id:
            return context

        # 1) 短期记忆
        if self.short_store is not None:
            try:
                msgs = self.short_store.get_messages(user_id, conversation_id)
                context["recent_messages"] = [m.model_dump(mode="json") for m in msgs]
            except Exception as exc:
                logger.warning("短期记忆读取失败，使用空上下文：%s", exc)

        # 2) 长期 Summary
        if self.sql_repo is not None:
            try:
                row = self.sql_repo.get_summary(user_id, conversation_id)
                if row:
                    context["conversation_summary"] = row.get("summary") or ""
            except Exception as exc:
                logger.warning("Conversation Summary 读取失败：%s", exc)

            # 3) 长期记忆相关性检索
            try:
                memories = self.sql_repo.list_memories(user_id)
                if memories:
                    recalled = self.retriever.retrieve(question, memories)
                    context["long_term_memories"] = recalled[:LONG_TERM_MAX_IN_CONTEXT]
                    context["retrieved_count"] = len(recalled)
            except Exception as exc:
                logger.warning("长期记忆检索失败：%s", exc)

        return context

    # =========================================================================
    # 写入：每轮对话结束后的更新，默认写入 redis 中
    # =========================================================================

    def save_message(
            self, user_id: str, conversation_id: str, role: str, content: str
    ) -> None:
        """保存一条短期消息（User / Assistant / System）。"""
        if not user_id or not conversation_id or self.short_store is None:
            return
        try:
            self.short_store.save_message(
                user_id, conversation_id, MemoryMessage(role=role, content=content)
            )
        except Exception as exc:
            logger.warning("短期记忆写入失败：%s", exc)

    def update_after_turn(
            self,
            user_id: Optional[str],
            conversation_id: Optional[str],
            question: str,
            answer: str,
    ) -> None:
        """

        1. 保存 User 消息 + Assistant 消息
        2. Token >= 阈值 → 生成新 Summary
        3. 提取长期记忆并新增/更新/失效

        该方法在后台线程执行（FastAPI BackgroundTasks），内部全部容错。
        """
        # Do not persist anonymous conversations into a shared user namespace.
        if not user_id or not conversation_id or not question:
            return

        self.save_message(user_id, conversation_id, "user", question)
        self.save_message(user_id, conversation_id, "assistant", answer)

        # 历史摘要压缩（含 token 阈值判断）
        self.trigger_summary_if_needed(user_id, conversation_id)

        # 长期记忆提取（LLM，失败不影响主链路）
        self.extract_and_update(user_id, question, answer)

    # =========================================================================
    # Summary 触发
    # =========================================================================

    def trigger_summary_if_needed(self, user_id: str, conversation_id: str) -> bool:
        """
        当"旧 Summary + 短期消息"累计 Token >= 阈值时，生成并保存新 Summary。

        Returns
        -------
        bool : 本次是否触发了 Summary
        """
        if self.short_store is None or self.sql_repo is None:
            return False
        try:
            msgs = self.short_store.get_messages(user_id, conversation_id)
            row = self.sql_repo.get_summary(user_id, conversation_id)
            old_summary = row["summary"] if row else ""

            msg_tokens = count_tokens(
                "\n".join(m.content for m in msgs)
            )
            summary_tokens = count_tokens(old_summary)
            total_tokens = msg_tokens + summary_tokens

            logger.info(
                "Summary 检查：user=%s conv=%s 消息 %d 条 / %d tokens + 摘要 %d tokens = %d（阈值 %d）",
                user_id, conversation_id, len(msgs), msg_tokens, summary_tokens,
                total_tokens, SUMMARY_TOKEN_THRESHOLD,
            )

            if total_tokens < SUMMARY_TOKEN_THRESHOLD:
                return False

            # 触发 Summary：旧 Summary + 近期消息 → 新 Summary
            # 将新的 Summary 保存在 数据库表conversation_summary 中去
            new_summary = self.summarizer.summarize(msgs, old_summary)
            self.sql_repo.upsert_summary(
                user_id,
                conversation_id,
                new_summary,
                token_count=total_tokens,
                summary_token_count=count_tokens(new_summary),
            )
            # 保留最近 N 条消息（短期记忆[ Redis | 内存 ]本身已按 max_messages 截断）
            recent_count = len(
                self.short_store.get_messages(user_id, conversation_id)
            )
            logger.info(
                "Summary 已触发并保存：新摘要 %d 字，保留最近 %d 条消息（上限 %d）",
                len(new_summary), recent_count, SUMMARY_KEEP_RECENT_MESSAGES,
            )
            return True
        except Exception as exc:
            logger.warning("Summary 触发失败（不影响主链路）：%s", exc)
            return False

    # =========================================================================
    # 长期记忆[ 对应数据库表 long_term_memory ]提取与更新
    # =========================================================================

    def extract_and_update(self, user_id: str, question: str, answer: str) -> int:
        """
        从本轮对话提取候选长期记忆并落库。

        Returns
        -------
        int : 落库的候选记忆条数
        """
        if self.sql_repo is None:  # 若数据库不存在
            return 0
        try:
            # 列出数据库中已经存在的 用户的长期记忆
            existing = self.sql_repo.list_memories(user_id)
            candidates = self.extractor.extract(question, answer, existing=existing)
            if not candidates:
                return 0
            self._apply_candidates(user_id, candidates)
            return len(candidates)
        except Exception as exc:
            logger.warning("长期记忆提取/更新失败（不影响主链路）：%s", exc)
            return 0

    def _apply_candidates(self, user_id: str, candidates: List[dict]) -> None:
        """把候选记忆（cand）与已有记忆（existing）对比后落库：新增 / 更新 / 作废。"""
        for cand in candidates:
            mem_type = cand["type"]
            key = cand["key"]
            status = cand["status"]
            value_json = json.dumps(cand.get("value") or {}, ensure_ascii=False)
            expires_at = None
            if cand.get("expires_in_days"):
                expires_at = datetime.now(timezone.utc) + timedelta(
                    days=int(cand["expires_in_days"])
                )

            existing = self.sql_repo.find_memory(user_id, mem_type, key)

            if status in ("inactive", "expired", "deleted"):
                # 用户明确停用/过期 → 更新已有记录状态，而不是新建冲突记录
                if existing is not None:
                    self.sql_repo.set_status(user_id, mem_type, key, status)
                    logger.info(
                        "长期记忆更新：user=%s [%s] %s → %s", user_id, mem_type, key, status
                    )
                else:
                    self.sql_repo.upsert_memory(
                        user_id, mem_type, key, value_json, status,
                        cand["confidence"], cand["source"], expires_at,
                    )
                continue

            # 新增 / 覆盖更新（用户最新信息优先）
            self.sql_repo.upsert_memory(
                user_id, mem_type, key, value_json, status,
                cand["confidence"], cand["source"], expires_at,
            )
            logger.info(
                "长期记忆落库：user=%s [%s] %s (status=%s, conf=%s, source=%s)",
                user_id, mem_type, key, status, cand["confidence"], cand["source"],
            )

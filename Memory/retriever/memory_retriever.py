"""
retriever/memory_retriever.py
=============================
长期记忆相关性检索

职责：
    Query → 对用户长期记忆打分 → 相关性过滤 → Top-K

· 采用确定性的词元（bigram）重叠 + 关键术语命中打分，中文无需分词即可用，
  且不依赖外部 API（记忆检索路径不做网络依赖，保证 Memory 不是单点故障）。
· medication 药名 / behavior 主题命中查询时优先召回（文档第十八节示例：
  查"阿司匹林和布洛芬可以一起吃吗"应优先召回用药记忆，而不是用户偏好）。
· 支持按 memory_type 加权与数量限制。
"""

from __future__ import annotations

import logging
from typing import List, Optional

from App.Core.config import LONG_TERM_TOP_K
from App.Memory.utils import safe_json_loads

logger = logging.getLogger(__name__)

# 相关性分数低于该值视为无关，不进入 Memory Context
_RELEVANCE_THRESHOLD = 0.05

# 各类型的相关性权重（behavior 咨询主题对医疗问题有一定价值）
_TYPE_WEIGHT = {
    "medication": 1.0,
    "health_context": 0.9,
    "behavior": 0.8,
    "preference": 0.5,
}

_INTERACTION_TERMS = (
    "相互作用",
    "一起吃",
    "一起用",
    "同服",
    "联用",
    "合用",
    "配伍",
)


def _bigrams(text: str):
    """中英通用的 2-gram 词元集合。"""
    text = (text or "").lower()  # 转小写。
    chars = [c for c in text if not c.isspace() and c.isalnum()]  # 过滤掉空格，只保留字母和数字（isalnum）
    # 使用滑动窗口生成所有相邻的两个字符组合（即 2-gram）。例如 "阿司匹林" -> {'阿司', '司匹', '匹林'}。
    return {"".join(chars[i: i + 2]) for i in range(max(0, len(chars) - 1))}


class MemoryRetriever:
    """长期记忆相关性检索器。"""

    def __init__(self, top_k: int = LONG_TERM_TOP_K):
        self.top_k = top_k

    def retrieve(
            self,
            query: str,
            memories: List[dict],
            top_k: Optional[int] = None,
            memory_type: Optional[str] = None,
    ) -> List[dict]:
        """
        对长期记忆做相关性筛选，返回 Top-K（已附带 relevance_score）。

        Parameters
        ----------
        query      : 当前用户问题
        memories   : 候选长期记忆（SQLMemoryRepository.list_memories() 的输出）
        top_k      : 返回条数上限（默认 LONG_TERM_TOP_K）
        memory_type: 只检索某类型时传入（例如只召用药记忆）
        """
        if not query or not memories:
            return []

        k = top_k or self.top_k
        scored: List[dict] = []
        for mem in memories:
            if memory_type and mem.get("memory_type") != memory_type:
                continue
            score = self._score(query, mem)
            if score >= _RELEVANCE_THRESHOLD:
                scored.append({**mem, "relevance_score": score})

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        result = scored[:k]

        logger.info(
            "MemoryRetriever：%d 条候选 → 命中 %d 条，top=%d（%s）",
            len(memories), len(scored), k,
            [m["memory_key"] for m in result],
        )
        return result

    # ── 打分 ──────────────────────────────────────────────────────────────────

    def _score(self, query: str, memory: dict) -> float:
        key = memory.get("memory_key", "")
        mem_type = memory.get("memory_type", "")
        value_text = safe_json_loads(memory.get("memory_value", ""))
        if isinstance(value_text, dict):
            value_text = " ".join(str(v) for v in value_text.values())
        else:
            value_text = str(value_text or "")

        q_bg = _bigrams(query)
        t_bg = _bigrams(f"{key} {value_text}")
        if not q_bg:
            return 0.0

        # 词元重叠度（基础分）
        overlap = len(q_bg & t_bg) / len(q_bg)

        # 关键术语（药名 / 主题 / 症状）精确命中查询 → 明显提权
        boost = 0.0
        if key and key in query:
            boost += 0.4
            if mem_type == "medication":
                boost += 0.2  # 用药记忆最相关

        # 行为记忆通常使用规范化主题 key（例如 drug_interaction），
        # 而用户问题使用自然语言。对明显的联合用药表达做主题映射，
        # 避免仅依赖英文 key 或完全相同的中文短语。
        topic = f"{key} {value_text}".lower()
        is_interaction_topic = (
            mem_type == "behavior"
            and (
                "drug_interaction" in topic
                or "药物相互作用" in topic
                or "相互作用" in topic
            )
        )
        if is_interaction_topic and any(term in query for term in _INTERACTION_TERMS):
            boost += 0.5

        score = overlap * _TYPE_WEIGHT.get(mem_type, 0.8) + boost
        return round(max(0.0, min(score, 1.0)), 4)

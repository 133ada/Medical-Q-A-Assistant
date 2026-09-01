"""
Conversation Summary

输入：旧 Summary + 需要压缩的近期消息
输出：融合后的新 Summary（LLM）

要点：
  · 新 Summary 融合"旧 Summary + 历史消息"，而不是只概括最近 20 条。
  · 提取对话主题、用户重要信息、讨论过的药物、健康相关信息、已答问题、未决问题、
    后续关注点。
  · 医疗安全：不允许把用户猜测/怀疑/普通症状描述升级为确定诊断
    （例如"我好像对青霉素过敏"只能写成"用户曾表示怀疑，未经确认"）。
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from App.LLM.llm import make_llm
from App.Repositories.MemoryMessage import MemoryMessage

logger = logging.getLogger(__name__)


def format_messages_for_summary(messages: List[MemoryMessage]) -> str:
    """把消息列表格式化为传给 LLM 的对话文本。"""
    lines = []
    for m in messages:
        speaker = {"user": "用户", "assistant": "助手", "system": "系统"}.get(m.role, m.role)
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是医疗问答系统里的对话摘要 Agent。请把"旧对话摘要 + 近期对话"融合成一份新的对话摘要。
            
            【旧对话摘要】
            {old_summary}
            
            【近期对话】
            {conversation}
            
            要求生成的新摘要必须包含：
            1. 当前对话主题
            2. 用户已提供的重要信息（尤其是健康相关信息、用药信息）
            3. 已讨论过的药物
            4. 已经回答过的重要问题
            5. 尚未解决的问题
            6. 后续对话需要关注的信息
            
            【医疗安全要求——必须严格遵守】
            - 不允许把用户的猜测、怀疑或普通症状描述自动升级成确定诊断。
              例如用户说"我好像对青霉素过敏"，必须写"用户曾表示怀疑自己可能对青霉素过敏，目前未经确认"，而不能写"用户对青霉素过敏"。
            - 用词保持客观，区分"用户自述/怀疑"与"已确认"。
            - 不要输出用户的敏感对话原文，只需提炼事实要点。
            
            只输出融合后的新摘要文本，不要任何前缀、解释或多余格式。"""
        ),
        ("human", "")
    ]
)


class ConversationSummarizer:
    """旧 Summary + 消息 → LLM → 新 Summary。"""

    def __init__(self, llm=None):
        # 延迟到首次调用时创建，避免构造阶段触发任何 API 请求
        self._llm = llm
        self._prompt = _SUMMARY_PROMPT

    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_llm(temperature=0.0)
        return self._llm

    def summarize(
            self, messages: List[MemoryMessage], old_summary: str = ""
    ) -> str:
        """
        生成融合后的新 Summary。

        Parameters
        ----------
        messages   : 需要压缩的历史消息（含短期记忆中的近期消息）
        old_summary: 数据库中已有的旧 Summary（首次可能为空）

        Returns
        -------
        str : 新 Summary 文本
        """
        conversation = format_messages_for_summary(messages)
        chain = self._prompt | self.llm | StrOutputParser()
        result = chain.invoke(
            {"old_summary": old_summary or "（暂无旧摘要）", "conversation": conversation}
        )
        new_summary = result.strip()
        logger.info(
            "Summary 生成完成：旧摘要 %d 字 + %d 条消息 → 新摘要 %d 字",
            len(old_summary or ""), len(messages), len(new_summary),
        )
        return new_summary or old_summary

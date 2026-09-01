"""
extractor/memory_extractor.py
=============================
长期记忆提取（CoverLayWindow.md 第十一～十三、十五～十六、二十、二十六节）。

流程：
    (user_question, assistant_answer, existing) → LLM → 结构化候选记忆 → 安全策略 → 候选列表

· 输出结构化 JSON 数组（而不是普通字符串），每条包含
  type / key / value / status / confidence / source / expires_in_days。
· 医疗安全策略是【确定性代码强制】，不依赖 LLM 自觉：
    - source=assistant_inference 且 status 为 confirmed/active → 强制降级 candidate
    - status=confirmed 但 confidence < 0.7 → 强制降级 candidate
    - 模型推断信息不能自动成为 confirmed medical fact（文档第二十六节）
"""

from __future__ import annotations

import logging
from typing import List, Optional, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from App.LLM import make_llm
from App.Memory.utils import extract_json_object

logger = logging.getLogger(__name__)

# 模型推断进入 confirmed 需要的最低置信度；低于此值强制降级
_CONFIRM_MIN_CONFIDENCE = 0.7
# 保留为候选记忆的最低置信度（过滤噪音）
_MIN_CONFIDENCE = 0.15

MEMORY_SOURCES = Literal["user", "assistant", "system"]
MEMORY_STATUS = Literal["confirmed", "active", "candidate"]
MEMORY_TYPES = Literal["preference", "health_context", "medication"]
_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是医疗问答系统的"记忆提取 Agent"。根据一轮对话（用户问题 + 助手回答），
            提取值得【长期保存】的用户记忆，输出一个 JSON 数组。
            
            【四类长期记忆】
            1. preference（用户偏好）—— 长期稳定的交互偏好，如：喜欢简洁回答、回答要包含用药注意事项、偏好中文。
               普通一次性表达不要保存。
            2. medication（用药记忆）—— 用户明确表达的长期或当前用药信息，如：正在服用阿司匹林、曾经服用布洛芬、已停药。
            3. health_context（健康上下文）—— 具有长期价值的健康信息，如：年龄、过敏史、既往疾病、长期健康状况。
            4. behavior（咨询行为）—— 用户长期咨询的主题（用于个性化路由/画像），如：药物相互作用、老年人用药、糖尿病、副作用。
               不要把每条聊天都转成行为记忆，只保留明显反复出现/重要的主题。
            
            【输出格式——每条记忆是 JSON 对象】
            {{
                "type": "medication",
                "key": "阿司匹林",                          // medication: 药名；behavior: 主题；其余：简短标识
                "value": {{"drug_name": "阿司匹林", "dosage": "", "frequency": ""}},  // 结构化信息，没有的字段给空串
                "status": "active",                         // candidate / confirmed / active / inactive / expired
                "confidence": 0.95,                          // 0.0 ~ 1.0
                "source": "user",                            // user / assistant_inference
                "expires_in_days": null                      // 有时效性的记忆给天数，否则 null
            }}
            
            【规则】
            - 只提取用户明确表达的信息；助手回答中的科普内容不要转成用户记忆。
            - status 规则：
                * 用户明确说正在服用 → active
                * 用户明确说已停药/已不吃 → inactive
                * 用户"好像/可能/怀疑"有某疾病或过敏 → candidate（绝不能 confirmed）
                * 只有用户明确确认（如"确诊了/医生说了"）才能 confirmed
            - source 规则：用户亲口说的 → user；基于上下文推断 → assistant_inference。
            - 助手推断出来的信息，confidence 不要高于 0.4。
            - 没有值得长期保存的信息时，输出空数组 []。
            - 只输出 JSON 数组本身，不要解释、不要 markdown 代码块。"""
        ),
        (
            "human",
            """【用户问题】
{question}

【助手回答】
{answer}

【该用户已有记忆（供去重参考，可能为空）】
{existing}

请输出提取到的长期记忆 JSON 数组："""
        ),
    ]
)


class MemoryExtractor:
    """对话 → 结构化候选记忆（已过安全策略）。"""

    def __init__(self, llm=None):
        self._llm = llm
        self._prompt = _EXTRACT_PROMPT

    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_llm(temperature=0.0)
        return self._llm

    def extract(
            self,
            user_question: str,
            assistant_answer: str,
            existing: Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        提取候选长期记忆。

        Parameters
        ----------
        user_question    : 本轮用户问题
        assistant_answer : 本轮助手回答
        existing         : 用户已有的长期记忆（供 LLM 去重参考）

        Returns
        -------
        list[dict] : 已通过安全策略的候选记忆（失败时返回空列表，不抛异常）
        """
        if not user_question and not assistant_answer:
            return []

        existing_text = ""
        if existing:
            items = [
                f"- [{m.get('memory_type')}] {m.get('memory_key')} = {m.get('memory_value')}"
                f" (status={m.get('status')}, conf={m.get('confidence')})"
                for m in existing
            ]
            existing_text = "\n".join(items) or "（无）"

        try:
            raw = (self._prompt | self.llm | StrOutputParser()).invoke(
                {
                    "question": user_question,
                    "answer": assistant_answer,
                    "existing": existing_text or "（无）",
                }
            )
            parsed = extract_json_object(raw)
            if not isinstance(parsed, list):
                logger.warning("MemoryExtractor：LLM 输出不是数组，忽略本次提取")
                return []

            candidates = [apply_memory_policy(c) for c in parsed]
            candidates = [c for c in candidates if c is not None]
            logger.info(
                "MemoryExtractor：原始 %d 条 → 通过策略 %d 条",
                len(parsed), len(candidates),
            )
            return candidates
        except Exception as exc:
            # 提取失败不影响主问答链路
            logger.warning("MemoryExtractor 执行失败：%s", exc)
            return []


def apply_memory_policy(candidate: dict) -> Optional[dict]:
    """
    对单条候选记忆执行确定性医疗安全策略。

    Returns
    -------
    dict 或 None（该条被过滤）。
    """
    if not isinstance(candidate, dict):
        return None

    mem_type = str(candidate.get("type", "")).strip().lower()
    if mem_type not in MEMORY_TYPES:
        mem_type = "preference"

    key = str(candidate.get("key", "")).strip()
    if not key:
        return None

    source = str(candidate.get("source", "user")).strip().lower()
    if source not in MEMORY_SOURCES:
        source = "user"

    status = str(candidate.get("status", "candidate")).strip().lower()
    if status not in MEMORY_STATUS:
        status = "candidate"

    try:
        confidence = float(candidate.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # ── 医疗安全强制规则 ──────────────────────────────────
    # 1. 模型推断 → 绝不能自动 confirmed/active
    if source == "assistant_inference" and status in ("confirmed", "active"):
        status = "candidate"
        confidence = min(confidence, 0.4)
    # 2. confirmed 需要足够高的置信度
    if status == "confirmed" and confidence < _CONFIRM_MIN_CONFIDENCE:
        status = "candidate"

    # 过滤噪音
    if confidence < _MIN_CONFIDENCE:
        return None

    value = candidate.get("value")
    if not isinstance(value, dict):
        # 允许 value 为简单字符串，统一包装为 dict
        value = {"detail": str(value or "")}

    # medication 的 key 统一为药名（避免"阿司匹林"和"阿司匹林片"分裂成两条）
    if mem_type == "medication":
        value["drug_name"] = key

    normalized = {
        "type": mem_type,
        "key": key,
        "value": value,
        "status": status,
        "confidence": round(confidence, 4),
        "source": source,
        "expires_in_days": candidate.get("expires_in_days"),
    }
    return normalized

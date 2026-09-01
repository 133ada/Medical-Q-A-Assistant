"""
Medical_agent_graph.py
======================
执行层 —— 多智能体（Multi-Agent）节点逻辑 + LangGraph 图构建。

架构（专业分工 + 协同决策，每个职责单一）：

                              ┌─────────────┐
                     START ─▶ │  supervisor │ 主控 Agent：身份/意图识别
                              └──────┬──────┘
                    ┌────────────────┼──────────────────┐
                 MEMORY          RETRIEVE            ANSWER（纯寒暄/无关，直答）
                    ▼                 ▼                  ▼
        ┌─────────────────┐   ┌───────────────┐   ┌──────────────┐
        │ memory_retriever│   │ retriever_drug│   │    answer    │  回答生成 Agent
        │  查询用户历史信息 │ ─▶ │  药物基础信息   │   └─────┬───────┘
        └────────┬────────┘   └───────┬───────┘          ▲     ▼
                                      ▼                  │   answer → END
                             ┌───────────────────┐       │
                             │   retriever_ddi   │       │
                             │  药物相互作用 Agent │ ──────┘
                             └────────┬──────────┘

要点：
  · supervisor 用 LLM 做「身份 + 意图」判断，三路路由：
      - MEMORY   → 请求带 id 或问题文本明确带身份信息 → 先走 memory_retriever 查用户历史，再检索
      - RETRIEVE → 无身份信息、需要检索药品库 → 直接检索
      - ANSWER   → 纯寒暄/无关 → 直接回答
  · 药物知识现已拆分为两个独立向量库：
      - drug 库（vector_store_other.py 产出）：药物基础信息（描述/适应症/药理/毒性等）
      - ddi  库（vector_store_ddi.py 产出）  ：Drug Interactions 药物相互作用信息
    因此检索层拆成两个节点：retriever_drug → retriever_ddi，依次执行、共同产出摘要，
    再统一交给 answer 生成最终回答（路由：supervisor →(MEMORY)→ memory_retriever →
    retriever_drug → retriever_ddi → answer；supervisor →(RETRIEVE)→ retriever_drug →
    retriever_ddi → answer；supervisor →(ANSWER)→ answer）。
  · retriever_* 检索后会把原文「压缩」成结构化 JSON 摘要（大上下文进、小结果出），
    answer 只消费摘要、不再读原文 —— 真正实现上下文隔离；
    检索到的原文片段仍会打印到控制台，便于调试。
  · 后续新增症状库/报告库等，只需在 retriever_ddi 之后串联新检索节点即可。

记忆集成（App/Memory 模块）：
  · 记忆检索已移入图内：supervisor 根据「是否输入 id / 问题是否明确带身份信息」
    条件路由到 memory_retriever 节点，复用 MemoryManager.build_memory_context()
    组装记忆上下文（近期对话 / 历史摘要 / 相关长期记忆）注入 State，answer 节点消费。
  · 请求不带 id 且文本无身份信息时，跳过记忆节点直接文档检索；memory_retriever
    查不到信息 = 空上下文，天然等同「直接检索文档」。
  · 每轮对话结束后的 Memory 更新（存消息 / Summary / 提取长期记忆）由
    FastAPI 后台任务调用 MemoryManager.update_after_turn() 完成，不在图中。

若要修改 Prompt 措辞、增减 Skill、调整路由 ——
  → 修改 Skills/medical_skills.py，本文件无需改动。
"""

from __future__ import annotations
import json
import logging
import re
from operator import add
from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from sentence_transformers import CrossEncoder

from App.LLM import make_llm
from App.Skills import REGISTRY, RERANKER_CONFIG, skill
from App.Memory import format_memory_context

logger = logging.getLogger(__name__)

# 向量检索的候选文档数（精排前），精排后由 RERANKER_CONFIG["top_n"] 截取
RETRIEVAL_K = 6


# =============================================================================
# 0. DocumentReranker —— CrossEncoder 二次精排工具类
#
#  职责：接收向量检索的候选文档列表，使用 CrossEncoder 对每个
#        (question, passage) 对打分，按分数降序排列后截取 Top-N 返回。
#
#  配置入口：Skills/medical_skills.py → RERANKER_CONFIG
#    · model_name : HuggingFace CrossEncoder 模型标识符
#    · top_n      : 精排后保留的文档数
#
# =============================================================================

class DocumentReranker:
    """使用本地 CrossEncoder 对候选文档列表进行二次打分排序。"""

    def __init__(self, model_name: str, top_n: int):
        self.model_name = model_name
        self.top_n = top_n
        self._encoder = None  # 懒加载：首次调用 rerank() 时才初始化

    def _load_encoder(self):
        if self._encoder is None:
            logger.info("[Reranker] 加载 CrossEncoder 模型：%s", self.model_name)
            self._encoder = CrossEncoder(self.model_name)
        return self._encoder

    def rerank(self, question: str, docs: List[Document]) -> List[Document]:
        """对候选文档列表进行 CrossEncoder 打分并排序，截取 top_n。"""
        if not docs:
            return docs

        encoder = self._load_encoder()
        pairs = [(question, doc.page_content) for doc in docs]
        scores: List[float] = encoder.predict(pairs).tolist()

        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_n = min(self.top_n, len(scored_docs))

        result: List[Document] = []
        for score, doc in scored_docs[:top_n]:
            doc.metadata["rerank_score"] = round(score, 4)
            result.append(doc)

        logger.info(
            "[Reranker] 精排完成：%d → %d 篇，分数 %s",
            len(docs), len(result),
            [f"{s:.4f}" for s, _ in scored_docs[:top_n]]
        )
        return result


# =============================================================================
# 1. 共享 State 定义
#    —— 各 Agent 通过它传递问题、检索结果，实现协同决策。
# =============================================================================

class MedicalState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    question: str
    user_id: str | None  # 请求显式携带的用户 id；未提供为 None（supervisor 据此判断是否查记忆）
    conversation_id: str | None  # 会话 id（memory_retriever 查询历史需要）
    memory_allowed: bool  # 只有服务端认证成功的用户才允许访问 Memory
    next: str  # supervisor 决定下一跳：memory_retriever / retriever_drug / answer
    drug_summary: str  # retriever_drug 抽取出的结构化摘要（JSON 字符串）
    ddi_summary: str  # retriever_ddi 抽取出的结构化摘要（JSON 字符串）
    final_answer: str  # answer 的最终答案
    memory_context: dict  # memory_retriever 注入的上下文（近期对话/历史摘要/长期记忆）


# =============================================================================
# 3. 检索引擎
#    查询增强 → 向量检索 → Reranker 精排 → 拼接 context
#    每次检索都会把命中与精排后的片段打印到控制台，便于调试。
# =============================================================================

class RetrievalEngine:
    """封装一次完整的检索流程，供检索 Agent 使用。"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.reranker = DocumentReranker(
            model_name=RERANKER_CONFIG["model_name"],
            top_n=RERANKER_CONFIG["top_n"]
        )

    def retrieve(self, question: str, enhance_prompt, label: str = "检索") -> str:
        """
        执行检索并返回拼接好的 context 字符串。

        Parameters
        ----------
        question      : str  用户原始问题
        enhance_prompt: ChatPromptTemplate  查询增强模板
        label         : str  检索来源标签（用于控制台打印）
        """
        llm = make_llm()

        # Step 1 · 查询增强
        enhanced_query = (
                enhance_prompt | llm | StrOutputParser()
        ).invoke({"question": question})
        print(f"\n[🔍 {label}] 增强后查询 → 「{enhanced_query.strip()}」")

        # Step 2 · 向量检索
        docs: List[Document] = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K},
        ).invoke(enhanced_query)
        print(f"[🔍 {label}] 向量检索：命中 {len(docs)} 篇候选文档")

        # Step 3 · Reranker 精排
        if len(docs) > 1:
            try:
                docs = self.reranker.rerank(question, docs)
            except Exception as exc:
                logger.warning("[RetrievalEngine] Reranker 执行失败，回退原始顺序：%s", exc)

        # Step 4 · 控制台打印检索到的片段
        self._print_docs(docs, label)

        return self._build_context(docs)

    @staticmethod
    def _print_docs(docs: List[Document], label: str) -> None:
        """把精排后保留的每个文档片段打印到控制台。"""
        if not docs:
            print(f"[🔍 {label}] ⚠️ 未检索到相关文档片段")
            return

        print(f"[🔍 {label}] 精排后保留 {len(docs)} 个片段：")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("drug_name", doc.metadata.get("source", "?"))
            score = doc.metadata.get("rerank_score")
            score_tag = f"  [分数 {score}]" if score is not None else ""
            print(f"\n  ── 片段 {i} · 来源: {source}{score_tag} ──")
            print(doc.page_content.strip())

    @staticmethod
    def _build_context(docs: List[Document]) -> str:
        """将文档列表格式化为 context 字符串，附带来源与精排分数。"""
        if not docs:
            return "（未检索到相关文档）"

        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("drug_name", doc.metadata.get("source", "?"))
            score_tag = (
                f"  [精排分数: {doc.metadata['rerank_score']}]"
                if "rerank_score" in doc.metadata
                else ""
            )
            parts.append(
                f"[文档 {i} · 来源: {source}]{score_tag}\n"
                + doc.page_content.strip()
            )
        return "\n\n".join(parts)


# =============================================================================
# 4. 节点实现
#    每个节点通过 @skill(REGISTRY["..."]) 绑定元数据；
#    Prompt 统一通过 meta.get_prompt("key") 获取，节点内不硬编码模板。
# =============================================================================

# ── 4-A  主控 Agent（Supervisor）──────────────────────────────────────────────

class SupervisorNode:
    """
    主控 Agent：身份 + 意图识别。

    调用 LLM 判断下一步流向（三路路由）：
      - MEMORY   → 下一跳 memory_retriever（请求带 id 或文本明确带身份信息 → 先查用户历史）
      - RETRIEVE → 下一跳 retriever_drug（无身份信息，需要检索药品知识库）
      - ANSWER   → 下一跳 answer（纯寒暄/无关，直接回答）

    memory_enabled=False 时系统未注册记忆节点，即使误判 MEMORY 也降级到文档检索。
    """

    def __init__(self, memory_enabled: bool = True):
        self.memory_enabled = memory_enabled
        self.meta = REGISTRY["supervisor"]

    @skill(REGISTRY["supervisor"])
    def __call__(self, state: MedicalState):
        llm = make_llm(temperature=0.0)
        question = state["question"]
        has_explicit_id = "是" if state.get("user_id") else "否"
        memory_available = (
            "是"
            if self.memory_enabled and state.get("memory_allowed", False)
            else "否"
        )

        raw = (
                self.meta.get_prompt("route") | llm | StrOutputParser()
        ).invoke({
            "question": question,
            "has_explicit_id": has_explicit_id,
            "memory_available": memory_available,
        })

        if "MEMORY" in raw.upper():
            next_node = "memory_retriever"
        elif "ANSWER" in raw.upper():
            next_node = "answer"
        else:
            next_node = "retriever_drug"

        # 防御：系统未启用记忆检索时，即使误判 MEMORY 也直接进入文档检索
        if (
            next_node == "memory_retriever"
            and not self.memory_enabled
        ) or (
            next_node == "memory_retriever"
            and not state.get("memory_allowed", False)
        ):
            next_node = "retriever_drug"

        self.meta.log_step(f"意图识别结果 → 「{raw.strip()}」")
        self.meta.log_step(f"下一跳 → {next_node}")

        return {"next": next_node}


# ── 4-A2  记忆检索 Agent（Memory Retriever）────────────────────────────────────

class MemoryRetrieverNode:
    """
    记忆检索 Agent：查询用户历史信息。

    复用 MemoryManager.build_memory_context()（memory_manager.py），一次组装：
      短期记忆（近期对话） + 历史摘要（Conversation Summary） + 相关长期记忆（相关性检索），
    结果写入 State["memory_context"]，供下游 answer 节点消费。

    · 节点内不做二次路由：查询完成一律进入 retriever_drug（文档检索）。
      没检索到任何信息 = 空上下文，天然等同"直接检索文档"。
    · build_memory_context 内部已全容错（Memory 不是单点故障，文档第二十八节）。
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.meta = REGISTRY["memory_retriever"]

    @skill(REGISTRY["memory_retriever"])
    def __call__(self, state: MedicalState):
        question = state["question"]
        user_id = state.get("user_id")
        conversation_id = state.get("conversation_id")

        if not user_id or not state.get("memory_allowed", False):
            self.meta.log_step("未通过用户身份认证，跳过用户历史记忆")
            return {"memory_context": {}}

        self.meta.log_step(
            f"查询用户历史信息（user_id={user_id}, conversation_id={conversation_id}）…"
        )
        context = self.memory_manager.build_memory_context(
            user_id, conversation_id, question
        )

        recent_count = len(context.get("recent_messages") or [])
        recalled = context.get("retrieved_count", 0)
        self.meta.log_step(
            f"记忆上下文就绪：近期对话 {recent_count} 条，召回长期记忆 {recalled} 条"
        )
        return {"memory_context": context}


# ── 4-B  检索 Agent（药物基础信息）─────────────────────────────────────────────

class RetrievalDrugNode:
    """
    检索 Agent：药物基础信息（描述/适应症/药理/毒性/用法等，来自 vector_store_other.py 库）。

    查询增强 → 向量检索 → Reranker 精排 → 抽取结构化摘要。

    关键设计（上下文隔离）：
      检索命中的原文（大上下文）只在本节点内消化，通过 extract 抽取成
      结构化 JSON 摘要（小结果）写入 State["drug_summary"]；下游 answer
      只消费摘要，不再读原文。
    """

    def __init__(self, engine: RetrievalEngine):
        self.engine = engine
        self.meta = REGISTRY["retriever_drug"]

    @skill(REGISTRY["retriever_drug"])
    def __call__(self, state: MedicalState):
        llm = make_llm(temperature=0.0)
        question = state["question"]

        # Step 1 · 检索：命中原文（供抽取），并打印片段到控制台
        self.meta.log_step("检索药物基础信息知识库中…")
        raw_context = self.engine.retrieve(
            question, self.meta.get_prompt("enhance"), label="药物基础信息检索"
        )

        # Step 2 · 压缩：原文 → 结构化 JSON 摘要（大上下文进、小结果出）
        self.meta.log_step(f"抽取结构化摘要（原文 {len(raw_context)} 字符）…")
        summary = self._extract(llm, question, raw_context)

        self.meta.log_step(f"摘要就绪（{len(summary)} 字符）")
        return {"drug_summary": summary}

    def _extract(self, llm, question: str, raw_context: str) -> str:
        raw = (
                self.meta.get_prompt("extract") | llm | StrOutputParser()
        ).invoke({"question": question, "context": raw_context})
        return _parse_json(raw, raw_context)


# ── 4-C  检索 Agent（药物相互作用 DDI）─────────────────────────────────────────

class RetrievalDDINode:
    """
    检索 Agent：药物相互作用信息（Drug Interactions，来自 vector_store_ddi.py 库）。

    在 retriever_drug 之后执行，与其并列为「检索层」的第二跳，
    产出的摘要单独写入 State["ddi_summary"]，与 drug_summary 互不覆盖，
    下游 answer 会同时消费两份摘要。
    """

    def __init__(self, engine: RetrievalEngine):
        self.engine = engine
        self.meta = REGISTRY["retriever_ddi"]

    @skill(REGISTRY["retriever_ddi"])
    def __call__(self, state: MedicalState):
        llm = make_llm(temperature=0.0)
        question = state["question"]

        # Step 1 · 检索：命中原文（供抽取），并打印片段到控制台
        self.meta.log_step("检索药物相互作用（DDI）知识库中…")
        raw_context = self.engine.retrieve(
            question, self.meta.get_prompt("enhance"), label="药物相互作用检索"
        )

        # Step 2 · 压缩：原文 → 结构化 JSON 摘要（大上下文进、小结果出）
        self.meta.log_step(f"抽取结构化摘要（原文 {len(raw_context)} 字符）…")
        summary = self._extract(llm, question, raw_context)

        self.meta.log_step(f"摘要就绪（{len(summary)} 字符）")
        return {"ddi_summary": summary}

    def _extract(self, llm, question: str, raw_context: str) -> str:
        raw = (
                self.meta.get_prompt("extract") | llm | StrOutputParser()
        ).invoke({"question": question, "context": raw_context})
        return _parse_json(raw, raw_context)


def _parse_json(raw: str, fallback: str) -> str:
    """把 LLM 输出解析为 JSON 字符串；解析失败时降级为原文截断，保证主流程不断。"""
    text = raw.strip()

    # 1) 直接解析
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) 剥离 ```json ... ``` / ``` ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        candidate = m.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, TypeError):
            pass

    # 3) 提取第一个 {...} 块
    m = re.search(r"\{[\s\S]*}", text)
    if m:
        candidate = m.group(0)
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, TypeError):
            pass

    # 4) 降级：返回原文截断
    return fallback[:2000]


# ── 4-D  回答生成 Agent（Answer）───────────────────────────────────────────────

class AnswerNode:
    """
    回答生成 Agent：基于检索结果（药物基础信息 + 药物相互作用）+ 风险判断 + 免责声明，
    输出最终答案。这是图的最后一跳（无论是否经过检索，都会走到这里）。
    """

    def __init__(self):
        self.meta = REGISTRY["answer"]

    @skill(REGISTRY["answer"])
    def __call__(self, state: MedicalState):
        llm = make_llm(temperature=0.0)
        question = state["question"]
        # 从共享 State 读取两个检索节点写入的摘要
        drug_summary = state.get("drug_summary", "")
        ddi_summary = state.get("ddi_summary", "")
        # Memory 上下文（近期对话/历史摘要/相关长期记忆）；无记忆时为空串
        memory_context = state.get("memory_context") or {}
        memory_text = format_memory_context(memory_context) if memory_context else "（无）"

        self.meta.log_step(
            f"生成最终回答（药物摘要 {len(drug_summary)} 字符，DDI 摘要 {len(ddi_summary)} 字符，"
            f"记忆上下文 {len(memory_text)} 字符）…"
        )
        final_answer = (
                self.meta.get_prompt("answer") | llm | StrOutputParser()
        ).invoke({
            "question": question,
            "drug_summary": drug_summary,
            "ddi_summary": ddi_summary,
            "memory_context": memory_text,
        })

        self.meta.log_step(f"最终答案就绪（{len(final_answer)} 字符）")
        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }


# =============================================================================
# 5. 构建 LangGraph 图
# =============================================================================

def _route_supervisor(state: MedicalState) -> str:
    """读取 supervisor 写入的 state['next']，返回下一跳节点。"""
    return state.get("next", "retriever_drug")


def build_medical_graph(drug_vectorstore, ddi_vectorstore, memory_manager=None):
    """
    构建医疗多智能体图。

    Parameters
    ----------
    drug_vectorstore : Chroma  药物基础信息向量库（vector_store_other.py 产出）
    ddi_vectorstore  : Chroma  药物相互作用向量库（vector_store_ddi.py 产出）
    memory_manager   : MemoryManager | None  传入时注册 memory_retriever 节点，
                       由 supervisor 根据是否携带 id/身份信息条件路由查询用户历史。
    """

    builder = StateGraph(MedicalState)

    drug_engine = RetrievalEngine(drug_vectorstore)
    ddi_engine = RetrievalEngine(ddi_vectorstore)

    builder.add_node("supervisor", SupervisorNode(memory_enabled=memory_manager is not None))
    builder.add_node("retriever_drug", RetrievalDrugNode(drug_engine))
    builder.add_node("retriever_ddi", RetrievalDDINode(ddi_engine))
    builder.add_node("answer", AnswerNode())

    # 记忆检索节点（仅在传入 memory_manager 时启用）
    if memory_manager is not None:
        builder.add_node("memory_retriever", MemoryRetrieverNode(memory_manager))
        # 记忆查询完成后无条件进入文档检索（没检索到 = 空上下文，等同直接检索）
        builder.add_edge("memory_retriever", "retriever_drug")

    # 入口 → 主控 Agent
    builder.add_edge(START, "supervisor")

    # 主控 Agent 条件路由：
    #   MEMORY   → memory_retriever（先查用户历史）→ retriever_drug
    #   RETRIEVE → retriever_drug（直接检索）
    #   ANSWER   → 直接回答
    builder.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "memory_retriever": "memory_retriever" if memory_manager is not None else "retriever_drug",
            "retriever_drug": "retriever_drug",
            "answer": "answer",
        },
    )

    # 检索层内部串联：先查药物基础信息，再查药物相互作用
    builder.add_edge("retriever_drug", "retriever_ddi")

    # 检索完成后进入回答生成
    builder.add_edge("retriever_ddi", "answer")

    builder.add_edge("answer", END)

    return builder.compile()


# =============================================================================
# 6. 便捷调用函数
# =============================================================================

def build_initial_state(
        question: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
) -> MedicalState:
    return MedicalState(
        messages=[HumanMessage(content=question)],
        question=question,
        user_id=user_id,
        conversation_id=conversation_id,
        memory_allowed=user_id is not None,
        next="retriever_drug",
        drug_summary="",
        ddi_summary="",
        final_answer="",
        memory_context={},
    )


def run_medical_agent(
        graph,
        question: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
) -> str:
    """
    同步调用，返回最终答案字符串。

    记忆检索已移入图内（memory_retriever 节点，由 supervisor 条件路由）：
    请求带 id 或问题文本明确带身份信息时，自动查询该用户的
    短期记忆 + 历史摘要 + 相关长期记忆，实现带记忆的连续问答。
    """
    final_state = graph.invoke(
        build_initial_state(question, user_id=user_id, conversation_id=conversation_id)
    )
    answer = final_state.get("final_answer", "").strip()
    if answer:
        return answer
    # 兜底：从 messages 里取最后一条 AI 消息
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content.strip():
            return msg.content
    return "抱歉，暂时无法处理您的问题。"


def stream_medical_agent(
        graph,
        question: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
):
    """
    流式调用，按节点完成顺序 yield 各 Agent 的进度信息与最终答案。

    格式说明：
      - [🧭 主控] / [🧠 记忆] / [🔍 检索] = 中间进度
      - 最后一段无前缀文字 = answer 输出的最终答案

    记忆检索由图内 memory_retriever 节点负责（supervisor 条件路由）。
    """
    try:
        for chunk in graph.stream(
                build_initial_state(question, user_id=user_id, conversation_id=conversation_id),
                stream_mode="updates",
        ):
            for node_name, updates in chunk.items():
                if node_name == "supervisor":
                    next_node = updates.get("next", "")
                    desc = {
                        "memory_retriever": "先查询用户历史记忆",
                        "retriever_drug": "直接检索药品知识库",
                        "answer": "直接回答",
                    }.get(next_node, "检索")
                    yield f"\n[🧭 主控] 决策：{desc}\n"
                elif node_name == "memory_retriever":
                    yield "\n[🧠 记忆] 用户历史信息查询完成\n"
                elif node_name == "retriever_drug":
                    yield "\n[🔍 检索] 药物基础信息完成\n"
                elif node_name == "retriever_ddi":
                    yield "\n[🔍 检索] 药物相互作用完成\n"
                elif node_name == "answer":
                    yield updates.get("final_answer", "")
    except StopIteration:
        # 屏蔽 StopIteration 泄漏到 asyncio Future 的问题
        return

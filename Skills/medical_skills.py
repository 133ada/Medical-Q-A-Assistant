"""
Skills/medical_skills.py
========================
配置层 —— 唯一一处声明"这个多智能体系统有哪些 Agent、每个 Agent 长什么样"的文件。

职责：
  1. 实例化全部 SkillMeta，每个实例的 prompts 字典内嵌对应的 ChatPromptTemplate。
  2. 向全局 REGISTRY 注册，并将 REGISTRY 导出供 Medical_agent_graph.py 使用。

多智能体分工（专业分工 + 协同决策，每个职责单一）：

  🧭 supervisor       主控 Agent            —— 身份/意图识别：判断是否先查用户记忆（MEMORY）、是否需要检索药品库（RETRIEVE）、或直接回答（ANSWER）
  🧠 memory_retriever 记忆检索 Agent        —— 查询用户历史信息（短期记忆 + 历史摘要 + 相关长期记忆），复用 MemoryManager.build_memory_context()
  🔍 retriever_drug   检索 Agent（药物基础信息）—— 查询增强 → 向量检索 → Reranker 精排，产出药物基础信息摘要
  🧪 retriever_ddi    检索 Agent（药物相互作用）—— 查询增强 → 向量检索 → Reranker 精排，产出 DDI 摘要
  🧬 answer           回答生成 Agent         —— 基于两份检索摘要 + 风险判断 + 免责声明，输出最终答案

说明：药物知识库现已拆分为两个独立向量库（drug 基础信息库 / ddi 相互作用库），
     因此检索层拆成两个节点：retriever_drug → retriever_ddi，依次执行、
     各自产出结构化摘要，再统一交给 answer 生成最终回答。
     后续若有症状库/检查报告库等，只需在 retriever_ddi 之后新增对应检索节点即可。

Medical_agent_graph.py 只需::

    from App.Skills.medical_skills import REGISTRY
    from App.Skills.registry import skill

若需要替换某个 Prompt，只改这里，节点逻辑完全不动。
"""

from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from .registry import SkillMeta, SkillRegistry, skill  # re-export skill 供外层直接用

# =============================================================================
# 全局注册表（单例）
# =============================================================================

REGISTRY = SkillRegistry()

# =============================================================================
# 1. supervisor —— 主控 Agent
#    判断用户问题是否需要检索药品知识库：RETRIEVE（检索）或 ANSWER（直接回答）。
# =============================================================================

REGISTRY.register(SkillMeta(
    name="supervisor",
    icon="🧭",
    description="主控 Agent：身份/意图识别，判断是否先查用户记忆、是否需要检索药品知识库",
    version="4.0.0",
    color_key="purple",
    tags=["router", "intent", "llm", "Memory"],
    prompts={
        "route": ChatPromptTemplate.from_messages([
            (
                "system",
                """你是医疗问答系统的"主控 Agent"，负责决定下一步流向。
                
                输入信息：
                - 请求是否显式携带了用户 id：{has_explicit_id}
                - 系统是否支持用户记忆检索：{memory_available}
                
                判断规则（按优先级）：
                1. 满足以下任一 → 输出 MEMORY（需要先查询用户历史信息）：
                   · 请求显式携带用户 id（{has_explicit_id} = 是）
                   · 问题文本明确带有身份信息（如自称身份、提到自己的用药/病史/病历、提到"我之前/上次"等个人历史）
                2. 否则，若问题涉及药物、药品、用药、疾病、症状、医学健康知识 → 输出 RETRIEVE
                3. 否则（纯寒暄、问候、闲聊或与医疗完全无关）→ 输出 ANSWER
                
                注意：
                - 若问题仅是纯寒暄（如"你好""谢谢"）且与医疗完全无关，即使携带了 id 也输出 ANSWER，避免无谓检索
                - 若 {memory_available} = 否，不要输出 MEMORY
                - 只输出 MEMORY / RETRIEVE / ANSWER 之一，不要输出任何解释、标点或多余文字。"""
            ),
            ("human", "{question}")
        ])
    }
))

# =============================================================================
#  memory_retriever —— 记忆检索 Agent
#     查询用户历史信息（短期记忆 / 历史摘要 / 相关长期记忆）。
#     复用 MemoryManager.build_memory_context()（memory_manager.py），
#     不调用独立 LLM，因此 prompts 为空字典。
# =============================================================================

REGISTRY.register(SkillMeta(
    name="memory_retriever",
    icon="🧠",
    description="记忆检索 Agent：查询用户历史信息，注入记忆上下文",
    version="1.0.0",
    color_key="teal",
    tags=["Memory", "retrieval", "user-history"],
    prompts={},
))

# =============================================================================
# 2A. retriever_drug —— 检索 Agent（药物基础信息）
#     查询增强（enhance）→ 向量检索 drug 库 → Reranker 精排，产出结构化摘要。
# =============================================================================

REGISTRY.register(SkillMeta(
    name="retriever_drug",
    icon="🔍",
    description="检索 Agent（药物基础信息）：检索 + 抽取结构化摘要，实现上下文隔离",
    version="1.0.0",
    color_key="blue",
    tags=["Rag", "retrieval", "drugbank", "drug-info"],
    prompts={
        # ── 查询增强：把用户问题改写成更利于药品基础信息检索的查询 ───────────
        "enhance": ChatPromptTemplate.from_messages([
            (
                "system",
                "请将用户的医疗/药物问题改写为更适合在「药品基础信息知识库」中检索的查询语句。"
                "重点突出：药品名称、适应症、用法用量、禁忌、不良反应、药理作用、疾病/症状等关键词。"
                "不需要突出药物相互作用相关内容。只输出改写后的查询，不要解释。"
            ),
            ("human", "{question}")
        ]),

        # ── 抽取压缩：把检索原文抽取成结构化 JSON 摘要（大上下文进、小结果出）──
        "extract": ChatPromptTemplate.from_messages([
            (
                "system",
                """你是药物基础信息抽取 Agent。请根据下方检索到的药品资料，抽取与用户问题最相关的关键信息，输出一个 JSON 对象。

                输出 JSON 对象，字段如下：
                - "drug": 涉及的药品名称（字符串，多个用顿号连接）
                - "indications": 适应症/用途（字符串数组）
                - "dosage": 用法用量（字符串，资料无则空串）
                - "contraindications": 禁忌症/慎用人群（字符串数组）
                - "adverse_reactions": 不良反应（字符串数组）
                - "notes": 其他重要信息（字符串，无则空串）
                - "sources": 信息来源（字符串数组，取每段资料"[文档 N · 来源: xxx]"里的 xxx）

                要求：
                1. 严格基于资料抽取，不得编造；资料没有的字段填空数组 [] 或空串 ""
                2. 只输出 JSON 对象本身，不要输出任何解释、也不要 markdown 代码块标记
                3. 输出必须能被 json.loads 解析

                【检索到的药品资料】
                {context}

                【用户问题】
                {question}"""
            )
        ])
    }
))

# =============================================================================
# 2B. retriever_ddi —— 检索 Agent（药物相互作用 / DDI）
#     查询增强（enhance）→ 向量检索 ddi 库 → Reranker 精排，产出结构化摘要。
# =============================================================================

REGISTRY.register(SkillMeta(
    name="retriever_ddi",
    icon="🧪",
    description="检索 Agent（药物相互作用）：检索 + 抽取结构化摘要，实现上下文隔离",
    version="1.0.0",
    color_key="orange",
    tags=["Rag", "retrieval", "drugbank", "ddi"],
    prompts={
        # ── 查询增强：把用户问题改写成更利于 DDI 检索的查询 ──────────────────
        "enhance": ChatPromptTemplate.from_messages([
            (
                "system",
                "请将用户的医疗/药物问题改写为更适合在「药物相互作用（DDI）知识库」中检索的查询语句。"
                "重点突出：涉及的药品名称、联合用药、相互作用、配伍禁忌等关键词。"
                "只输出改写后的查询，不要解释。"
            ),
            ("human", "{question}")
        ]),

        # ── 抽取压缩：把检索原文抽取成结构化 JSON 摘要（大上下文进、小结果出）──
        "extract": ChatPromptTemplate.from_messages([
            (
                "system",
                """你是药物相互作用（DDI）信息抽取 Agent。请根据下方检索到的药物相互作用资料，抽取与用户问题最相关的关键信息，输出一个 JSON 对象。

                输出 JSON 对象，字段如下：
                - "drug": 涉及的药品名称（字符串，多个用顿号连接）
                - "interactions": 药物相互作用（字符串数组，每条注明相互作用的药物及具体后果/机制）
                - "severity_notes": 严重程度/风险提示（字符串数组，资料无则空数组）
                - "notes": 其他重要信息（字符串，无则空串）
                - "sources": 信息来源（字符串数组，取每段资料"[文档 N · 来源: xxx]"里的 xxx）

                要求：
                1. 严格基于资料抽取，不得编造；资料没有的字段填空数组 [] 或空串 ""
                2. 只输出 JSON 对象本身，不要输出任何解释、也不要 markdown 代码块标记
                3. 输出必须能被 json.loads 解析

                【检索到的药物相互作用资料】
                {context}

                【用户问题】
                {question}"""
            )
        ])
    }
))

# =============================================================================
# 3. answer —— 回答生成 Agent
#    统一"药物 + 医学"视角，基于检索结果生成最终答案（含风险判断与免责声明）。
# =============================================================================

REGISTRY.register(SkillMeta(
    name="answer",
    icon="🧬",
    description="回答生成 Agent：基于检索结果 + 风险判断 + 免责声明，输出最终答案",
    version="3.0.0",
    color_key="green",
    tags=["generation", "final-output", "safety"],
    prompts={
        "answer": ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一名专业的医疗问答助手，具备药理学与临床医学背景。

        请基于下方两个独立的知识检索结果回答用户问题：
        1. 药品基础信息摘要
        2. 药物相互作用（DDI）摘要

        同时参考下方【用户记忆上下文】（近期对话 / 历史摘要 / 该用户的长期记忆），
        用于保持对话连续性、避免重复询问用户已经提供过的信息。

        【用户记忆上下文】
        {memory_context}

        记忆使用规则：
        - 记忆仅作参考，必须与本次问题结合判断，不得把记忆当作当前事实无条件采信
        - 若记忆与用户当前表述冲突，以用户当前表述为准
        - 若记忆中没有相关信息，忽略即可，不要编造

        必须严格基于提供的摘要回答，不得编造摘要之外的医学事实。

        【药品基础信息摘要（JSON）】
        {drug_summary}

        【药物相互作用（DDI）摘要（JSON）】
        {ddi_summary}

        回答要求：
        1. 先给出核心结论（一句话）
        2. 再分点展开回答
        3. 如果问题涉及药物的适应症、用法用量、禁忌、不良反应等，应优先参考药品基础信息摘要
        4. 如果问题涉及联合用药、药物相互作用、配伍禁忌等，应优先参考 DDI 摘要
        5. 如果两个摘要中存在相关信息，应综合两个摘要进行回答
        6. 若问题涉及“能否用药/如何用药”，务必包含风险提示，并强调“请遵医嘱”
        7. 若摘要信息不足，不得自行补充，应明确告知用户信息不足，并建议咨询医生或药剂师
        8. 结尾追加免责声明（保持原文，不要改动）：
        ---
        ⚠️ 免责声明：本回答仅供医学知识参考，不构成诊断或治疗建议。具体用药方案请务必咨询执业医师或药剂师。
        ---
        9. 使用专业但通俗的语言，直接输出最终回答
        
        若两个摘要均为空（例如纯寒暄、与医疗无关的闲聊），请以专业医疗助手的身份友好回应用户；涉及健康建议时追加上述免责声明。"""
            ),
            ("human", "{question}")
        ])
    }
))

# =============================================================================
# Reranker 配置（归属于检索阶段）
#
#  使用 CrossEncoder 对向量检索候选文档进行二次打分排序。
#  CrossEncoder 直接接受 (query, passage) 对，无需额外 Prompt 模板。
#
#  字段说明：
#    model_name —— HuggingFace CrossEncoder 模型名称。
#                  推荐备选（支持中文）：
#                    · "BAAI/bge-reranker-v2-m3"          多语言，效果最佳
#                    · "cross-encoder/ms-marco-MiniLM-L-6-v2"  英文，速度最快
#    top_n      —— 精排后保留的文档数（建议 ≤ 向量检索 k 值）
# =============================================================================

RERANKER_CONFIG: dict = {
    "model_name": r"E:\models\bge-reranker-base",
    "top_n": 4,
}




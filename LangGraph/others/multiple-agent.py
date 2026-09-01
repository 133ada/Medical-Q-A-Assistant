from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from LangChain.app.config import config

# 初始化 LLM
llm = ChatOpenAI(
    model=config.CONFIG_DEEPSEEK["model"],
    openai_api_key=config.CONFIG_DEEPSEEK["api_key"],
    openai_api_base=config.CONFIG_DEEPSEEK["base_url"],
    temperature=0.7
)

# 定义专家 Agent
def research_agent(state: MessagesState) -> dict:
    """研究 Agent：负责信息收集"""
    system = SystemMessage(content="你是一个专业的研究员，负责收集和整理信息。请简洁地总结关键信息。")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def writing_agent(state: MessagesState) -> dict:
    """写作 Agent：负责内容创作"""
    system = SystemMessage(content="你是一个专业的写作者，负责根据已有信息撰写内容。请保持内容清晰流畅。")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def review_agent(state: MessagesState) -> dict:
    """审校 Agent：负责质量控制"""
    system = SystemMessage(content="你是一个专业的编辑，负责审核和改进内容质量。请指出问题并给出改进建议。")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


# 主管 Agent 决定流程
def supervisor_node(state: MessagesState) -> dict:
    """主管：协调各专家 Agent 的工作"""
    system = SystemMessage(content="""你是一个工作流主管。
    根据任务进度决定下一步应该由哪个 Agent 处理。
    分析对话历史，只返回以下之一：RESEARCH、WRITING、REVIEW、FINISH
    - RESEARCH：需要收集更多信息
    - WRITING：信息充足，可以开始写作
    - REVIEW：写作完成，需要审核
    - FINISH：任务已完成
    """)
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


def route_by_supervisor(state: MessagesState) -> str:
    """根据主管决策路由"""
    last_msg = state["messages"][-1].content.strip().upper()

    if "RESEARCH" in last_msg:
        return "research"
    elif "WRITING" in last_msg:
        return "writing"
    elif "REVIEW" in last_msg:
        return "review"
    else:
        return END


# 构建多 Agent 图
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("research", research_agent)
builder.add_node("writing", writing_agent)
builder.add_node("review", review_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_by_supervisor)

# 每个专家完成后返回主管
for agent in ["research", "writing", "review"]:
    builder.add_edge(agent, "supervisor")

graph = builder.compile()

# # 测试多 Agent 协作
# result = graph.invoke({
#     "messages": [HumanMessage(content="请帮我写一篇关于 Python 装饰器的简短介绍文章")]
# })
#
# print("=== 多 Agent 协作完成 ===")
# for i, msg in enumerate(result["messages"]):
#     print(f"\n[{i + 1}] {msg.type}: {msg.content[:150]}...")


# 流式观察 Agent 的每一步
for chunk in graph.stream(
    {"messages": [HumanMessage(content="请帮我写一篇关于 Python 装饰器的简短介绍文章")]},
    stream_mode="updates"
):
    for node_name, updates in chunk.items():
        print(f"\n=== 节点: {node_name} ===")
        for msg in updates.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  -> 调用工具: {tc['name']}({tc['args']})")
            else:
                print(f"  -> 输出: {msg.content[:200] if msg.content else '(无内容)'}")
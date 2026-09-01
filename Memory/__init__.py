"""
Memory 包 —— Agent Memory 机制
===============================
短期记忆（Redis）+ 长期记忆（PostgreSQL）+ Summary + 检索 + 提取 + 管理。
与 LangGraph / FastAPI 解耦，具体接入见：
  - App/LangGraph/Medical_agent_graph.py（集成层）
  - App/main.py（FastAPI 集成层）

模块划分：
  Models/      Pydantic 短期记忆模型 + SQLAlchemy 长期记忆表模型
  repository/  Redis / SQL 持久化仓库（纯 CRUD，无 LLM 逻辑）
  retriever/   长期记忆相关性检索
  summarizer/  Conversation Summary（LLM 压缩）
  extractor/   长期记忆提取（LLM 抽取 + 医疗安全策略）
  manager/     MemoryManager 统一协调
"""

from App.Memory.database import init_db, close_engine
from App.Memory.manager import MemoryManager
from App.Memory.manager.memory_manager import DEFAULT_USER_ID, format_memory_context

__all__ = [
    'init_db',
    'close_engine',
    'MemoryManager',
    "DEFAULT_USER_ID",
    "format_memory_context"
]

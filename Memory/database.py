"""
Memory/database.py
==================
数据库连接与会话管理（SQLAlchemy）。

· 默认连接 PostgreSQL（psycopg 驱动）；DATABASE_URL 支持 sqlite 兜底（测试用）。
· init_db() 创建表；失败只记 warning，不阻塞应用启动（Memory 是增强模块）。
· 提供 session_scope() 上下文管理器，统一提交/回滚/关闭。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from App.Core.config import DATABASE_URL
from App.Models.base import Base

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None
_db_available: bool = False


def _build_engine(database_url: str) -> Engine:
    kwargs = {}
    if database_url.startswith("sqlite"):
        # SQLite 默认单线程，需要允许跨线程（FastAPI 线程池中访问）
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PostgreSQL：连接池 pre_ping，避免拿到失效连接
        kwargs["pool_pre_ping"] = True
    return create_engine(database_url, **kwargs)


def init_db(database_url: Optional[str] = None) -> bool:
    """
    初始化全局 engine + 会话工厂并建表。

    Returns
    -------
    bool : 是否初始化成功（失败时不抛异常，仅记 warning）。
    """
    global _engine, _SessionLocal, _db_available
    url = database_url or DATABASE_URL
    if not url:
        _engine = None
        _SessionLocal = None
        _db_available = False
        logger.warning("Memory 数据库未配置 DATABASE_URL，数据库功能不可用")
        return False
    try:
        # Import Chat Models before create_all so the shared Base registers
        # users/sessions/messages as well as the existing Memory tables.
        # from App.Models import long_term_memory as _chat_models  # noqa: F401
        # from App.Models import conversation_summary as _memory_models  # noqa: F401

        _engine = _build_engine(url)
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        _db_available = True
        logger.info("Memory 数据库初始化成功：%s", url.split("@")[-1])
    except Exception as exc:  # 数据库不可用不能拖垮应用
        _engine = None
        _SessionLocal = None
        _db_available = False
        logger.warning("Memory 数据库初始化失败，长期记忆将不可用：%s", exc)
    return _db_available


def is_db_available() -> bool:
    return _db_available


def get_session_factory() -> Optional[sessionmaker]:
    """返回全局会话工厂；数据库不可用时返回 None。"""
    return _SessionLocal


def close_engine() -> None:
    global _engine, _SessionLocal, _db_available
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    _db_available = False


@contextmanager
def session_scope(session_factory: Optional[sessionmaker] = None) -> Iterator[Session]:
    """
    事务作用域：正常提交、异常回滚、最终关闭。

    Raises
    ------
    RuntimeError : 数据库未初始化 / 不可用时抛出，由上层（MemoryManager）降级处理。
    """
    factory = session_factory or _SessionLocal
    if factory is None:
        raise RuntimeError("Memory 数据库不可用")
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


"""Database access facade for the Core layer.

The implementation remains in the Memory infrastructure module so existing
Memory tests and Integrations keep working; new Repositories depend on this
stable Core import instead of reaching into a feature package.
"""
from App.Memory.database import (
    close_engine, get_session_factory, init_db, is_db_available, session_scope,
)

__all__ = ["close_engine", "get_session_factory", "init_db", "is_db_available", "session_scope"]

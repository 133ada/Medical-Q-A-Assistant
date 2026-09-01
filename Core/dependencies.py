"""Dependency factories and authentication guards for HTTP endpoints."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from App.Repositories.chat_repository import ChatRepository
from App.Repositories.user_repository import UserRepository
from App.Security.authentication import resolve_user_id


def get_chat_repository() -> ChatRepository:
    return ChatRepository()


def get_user_repository() -> UserRepository:
    return UserRepository()


def get_current_user_id(
    request: Request,
    users: UserRepository = Depends(get_user_repository),
) -> str:
    """Require a valid bearer identity that maps to a persisted user."""
    user_id = resolve_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login is required for this operation",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if users.get_by_id(user_id) is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return user_id


def get_optional_user_id(
    request: Request,
    users: UserRepository = Depends(get_user_repository),
) -> str | None:
    """Return a valid optional identity, preserving guest Chat support."""
    user_id = resolve_user_id(request)
    if user_id is not None and users.get_by_id(user_id) is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return user_id


__all__ = [
    "get_chat_repository", "get_user_repository", "get_current_user_id", "get_optional_user_id",
]

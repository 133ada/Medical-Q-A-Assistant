"""Authentication business operations."""
from __future__ import annotations

from fastapi import HTTPException, status

from App.Repositories.user_repository import UserRepository
from App.Security.jwt import create_access_token
from App.Security.password import hash_password, verify_password


def register_user(repository: UserRepository, username: str, password: str, nickname: str | None) -> dict:
    """Create an account and return its access token plus public profile."""
    if repository.get_by_username(username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")
    try:
        user = repository.create(username, hash_password(password), nickname or username)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="注册失败，请检查数据库连接") from exc
    return {
        "token": create_access_token(user["id"], user["username"]),
        "user": user,
    }


def login_user(repository: UserRepository, username: str, password: str) -> dict:
    """Validate a password, update login metadata, and issue a new token."""
    user = repository.get_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    repository.mark_login(user["id"])
    return {
        "token": create_access_token(user["id"], user["username"]),
        "user": user,
    }


__all__ = ["register_user", "login_user"]

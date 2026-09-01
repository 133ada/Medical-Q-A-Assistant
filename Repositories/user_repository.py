"""Persistence operations for application users."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import select

from App.Core.database import session_scope
from App.Models.user import User
from App.Repositories.base import BaseRepository


def _public_user(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "nickname": user.nickname,
        "created_at": user.created_at,
    }


class UserRepository(BaseRepository[User]):
    """Encapsulate user reads and writes without exposing SQLAlchemy sessions."""

    def __init__(self) -> None:
        super().__init__(User)

    def create(self, username: str, password_hash: str, nickname: str) -> dict:
        with session_scope() as db:
            user = User(
                id=str(uuid4()),
                username=username,
                password_hash=password_hash,
                nickname=nickname,
            )
            db.add(user)
            db.flush()
            return {**_public_user(user), "password_hash": user.password_hash}

    def get_by_username(self, username: str) -> Optional[dict]:
        with session_scope() as db:
            user = db.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
            return {**_public_user(user), "password_hash": user.password_hash} if user else None

    def get_by_id(self, user_id: str) -> Optional[dict]:
        with session_scope() as db:
            user = db.get(User, user_id)
            return _public_user(user) if user else None

    def mark_login(self, user_id: str) -> None:
        with session_scope() as db:
            user = db.get(User, user_id)
            if user is not None:
                user.last_login_at = datetime.now(timezone.utc)


__all__ = ["UserRepository"]

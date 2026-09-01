"""Persistence operations for users, sessions, and messages."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import desc, func, select

from App.Models import ChatMessage, ChatSession, User
from App.Core.database import session_scope


def _public_user(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "nickname": user.nickname,
        "created_at": user.created_at,
    }


def _message_dict(message: ChatMessage) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
    }


def _session_dict(session: ChatSession, message_count: int = 0) -> dict:
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": message_count,
    }


class ChatRepository:
    def create_user(self, username: str, password_hash: str, nickname: str) -> dict:
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

    def get_user_by_username(self, username: str) -> Optional[dict]:
        with session_scope() as db:
            user = db.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
            if user is None:
                return None
            return {**_public_user(user), "password_hash": user.password_hash}

    def get_user(self, user_id: str) -> Optional[dict]:
        with session_scope() as db:
            user = db.get(User, user_id)
            return _public_user(user) if user else None

    def mark_login(self, user_id: str) -> None:
        with session_scope() as db:
            user = db.get(User, user_id)
            if user is not None:
                user.last_login_at = datetime.now(timezone.utc)

    def create_session(self, user_id: str, title: str = "新对话") -> dict:
        with session_scope() as db:
            session = ChatSession(id=str(uuid4()), user_id=user_id, title=title or "新对话")
            db.add(session)
            db.flush()
            return _session_dict(session)

    def list_sessions(self, user_id: str, days: int = 7) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        with session_scope() as db:
            stmt = (
                select(ChatSession, func.count(ChatMessage.id).label("message_count"))
                .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
                .where(ChatSession.user_id == user_id, ChatSession.updated_at >= since)
                .group_by(ChatSession.id)
                .order_by(desc(ChatSession.updated_at))
            )
            rows = db.execute(stmt).all()
            return [_session_dict(session, int(count)) for session, count in rows]

    def get_session(self, user_id: str, session_id: str) -> Optional[dict]:
        with session_scope() as db:
            session = db.execute(
                select(ChatSession).where(
                    ChatSession.id == session_id, ChatSession.user_id == user_id
                )
            ).scalar_one_or_none()
            if session is None:
                return None
            messages = list(
                db.scalars(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session.id)
                    .order_by(ChatMessage.created_at, ChatMessage.id)
                )
            )
            return {
                **_session_dict(session, len(messages)),
                "messages": [_message_dict(message) for message in messages],
            }

    def add_message(
            self, user_id: str, session_id: str, role: str, content: str
    ) -> Optional[dict]:
        with session_scope() as db:
            session = db.execute(
                select(ChatSession).where(
                    ChatSession.id == session_id, ChatSession.user_id == user_id
                )
            ).scalar_one_or_none()
            if session is None:
                return None
            message = ChatMessage(session_id=session.id, role=role, content=content)
            db.add(message)
            session.updated_at = datetime.now(timezone.utc)
            db.flush()
            return _message_dict(message)

    def update_title_if_default(self, user_id: str, session_id: str, title: str) -> None:
        with session_scope() as db:
            session = db.execute(
                select(ChatSession).where(
                    ChatSession.id == session_id, ChatSession.user_id == user_id
                )
            ).scalar_one_or_none()
            if session is not None and session.title == "新对话":
                session.title = title[:160] or "新对话"



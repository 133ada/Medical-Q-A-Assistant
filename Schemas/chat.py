"""Pydantic contracts for Chat and session endpoints."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=160)


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class MessageRead(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class SessionDetail(SessionSummary):
    messages: list[MessageRead]


class MessagePairResponse(BaseModel):
    session: SessionSummary
    messages: list[MessageRead]


class ChatRequest(BaseModel):
    """One medical question, optionally attached to a persisted session."""
    question: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=64)
    conversation_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    question: str
    answer: str
    conversation_id: str
    session_id: str | None = None
    session: SessionSummary | None = None
    messages: list[MessageRead] = Field(default_factory=list)

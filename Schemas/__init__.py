"""Canonical API Schemas (Pydantic contracts)."""
from App.Schemas.auth import AuthResponse, UserLoginRequest, UserPublic, UserRegisterRequest
from App.Schemas.chat import (
    ChatRequest, ChatResponse, CreateSessionRequest, MessageCreateRequest,
    MessagePairResponse, MessageRead, SessionDetail, SessionSummary,
)

__all__ = [
    "AuthResponse", "UserLoginRequest", "UserPublic", "UserRegisterRequest",
    "ChatRequest", "ChatResponse", "CreateSessionRequest", "MessageCreateRequest",
    "MessagePairResponse", "MessageRead", "SessionDetail", "SessionSummary",
]

"""Authentication API Schemas."""
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class UserRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=72)
    nickname: str | None = Field(default=None, min_length=1, max_length=64)


class UserLoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=72)


class UserPublic(BaseModel):
    id: str
    username: str
    nickname: str
    created_at: datetime


class AuthResponse(BaseModel):
    token: str
    user: UserPublic

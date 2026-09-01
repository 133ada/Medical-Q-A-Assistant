"""Small reusable base class for SQLAlchemy Repositories."""
from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy.orm import DeclarativeBase

ModelT = TypeVar("ModelT", bound=DeclarativeBase)


class BaseRepository(Generic[ModelT]):
    """Store the ORM model type shared by a repository implementation."""

    def __init__(self, model: type[ModelT]) -> None:
        self.model = model


__all__ = ["BaseRepository"]

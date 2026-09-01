"""Shared SQLAlchemy declarative base for application ORM Models."""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class shared by Chat and Memory ORM entities."""


__all__ = ["Base"]

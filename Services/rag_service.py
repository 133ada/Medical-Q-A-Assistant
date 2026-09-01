"""RAG integration boundary.

The current LangGraph workflow owns the actual retrieval sequence.  This
module exists as the stable service-layer seam for future direct retrieval
endpoints or provider substitutions.
"""
from __future__ import annotations

from App.Integrations.rag import build_vectorstore_ddi, build_vectorstore_other

__all__ = ["build_vectorstore_ddi", "build_vectorstore_other"]


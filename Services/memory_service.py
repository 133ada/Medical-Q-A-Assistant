"""Application service façade for the existing Memory subsystem."""
from __future__ import annotations

from fastapi import BackgroundTasks, Request


def schedule_turn_memory_update(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str | None,
    conversation_id: str,
    question: str,
    answer: str,
) -> None:
    """Queue Memory extraction only for authenticated conversations."""
    manager = getattr(request.app.state, "memory_manager", None)
    if manager is not None and user_id is not None:
        background_tasks.add_task(
            manager.update_after_turn, user_id, conversation_id, question, answer
        )


__all__ = ["schedule_turn_memory_update"]

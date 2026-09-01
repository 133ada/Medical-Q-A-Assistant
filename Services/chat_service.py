"""Application service for the single medical-Chat execution path.

Routers call this module for all RAG/Agent work. It intentionally owns no
HTTP routes and has no FastAPI application construction code, which keeps the
same workflow usable from both the JSON and streaming endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import BackgroundTasks, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from App.Integrations.langgraph import run_medical_agent, stream_medical_agent
from App.Services.memory_service import schedule_turn_memory_update
from App.Repositories.chat_repository import ChatRepository
from App.Schemas.chat import ChatRequest

logger = logging.getLogger(__name__)


def _mock_reply(question: str) -> str:
    return (
        "我已经收到你的问题。当前服务运行在演示模式，未执行文档检索。\n\n"
        f"你的问题：{question}"
    )


def title_from_question(question: str) -> str:
    compact = " ".join(question.strip().split())
    return compact[:40] + ("…" if len(compact) > 40 else "")


def conversation_id_for(payload: ChatRequest) -> str:
    return payload.session_id or payload.conversation_id or str(uuid4())


async def generate_answer(
    request: Request,
    question: str,
    user_id: str | None,
    conversation_id: str,
) -> str:
    """Run the configured answer mode, defaulting to the real medical graph."""
    mode = getattr(request.app.state, "chat_reply_mode", "medical_agent")
    if mode == "mock":
        logger.warning("Chat API is in mock mode; document retrieval is skipped")
        return _mock_reply(question)

    graph = getattr(request.app.state, "medical_graph", None)
    if mode != "medical_agent" or graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="医疗 Agent 尚未就绪，暂时无法执行文档检索",
        )

    return await run_in_threadpool(
        run_medical_agent,
        graph,
        question,
        user_id=user_id,
        conversation_id=conversation_id,
    )


def schedule_memory_update(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str | None,
    conversation_id: str,
    question: str,
    answer: str,
) -> None:
    """Backward-compatible name for the Memory service boundary."""
    schedule_turn_memory_update(
        request, background_tasks, user_id, conversation_id, question, answer
    )

async def ask_anonymous(
    request: Request,
    payload: ChatRequest,
) -> dict:
    """Answer an anonymous question without server-side session persistence."""
    conversation_id = conversation_id_for(payload)
    answer = await generate_answer(request, payload.question, None, conversation_id)
    return {
        "question": payload.question,
        "answer": answer,
        "conversation_id": conversation_id,
        "session_id": None,
        "session": None,
        "messages": [],
    }


async def ask_in_session(
    request: Request,
    background_tasks: BackgroundTasks,
    repository: ChatRepository,
    user_id: str,
    payload: ChatRequest,
    session_id: str | None = None,
) -> dict:
    """Run RAG, persist the complete message pair, and queue Memory writes."""
    active_session_id = session_id or payload.session_id or payload.conversation_id
    if active_session_id is None:
        active_session_id = repository.create_session(user_id)["id"]
    elif repository.get_session(user_id, active_session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # Do not save only one half of a Chat pair when an LLM or vector store fails.
    answer = await generate_answer(
        request, payload.question, user_id, active_session_id
    )
    user_message = repository.add_message(
        user_id, active_session_id, "user", payload.question
    )
    assistant_message = repository.add_message(
        user_id, active_session_id, "assistant", answer
    )
    if user_message is None or assistant_message is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    repository.update_title_if_default(
        user_id, active_session_id, title_from_question(payload.question)
    )
    detail = repository.get_session(user_id, active_session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    schedule_memory_update(
        request,
        background_tasks,
        user_id,
        active_session_id,
        payload.question,
        answer,
    )
    return {
        "question": payload.question,
        "answer": answer,
        "conversation_id": active_session_id,
        "session_id": active_session_id,
        "messages": [user_message, assistant_message],
        "session": detail,
    }


_STOP = object()


def _next_or_stop(generator):
    try:
        return next(generator)
    except StopIteration:
        return _STOP


async def stream_answer(
    request: Request,
    payload: ChatRequest,
    user_id: str | None,
    background_tasks: BackgroundTasks,
) -> AsyncGenerator[str, None]:
    """Yield graph progress and final answer for the streaming compatibility API."""
    conversation_id = conversation_id_for(payload)
    mode = getattr(request.app.state, "chat_reply_mode", "medical_agent")
    if mode == "mock":
        answer = _mock_reply(payload.question)
        yield answer
        schedule_memory_update(
            request, background_tasks, user_id, conversation_id, payload.question, answer
        )
        return

    graph = getattr(request.app.state, "medical_graph", None)
    if mode != "medical_agent" or graph is None:
        yield "医疗 Agent 尚未就绪，暂时无法执行文档检索。"
        return

    generator = stream_medical_agent(
        graph=graph,
        question=payload.question,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    answer = ""
    loop = asyncio.get_running_loop()
    while True:
        token = await loop.run_in_executor(None, _next_or_stop, generator)
        if token is _STOP:
            break
        yield token
        if not token.startswith("["):
            answer = token

    if answer:
        schedule_memory_update(
            request, background_tasks, user_id, conversation_id, payload.question, answer
        )






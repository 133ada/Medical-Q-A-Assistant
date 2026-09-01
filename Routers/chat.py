"""Chat JSON/streaming endpoints and legacy route aliases."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse

from App.Core.dependencies import get_chat_repository, get_optional_user_id
from App.Repositories.chat_repository import ChatRepository
from App.Schemas.chat import ChatRequest, ChatResponse
from App.Services.chat_service import ask_anonymous, ask_in_session, stream_answer

router = APIRouter(prefix="/api", tags=["chat"])
legacy_router = APIRouter(tags=["legacy-chat"], deprecated=True)

@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request, background_tasks: BackgroundTasks,
               user_id: str | None = Depends(get_optional_user_id),
               repository: ChatRepository = Depends(get_chat_repository)):
    if user_id is None:
        return await ask_anonymous(request, payload)
    return await ask_in_session(request, background_tasks, repository, user_id, payload)

@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request,
                      user_id: str | None = Depends(get_optional_user_id)):
    background_tasks = BackgroundTasks()
    return StreamingResponse(
        stream_answer(request, payload, user_id, background_tasks),
        media_type="text/plain; charset=utf-8",
        background=background_tasks,
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
    )

legacy_router.add_api_route("/chat", chat, methods=["POST"], response_model=ChatResponse, deprecated=True)
legacy_router.add_api_route("/chat/stream", chat_stream, methods=["POST"], deprecated=True)

__all__ = ["router", "legacy_router", "chat", "chat_stream"]

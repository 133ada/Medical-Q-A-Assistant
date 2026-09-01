"""Session and persisted-message endpoints."""
from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from App.Core.dependencies import get_chat_repository, get_current_user_id
from App.Repositories.chat_repository import ChatRepository
from App.Schemas.chat import ChatRequest, CreateSessionRequest, MessageCreateRequest, MessagePairResponse, SessionDetail, SessionSummary
from App.Services.chat_service import ask_in_session

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

@router.get("", response_model=List[SessionSummary])
def list_sessions(user_id: str = Depends(get_current_user_id), repository: ChatRepository = Depends(get_chat_repository)):
    return repository.list_sessions(user_id, days=7)

@router.post("", response_model=SessionSummary, status_code=201)
def create_session(payload: Optional[CreateSessionRequest] = None,
                   user_id: str = Depends(get_current_user_id), repository: ChatRepository = Depends(get_chat_repository)):
    return repository.create_session(user_id, (payload.title if payload else None) or "新对话")

@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, user_id: str = Depends(get_current_user_id), repository: ChatRepository = Depends(get_chat_repository)):
    detail = repository.get_session(user_id, session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail

@router.post("/{session_id}/messages", response_model=MessagePairResponse)
async def append_message(session_id: str, payload: MessageCreateRequest, request: Request,
                         background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user_id),
                         repository: ChatRepository = Depends(get_chat_repository)):
    result = await ask_in_session(request, background_tasks, repository, user_id,
                                  ChatRequest(question=payload.content, session_id=session_id),
                                  session_id=session_id)
    detail = result["session"]
    return {"session": {key: detail[key] for key in SessionSummary.model_fields.keys()},
            "messages": result["messages"]}

__all__ = ["router"]

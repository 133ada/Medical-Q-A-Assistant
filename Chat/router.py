"""HTTP routes for authentication, Chat sessions, and medical questions.

The canonical frontend endpoint is POST /api/Chat. Session-specific endpoints
remain available for history navigation and compatibility, but all answer
generation is delegated to App.Chat.service.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from App.Utils.auth import resolve_user_id
from App.Repositories.chat_repository import ChatRepository
from App.Schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    MessageCreateRequest,
    MessagePairResponse,
    SessionDetail,
    SessionSummary,
    UserLoginRequest,
    UserPublic,
    UserRegisterRequest,
)
from App.Security.jwt import create_access_token
from App.Security.password import hash_password, verify_password
from App.Services.chat_service import ask_anonymous, ask_in_session, stream_answer

logger = logging.getLogger(__name__)

# All new clients should use this router under /api.
router = APIRouter(prefix="/api", tags=["account-Chat"])
repo = ChatRepository()


def _current_user_id(request: Request) -> str:
    """Require an authenticated, still-existing application user."""
    user_id = resolve_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login is required for this operation",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if repo.get_user(user_id) is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user_id


def _optional_user_id(request: Request) -> Optional[str]:
    """Resolve an optional identity for /api/Chat guest-mode support."""
    user_id = resolve_user_id(request)
    if user_id is not None and repo.get_user(user_id) is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user_id


@router.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(payload: UserRegisterRequest):
    if repo.get_user_by_username(payload.username):
        raise HTTPException(status_code=409, detail="用户名已存在")
    nickname = payload.nickname or payload.username
    try:
        user = repo.create_user(payload.username, hash_password(payload.password), nickname)
    except Exception:
        logger.exception("User registration failed")
        raise HTTPException(status_code=500, detail="注册失败，请检查数据库连接")
    token = create_access_token(user["id"], user["username"])
    return {"token": token, "user": UserPublic.model_validate(user)}


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: UserLoginRequest):
    user = repo.get_user_by_username(payload.username)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    repo.mark_login(user["id"])
    token = create_access_token(user["id"], user["username"])
    return {"token": token, "user": UserPublic.model_validate(user)}


@router.get("/auth/me", response_model=UserPublic)
def me(request: Request):
    user = repo.get_user(_current_user_id(request))
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions(request: Request):
    return repo.list_sessions(_current_user_id(request), days=7)


@router.post("/sessions", response_model=SessionSummary, status_code=201)
def create_session(request: Request, payload: Optional[CreateSessionRequest] = None):
    user_id = _current_user_id(request)
    title = (payload.title if payload else None) or "新对话"
    return repo.create_session(user_id, title)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, request: Request):
    detail = repo.get_session(_current_user_id(request), session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail


@router.post("/chat", response_model=ChatResponse)
async def chat(
        payload: ChatRequest,
        request: Request,
        background_tasks: BackgroundTasks,
):
    """Canonical medical-Chat endpoint for both guest and signed-in clients.

    Guest request: runs RAG but stores nothing server side.
    Authenticated request: runs RAG, persists the message pair, and writes
    Memory asynchronously. session_id is optional; a new session is created
    automatically when it is absent.
    """
    user_id = _optional_user_id(request)
    if user_id is None:
        return await ask_anonymous(request, payload)
    return await ask_in_session(request, background_tasks, repo, user_id, payload)


@router.post("/sessions/{session_id}/messages", response_model=MessagePairResponse)
async def append_message(
        session_id: str,
        payload: MessageCreateRequest,
        request: Request,
        background_tasks: BackgroundTasks,
):
    """Compatibility endpoint to append one question to a known session."""
    result = await ask_in_session(
        request,
        background_tasks,
        repo,
        _current_user_id(request),
        ChatRequest(question=payload.content, session_id=session_id),
        session_id=session_id,
    )
    detail = result["session"]
    return {
        "session": {key: detail[key] for key in SessionSummary.model_fields.keys()},
        "messages": result["messages"],
    }


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    """Stream Supervisor/Retriever progress and the final RAG answer."""
    user_id = _optional_user_id(request)
    background_tasks = BackgroundTasks()
    return StreamingResponse(
        stream_answer(request, payload, user_id, background_tasks),
        media_type="text/plain; charset=utf-8",
        background=background_tasks,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache",
        },
    )


"""Public router assembly for the layered API."""
from fastapi import APIRouter
from App.Routers.auth import router as auth_router
from App.Routers.chat import legacy_router, router as chat_router
from App.Routers.sessions import router as sessions_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(sessions_router)

__all__ = ["router", "auth_router", "chat_router", "sessions_router", "legacy_router"]

"""Authentication endpoints."""
from fastapi import APIRouter, Depends

from App.Core.dependencies import get_current_user_id, get_user_repository
from App.Repositories.user_repository import UserRepository
from App.Schemas.auth import AuthResponse, UserLoginRequest, UserPublic, UserRegisterRequest
from App.Services.auth_service import login_user, register_user

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(payload: UserRegisterRequest, users: UserRepository = Depends(get_user_repository)):
    return register_user(users, payload.username, payload.password, payload.nickname)

@router.post("/login", response_model=AuthResponse)
def login(payload: UserLoginRequest, users: UserRepository = Depends(get_user_repository)):
    return login_user(users, payload.username, payload.password)

@router.get("/me", response_model=UserPublic)
def me(user_id: str = Depends(get_current_user_id), users: UserRepository = Depends(get_user_repository)):
    return users.get_by_id(user_id)

__all__ = ["router"]

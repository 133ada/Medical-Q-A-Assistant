"""Request authentication helpers.

The API accepts a bearer token and resolves it to a server-side user id.
The user id in the JSON request body is intentionally never trusted.
"""

from __future__ import annotations

import hmac
import json
import logging
from typing import Optional

from fastapi import HTTPException, Request, status

from App.Security.jwt import decode_access_token
from App.Core.config import AUTH_REQUIRED, AUTH_TOKENS_JSON

logger = logging.getLogger(__name__)


def _load_token_map() -> dict[str, str]:
    if not AUTH_TOKENS_JSON.strip():
        return {}
    try:
        raw = json.loads(AUTH_TOKENS_JSON)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AUTH_TOKENS_JSON must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("AUTH_TOKENS_JSON must be an object: {token: user_id}")
    return {
        str(token): str(user_id)
        for token, user_id in raw.items()
        if str(token).strip() and str(user_id).strip()
    }


def resolve_user_id(request: Request) -> Optional[str]:
    """Resolve an authenticated user id without trusting request-body fields."""
    authorization = request.headers.get("Authorization", "")
    if not authorization:
        if AUTH_REQUIRED:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization must use the Bearer scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_map = _load_token_map()
    user_id = next(
        (mapped_user for configured_token, mapped_user in token_map.items()
         if hmac.compare_digest(configured_token, token.strip())),
        None,
    )
    if user_id is None:
        payload = decode_access_token(token.strip())
        user_id = str(payload["sub"]) if payload else None
    if user_id is None:
        logger.warning("Rejected request with an unknown bearer token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id

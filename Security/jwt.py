"""JWT creation and verification primitives.

The implementation deliberately uses only standard-library cryptography so
it is portable across local and container deployments.  The secret is loaded
once from :mod:`App.Core.config` during application startup.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from App.Core.config import JWT_SECRET

JWT_EXPIRE_SECONDS = int(os.environ.get("JWT_EXPIRE_SECONDS", str(60 * 60 * 24 * 7)))


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_access_token(subject: str, username: str) -> str:
    """Create a signed HS256 access token for a persisted user."""
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not configured")

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "username": username,
        "iat": now,
        "exp": now + JWT_EXPIRE_SECONDS,
    }
    encoded_header = _b64encode(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{_b64encode(signature)}"


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Return a validated token payload, or ``None`` for an invalid token."""
    if not JWT_SECRET:
        return None
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        header = json.loads(_b64decode(encoded_header))
        if header.get("alg") != "HS256" or header.get("typ") != "JWT":
            return None
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        expected = hmac.new(
            JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64decode(encoded_signature)):
            return None
        payload = json.loads(_b64decode(encoded_payload))
        if not isinstance(payload, dict):
            return None
        if int(payload.get("exp", 0)) <= int(time.time()):
            return None
        if not str(payload.get("sub", "")).strip():
            return None
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


__all__ = ["JWT_EXPIRE_SECONDS", "create_access_token", "decode_access_token"]

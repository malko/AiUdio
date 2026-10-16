"""Access control: HTTP Basic (UI) and API keys (machines).

Enabled by ``auth.enabled``. Paths listed in ``auth.exempt_paths`` (/health,
/ready) remain reachable without authentication.
"""

from __future__ import annotations

import secrets
from typing import Iterable, Optional

from fastapi import HTTPException, Request, WebSocket, status

UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Authentication required",
    headers={"WWW-Authenticate": "Basic realm=\"aiudio\""},
)


def _basic_ok(auth: str, username: str, password: str) -> bool:
    import base64

    try:
        scheme, _, payload = auth.partition(" ")
        if scheme.lower() != "basic":
            return False
        decoded = base64.b64decode(payload).decode("utf-8")
        user, _, pwd = decoded.partition(":")
    except Exception:
        return False
    return secrets.compare_digest(user, username) and secrets.compare_digest(pwd, password)


def _bearer_ok(token: str, api_keys: Iterable[str]) -> bool:
    return any(secrets.compare_digest(token, key) for key in api_keys)


def _path_exempt(path: str, exempt_paths: Iterable[str]) -> bool:
    return any(path == p or path.startswith(p.rstrip("/") + "/") for p in exempt_paths)


def _authorized(settings, authorization: Optional[str], x_api_key: Optional[str], query_key: Optional[str]) -> bool:
    auth = settings.auth
    if not auth.enabled:
        return True
    if x_api_key and _bearer_ok(x_api_key, auth.api_keys):
        return True
    if query_key and _bearer_ok(query_key, auth.api_keys):
        return True
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and _bearer_ok(token, auth.api_keys):
            return True
        if scheme.lower() == "basic" and auth.password:
            return _basic_ok(authorization, auth.username, auth.password)
    return False


def request_authorized(request: Request, settings) -> bool:
    """Synchronous version used by the HTTP middleware."""
    if _path_exempt(request.url.path, settings.auth.exempt_paths):
        return True
    return _authorized(
        settings,
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
        request.query_params.get("api_key"),
    )


async def require_auth(request: Request, settings) -> None:
    if _path_exempt(request.url.path, settings.auth.exempt_paths):
        return
    if not _authorized(
        settings,
        request.headers.get("authorization"),
        request.headers.get("x-api-key"),
        request.query_params.get("api_key"),
    ):
        raise UNAUTHORIZED


async def authorize_ws(websocket: WebSocket, settings) -> bool:
    if not settings.auth.enabled:
        return True
    return _authorized(
        settings,
        websocket.headers.get("authorization"),
        websocket.headers.get("x-api-key"),
        websocket.query_params.get("api_key"),
    )

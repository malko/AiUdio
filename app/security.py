"""Access control: HTTP Basic (UI) and API keys (machines).

Enabled by ``auth.enabled``. Paths listed in ``auth.exempt_paths`` (/health,
/ready) remain reachable without authentication.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Iterable, Optional

from fastapi import HTTPException, Request, WebSocket, status

from . import apikeys
from .apikeys import ApiKey, verify

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


def _match_key(settings, *candidates: Optional[str]) -> Optional[ApiKey]:
    """First candidate that matches a live (client or session) key, else None."""
    keys = apikeys.all_keys(settings)
    for candidate in candidates:
        if not candidate:
            continue
        found = verify(candidate, keys)
        if found is not None:
            apikeys.mark_used(found.id)
            return found
    return None


def _path_exempt(path: str, exempt_paths: Iterable[str]) -> bool:
    return any(path == p or path.startswith(p.rstrip("/") + "/") for p in exempt_paths)


def _authorized(settings, authorization: Optional[str], x_api_key: Optional[str], query_key: Optional[str]) -> bool:
    auth = settings.auth
    if not auth.enabled:
        return True
    if _match_key(settings, x_api_key, query_key) is not None:
        return True
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and _match_key(settings, token) is not None:
            return True
        if scheme.lower() == "basic" and auth.password:
            return _basic_ok(authorization, auth.username, auth.password)
    return False


def is_basic_admin(settings, authorization: Optional[str]) -> bool:
    """True when the request carries valid HTTP Basic admin credentials."""
    auth = settings.auth
    if not auth.enabled:
        return True
    if not authorization or not auth.password:
        return False
    scheme, _, _ = authorization.partition(" ")
    return scheme.lower() == "basic" and _basic_ok(authorization, auth.username, auth.password)


def is_admin(settings, authorization: Optional[str], x_api_key: Optional[str], query_key: Optional[str]) -> bool:
    """Admin privilege: Basic admin, or an admin-scoped session key."""
    if not settings.auth.enabled:
        return True
    if is_basic_admin(settings, authorization):
        return True
    bearer = None
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            bearer = token
    found = _match_key(settings, x_api_key, query_key, bearer)
    return bool(found and found.admin)


def presented_key(authorization: Optional[str], x_api_key: Optional[str], query_key: Optional[str]) -> Optional[str]:
    """The raw key presented on the request, whatever the transport."""
    if x_api_key:
        return x_api_key
    if query_key:
        return query_key
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            return token
    return None


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


def validate_media_ref(ref: Optional[str], settings, reference_dir: Optional[str] = None) -> Optional[str]:
    """Validate a client-supplied reference-audio reference.

    The Qwen3-TTS engine accepts paths, URLs and base64 data-URLs, which is a
    SSRF / local-file-disclosure vector if exposed to untrusted clients. By
    default only base64 data-URLs are allowed; remote URLs and local paths must
    be explicitly enabled in ``audio.allow_remote_media`` /
    ``audio.allow_local_ref_paths``. Local paths are confined to the reference
    directory.
    """
    if ref is None:
        return None
    ref = ref.strip()
    if not ref:
        return None

    low = ref.lower()
    if low.startswith("data:"):
        return ref
    if low.startswith(("http://", "https://")):
        if settings.audio.allow_remote_media:
            return ref
        raise HTTPException(400, "Remote media URLs are disabled")
    if "://" in low:  # file://, ftp://, etc.
        raise HTTPException(400, "Unsupported media reference scheme")

    # Anything else is treated as a filesystem path.
    if not settings.audio.allow_local_ref_paths:
        raise HTTPException(400, "Local reference paths are disabled")
    if reference_dir:
        base = Path(reference_dir).resolve()
        candidate = (base / ref).resolve() if not Path(ref).is_absolute() else Path(ref).resolve()
        if candidate != base and not candidate.is_relative_to(base):
            raise HTTPException(400, "Reference path escapes the reference directory")
        if not candidate.exists():
            raise HTTPException(400, "Reference file not found")
        return str(candidate)
    return ref

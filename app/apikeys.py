"""API keys: model, hashing and verification.

Keys are generated here (`aiud_` + urlsafe random), stored **hashed** (sha256) and
verified in constant time. Two kinds coexist:

- **client keys**: created from the admin, persisted in ``config.yaml`` (hashed),
  optional expiration;
- **session keys**: minted from the admin's HTTP Basic credentials
  (``POST /api/session``), short-lived, kept **in memory only** (never written to
  disk) so the admin UI gets a normal bearer key for its HTTP/WS calls and can
  still be logged out/revoked.

``last_used_at`` is tracked in memory (not persisted): persisting it would rewrite
``config.yaml`` on every authenticated request.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .textclean import Policy

KEY_PREFIX = "aiud_"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def hash_key(key: str) -> str:
    """Stable hash of a key (what is stored at rest)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class ApiKey(BaseModel):
    id: str = Field(default_factory=lambda: secrets.token_hex(6))
    name: str = ""
    # Short display prefix (never enough to guess the key).
    prefix: str = ""
    hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: Optional[datetime] = None
    # Session keys are admin-scoped (may manage other keys).
    admin: bool = False
    # Per-client input-text cleanup: only the fields set here override the global
    # `text` setting, `None` inherits it entirely.
    text: Optional[Policy] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        exp = _aware(self.expires_at)
        if exp is None:
            return False
        return exp <= (now or utcnow())


def new_key(
    name: str = "",
    *,
    expires_at: Optional[datetime] = None,
    ttl_minutes: Optional[int] = None,
    admin: bool = False,
    text: Optional[Policy] = None,
) -> tuple[str, ApiKey]:
    """Generate a fresh key. Returns ``(plaintext, record)`` (plaintext shown once)."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    if expires_at is None and ttl_minutes:
        expires_at = utcnow() + timedelta(minutes=int(ttl_minutes))
    record = ApiKey(
        name=name.strip(),
        prefix=raw[:12],
        hash=hash_key(raw),
        expires_at=expires_at,
        admin=admin,
        text=text,
    )
    return raw, record


def from_legacy(value: str, name: str = "") -> ApiKey:
    """Coerce a plaintext key (legacy config) into a hashed record."""
    return ApiKey(name=name, prefix=value[:12] if value else "", hash=hash_key(value))


def verify(candidate: Optional[str], keys: List[ApiKey]) -> Optional[ApiKey]:
    """Return the matching non-expired key, or None (constant-time compare)."""
    if not candidate:
        return None
    digest = hash_key(candidate)
    for key in keys:
        if key.is_expired():
            continue
        if key.hash and secrets.compare_digest(key.hash, digest):
            return key
    return None


# --------------------------------------------------------------- session store
_sessions: Dict[str, ApiKey] = {}
_last_used: Dict[str, datetime] = {}


def add_session(key: ApiKey) -> None:
    _sessions[key.id] = key


def drop_session(key_id: str) -> None:
    _sessions.pop(key_id, None)
    _last_used.pop(key_id, None)


def clear_sessions() -> None:
    _sessions.clear()


def sessions() -> List[ApiKey]:
    # Drop expired sessions lazily.
    for key_id in [k for k, v in _sessions.items() if v.is_expired()]:
        drop_session(key_id)
    return list(_sessions.values())


def all_keys(settings) -> List[ApiKey]:
    """Persisted client keys + live session keys."""
    return list(settings.auth.api_keys) + sessions()


def mark_used(key_id: str) -> None:
    _last_used[key_id] = utcnow()


def last_used(key_id: str) -> Optional[datetime]:
    return _last_used.get(key_id)


def public(key: ApiKey) -> Dict[str, object]:
    """Admin-facing view of a key: never the secret, never the hash."""
    return {
        "id": key.id,
        "name": key.name,
        "prefix": key.prefix,
        "created_at": key.created_at,
        "expires_at": key.expires_at,
        "admin": key.admin,
        "expired": key.is_expired(),
        "last_used_at": _last_used.get(key.id),
        "session": key.id in _sessions,
        # Only the fields that override the global `text` setting.
        "text": key.text.model_dump(exclude_unset=True) if key.text is not None else None,
    }

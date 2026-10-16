"""Reusable voices ("avatars"): cloned from audio or generated from a description.

Each avatar is a reference WAV (+ optional transcript) that can be used as a voice in
the Synthesis and Conversation tabs. A "design" avatar also keeps the text prompt used
to generate its reference, so it can be regenerated or edited.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "voice"


@dataclass
class Avatar:
    id: str
    name: str
    kind: str = "design"          # "design" | "clone"
    language: str = "Auto"
    prompt: str = ""              # design instruction (regeneration)
    ref_text: str = ""           # transcript of the reference WAV
    personality: str = ""        # extra system prompt used in conversation
    x_vector_only: bool = False  # cloned: use speaker embedding only (no transcript)
    has_photo: bool = False
    created_at: float = field(default_factory=time.time)


class AvatarStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _meta(self, avatar_id: str) -> Path:
        return self.dir / f"{avatar_id}.json"

    def wav_path(self, avatar_id: str) -> Path:
        return self.dir / f"{avatar_id}.wav"

    def photo_path(self, avatar_id: str) -> Path:
        return self.dir / f"{avatar_id}.png"

    def new_id(self, name: str) -> str:
        base = _slug(name)
        candidate = base
        while self._meta(candidate).exists():
            candidate = f"{base}-{uuid.uuid4().hex[:6]}"
        return candidate

    def list(self) -> List[Avatar]:
        avatars: List[Avatar] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                avatars.append(Avatar(**data))
            except Exception:
                continue
        return sorted(avatars, key=lambda a: a.created_at)

    def get(self, avatar_id: str) -> Optional[Avatar]:
        path = self._meta(avatar_id)
        if not path.is_file():
            return None
        try:
            return Avatar(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def save(self, avatar: Avatar) -> Avatar:
        avatar.has_photo = self.photo_path(avatar.id).is_file()
        self._meta(avatar.id).write_text(json.dumps(asdict(avatar), ensure_ascii=False, indent=2), encoding="utf-8")
        return avatar

    def delete(self, avatar_id: str) -> None:
        for path in (self._meta(avatar_id), self.wav_path(avatar_id), self.photo_path(avatar_id)):
            path.unlink(missing_ok=True)

"""Reusable voices ("avatars"): all avatars are voice clones.

Each avatar is a reference WAV (+ optional transcript) that can be used as a voice in
the Synthesis and Conversation tabs. A clone can be built from an uploaded clip or from
a ~10s fingerprint rendered by Voice Design; ``prompt`` keeps the description used to
render that fingerprint so it can be regenerated.
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

    def __post_init__(self) -> None:
        # A cloned voice with no transcript can only use the speaker embedding.
        if self.kind == "clone" and not (self.ref_text or "").strip():
            self.x_vector_only = True


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

    def _file_count(self, avatar_id: str) -> int:
        """Files actually on disk for one voice (json + wav + optional png)."""
        return sum(
            1
            for path in (self._meta(avatar_id), self.wav_path(avatar_id), self.photo_path(avatar_id))
            if path.is_file()
        )

    def hygiene(self) -> dict:
        """Audit the directory: what it holds, what is duplicated, what is stranded.

        Every voice is one `.json` (+ `.wav`, + `.png` when it has a photo), and a
        clip or photo whose `.json` is gone is invisible to the API but still on
        disk — that is what "orphans" means here.
        """
        entries = self.list()
        referenced = set()
        for avatar in entries:
            referenced.update(
                {
                    self._meta(avatar.id).name,
                    self.wav_path(avatar.id).name,
                    self.photo_path(avatar.id).name,
                }
            )
        files = sorted(path for path in self.dir.iterdir() if path.is_file())
        by_kind: dict = {}
        orphans = []
        total = 0
        for path in files:
            size = path.stat().st_size
            total += size
            kind = path.suffix.lstrip(".").lower() or "other"
            by_kind[kind] = by_kind.get(kind, 0) + 1
            if path.name not in referenced:
                orphans.append({"name": path.name, "bytes": size})
        by_name: dict = {}
        for avatar in entries:
            by_name.setdefault(avatar.name.strip().lower(), []).append(avatar.id)
        return {
            "avatars": len(entries),
            "files": len(files),
            "by_kind": by_kind,
            "total_bytes": total,
            "duplicates": [
                {"name": name, "ids": ids, "files": sum(self._file_count(i) for i in ids)}
                for name, ids in by_name.items()
                if len(ids) > 1
            ],
            "orphans": sorted(orphans, key=lambda item: item["name"]),
            "orphan_bytes": sum(item["bytes"] for item in orphans),
            "broken": [a.id for a in entries if not self.wav_path(a.id).is_file()],
        }

    def clean_orphans(self) -> list:
        """Delete the files no voice entry references (stranded clips and photos)."""
        removed = []
        for orphan in self.hygiene()["orphans"]:
            try:
                (self.dir / orphan["name"]).unlink()
                removed.append(orphan["name"])
            except OSError:  # pragma: no cover - gone or not ours
                continue
        return removed

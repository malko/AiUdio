"""On-demand STT model management for the NeMo-Speech.cpp service.

The STT runtime is a separate C++ process that loads its model once at startup and
cannot hot-reload. To support on-demand downloads from the UI, the gateway reads the
same model index NeMo-Speech.cpp uses, downloads the requested GGUF into the shared
model cache (matching NeMo's layout: <cache>/<org>/<repo>/<revision>/<file>), verifies
its SHA-256, and lets the STT entrypoint start (or restart) the server once the file is
present.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import httpx

# Pointer files written by the gateway so the STT container knows which model to load.
ACTIVE_ASR = ".active"
ACTIVE_DIAR = ".active-diar"


@dataclass
class Artifact:
    role: str
    filename: str
    size: int = 0
    sha256: str = ""
    type: str = "file"
    revision: str = ""


@dataclass
class IndexedModel:
    repo: str
    aliases: List[str] = field(default_factory=list)
    revision: str = ""
    artifacts: List[Artifact] = field(default_factory=list)

    def asr_artifact(self) -> Optional[Artifact]:
        for artifact in self.artifacts:
            if artifact.role == "asr" and artifact.type == "file":
                return artifact
        return None

    def artifact_revision(self, artifact: Artifact) -> str:
        return artifact.revision or self.revision

    def local_path(self, artifact: Artifact, models_dir: str) -> Path:
        # Flat layout (<models_dir>/<filename>) shared with the STT container, which
        # is started with an explicit --asr-model path.
        return Path(models_dir) / artifact.filename


class ModelIndex:
    def __init__(self, models: List[IndexedModel], defaults: Optional[Dict[str, str]] = None):
        self.models = models
        self.defaults = defaults or {}

    @classmethod
    def from_json(cls, data: Dict) -> ModelIndex:
        models: List[IndexedModel] = []
        for raw in data.get("models", []):
            artifacts = [
                Artifact(
                    role=a.get("role", ""),
                    filename=a.get("filename", ""),
                    size=int(a.get("size", 0) or 0),
                    sha256=a.get("sha256", ""),
                    type=a.get("type", "file"),
                    revision=a.get("revision", ""),
                )
                for a in raw.get("artifacts", [])
            ]
            models.append(
                IndexedModel(
                    repo=raw.get("repo", ""),
                    aliases=list(raw.get("aliases", []) or []),
                    revision=raw.get("revision", ""),
                    artifacts=artifacts,
                )
            )
        return cls(models, data.get("defaults", {}))

    @classmethod
    def from_file(cls, path: Path) -> ModelIndex:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def find(self, spec: str) -> Optional[IndexedModel]:
        spec = (spec or "").strip()
        if not spec:
            return None
        for model in self.models:
            if model.repo == spec or spec in model.aliases:
                return model
        return None

    def default_asr(self) -> Optional[IndexedModel]:
        repo = self.defaults.get("asr", "")
        return self.find(repo) if repo else (self.models[0] if self.models else None)

    def with_status(self, models_dir: str) -> List[Dict]:
        out: List[Dict] = []
        for model in self.models:
            artifact = model.asr_artifact()
            if artifact is None:
                continue
            path = model.local_path(artifact, models_dir)
            cached = path.is_file() and (not artifact.size or path.stat().st_size == artifact.size)
            out.append(
                {
                    "repo": model.repo,
                    "aliases": model.aliases,
                    "size": artifact.size,
                    "filename": artifact.filename,
                    "cached": cached,
                }
            )
        return out


def load_index(path: str) -> Optional[ModelIndex]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return ModelIndex.from_file(p)
    except Exception:
        return None


def mark_active(models_dir: str, filename: str, role: str = "asr") -> None:
    """Write the pointer file the STT container watches to load a model."""
    name = ACTIVE_DIAR if role == "diarization" else ACTIVE_ASR
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(filename, encoding="utf-8")


async def iter_download(
    model: IndexedModel,
    artifact: Artifact,
    models_dir: str,
    base_url: str,
    chunk_size: int = 1 << 20,
) -> AsyncIterator[Dict]:
    """Download an artifact, yielding progress dicts; the last yields ``path``."""
    revision = model.artifact_revision(artifact)
    dest = model.local_path(artifact, models_dir)
    total = artifact.size or 0

    if dest.is_file() and (not total or dest.stat().st_size == total):
        yield {"status": "cached", "path": str(dest), "downloaded": total, "total": total}
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    url = f"{base_url.rstrip('/')}/{model.repo}/resolve/{revision}/{artifact.filename}?download=true"
    hasher = hashlib.sha256()
    downloaded = 0
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(600.0, connect=20.0)) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                if not total:
                    total = int(resp.headers.get("content-length") or 0)
                with open(partial, "wb") as fh:
                    async for chunk in resp.aiter_bytes(chunk_size):
                        fh.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                        yield {"status": "downloading", "downloaded": downloaded, "total": total}
    except Exception as exc:
        partial.unlink(missing_ok=True)
        yield {"status": "error", "message": str(exc), "downloaded": downloaded, "total": total}
        return

    if artifact.sha256 and hasher.hexdigest().lower() != artifact.sha256.lower():
        partial.unlink(missing_ok=True)
        yield {"status": "error", "message": "sha256 mismatch", "downloaded": downloaded, "total": total}
        return
    if total and downloaded != total:
        partial.unlink(missing_ok=True)
        yield {"status": "error", "message": "size mismatch", "downloaded": downloaded, "total": total}
        return

    partial.replace(dest)
    yield {"status": "done", "path": str(dest), "downloaded": downloaded, "total": total}

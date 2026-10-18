"""Test fixtures: application with a mocked engine (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.apikeys import ApiKey, clear_sessions, hash_key  # noqa: E402
from app.config import ConfigStore, Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_sessions():
    clear_sessions()
    yield
    clear_sessions()


class DummyEngine:
    """Fake engine: returns a short silence for any synthesis."""

    backend = "dummy"

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._loaded: set[str] = set()

    def available(self):
        return [
            {"family": f, "repo_id": f"dummy/{f}", "loaded": f in self._loaded,
             "backend": "chatterbox" if f.startswith("chatterbox") else "qwen3"}
            for f in ("custom_voice", "voice_design", "base", "chatterbox")
        ]

    def loaded_families(self):
        return list(self._loaded)

    def active_family(self):
        return next(iter(self._loaded), None)

    def ensure(self, family):
        self._loaded.add(family)
        return family

    def unload(self, family=None):
        if family is None:
            self._loaded.clear()
        else:
            self._loaded.discard(family)

    def speakers(self, family="custom_voice"):
        return ["Vivian", "Ryan"] if family == "custom_voice" else []

    def _cached_clone_prompt(self, handle, ref_audio, ref_text, x_vector_only=False):
        return {"ref_audio": ref_audio}

    def clear_clone_prompts(self):
        pass

    def languages(self, family="custom_voice"):
        return ["French", "English"]

    def design_reference(self, instruct, language):
        return np.zeros(self.sample_rate // 10, dtype=np.float32), self.sample_rate

    def synthesize_batch(self, family, items):
        samples = np.zeros(self.sample_rate // 10, dtype=np.float32)
        return [(samples, self.sample_rate) for _ in items]


def _settings(tmp_path: Path, **overrides) -> Settings:
    settings = Settings()
    settings.stt.enabled = False
    settings.llm.base_url = "http://127.0.0.1:9/v1"  # unreachable
    # Never write into the repository: the voice bank, outputs, logs and reference
    # clips go to the test's own tmp dir (Docker mounts the real ones).
    settings.paths.voices = str(tmp_path / "voices")
    settings.paths.output = str(tmp_path / "outputs")
    settings.paths.logs = str(tmp_path / "logs")
    settings.paths.reference_audio = str(tmp_path / "reference_audio")
    for section, values in overrides.items():
        target = getattr(settings, section)
        for key, value in values.items():
            setattr(target, key, value)
    return settings


@pytest.fixture()
def client(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.settings = _settings(tmp_path)
    app = create_app(store, DummyEngine())
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_client(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    secret = ApiKey(name="test", prefix="k-123", hash=hash_key("k-123"))
    store.settings = _settings(
        tmp_path, auth={"enabled": True, "password": "s3cret", "api_keys": [secret]}
    )
    app = create_app(store, DummyEngine())
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def permissive_media_client(tmp_path):
    """Client with remote/local reference audio explicitly enabled (sandbox dir)."""
    (tmp_path / "ref.wav").write_bytes(b"RIFF....WAVE")
    store = ConfigStore(tmp_path / "config.yaml")
    store.settings = _settings(
        tmp_path,
        audio={"allow_remote_media": True, "allow_local_ref_paths": True},
        paths={"reference_audio": str(tmp_path)},
    )
    app = create_app(store, DummyEngine())
    with TestClient(app) as c:
        yield c

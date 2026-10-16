import io

import numpy as np
import soundfile as sf
from conftest import DummyEngine
from fastapi.testclient import TestClient

from app.config import ConfigStore
from app.main import create_app


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "dummy"


def test_config_endpoint(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert "families" in body["models"]
    assert body["audio"]["sample_rate"] == 24000


def test_models_and_load(client):
    assert client.get("/api/models").status_code == 200
    r = client.post("/api/models/load", json={"family": "custom_voice"})
    assert r.status_code == 200
    assert "custom_voice" in r.json()["loaded"]


def test_voices(client):
    r = client.get("/api/voices?family=custom_voice")
    assert r.status_code == 200
    assert "Vivian" in r.json()["speakers"]


def test_voices_load_preset(client):
    r = client.post("/api/voices/load", json={"family": "custom_voice"})
    assert r.status_code == 200
    assert "custom_voice" in r.json()["loaded"]


def test_voices_load_avatar(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.settings.paths.voices = str(tmp_path / "voices")
    app = create_app(store, DummyEngine())
    with TestClient(app) as c:
        r = c.post(
            "/api/avatars",
            data={"name": "Narrator", "kind": "clone", "ref_text": "hello"},
            files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200, r.text
        aid = r.json()["id"]
        r = c.post("/api/voices/load", json={"avatar": aid})
        assert r.status_code == 200
        assert "base" in r.json()["loaded"]


def test_voices_load_unknown(client):
    assert client.post("/api/voices/load", json={"avatar": "nope"}).status_code == 404
    assert client.post("/api/voices/load", json={"family": "nope"}).status_code == 400


def test_tts_non_stream_wav(client):
    r = client.post("/api/tts", json={"text": "Bonjour le monde.", "format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"


def test_tts_stream_pcm(client):
    r = client.post("/api/tts", json={"text": "Bonjour. Comment ça va ?", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/pcm")
    assert len(r.content) > 0


def test_openai_speech(client):
    r = client.post("/v1/audio/speech", json={"input": "Hello", "voice": "Vivian"})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


def test_tts_requires_text(client):
    assert client.post("/api/tts", json={}).status_code == 400


def test_auth_blocks_and_allows(auth_client):
    assert auth_client.get("/api/config").status_code == 401
    assert auth_client.get("/health").status_code == 200
    # API key via header
    assert auth_client.get("/api/config", headers={"X-API-Key": "k-123"}).status_code == 200
    # Basic
    r = auth_client.get("/api/config", auth=("admin", "s3cret"))
    assert r.status_code == 200


def test_settings_update_partial_and_validation(client):
    store = client.app.state.store
    store.settings.llm.api_key = "secret"

    r = client.put("/api/settings", json={"llm": {"base_url": "http://x/v1"}})
    assert r.status_code == 200
    assert store.settings.llm.base_url == "http://x/v1"
    assert store.settings.llm.api_key == "secret"  # not reset by a partial update

    assert client.put(
        "/api/settings", json={"generation": {"temperature": "abc"}}
    ).status_code == 400


def _clone_body(ref_audio):
    return {"text": "hello", "family": "base", "ref_audio": ref_audio}


def test_ref_audio_defaults_are_restrictive(client):
    # Remote URLs and local paths are refused by default (SSRF / file disclosure).
    assert client.post("/api/tts", json=_clone_body("http://example.com/a.wav")).status_code == 400
    assert client.post("/api/tts", json=_clone_body("/etc/passwd")).status_code == 400
    assert client.post("/api/tts", json=_clone_body("file:///etc/passwd")).status_code == 400
    # Base64 data-URL is allowed.
    assert client.post("/api/tts", json=_clone_body("data:audio/wav;base64,AAAA")).status_code == 200


def test_ref_audio_permissive_and_traversal(permissive_media_client):
    c = permissive_media_client
    assert c.post("/api/tts", json=_clone_body("http://example.com/a.wav")).status_code == 200
    assert c.post("/api/tts", json=_clone_body("ref.wav")).status_code == 200
    # Path traversal outside the reference directory is refused.
    assert c.post("/api/tts", json=_clone_body("../../etc/passwd")).status_code == 400

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


def test_models_variants_listed(client):
    body = client.get("/api/models").json()
    ids = {v["id"] for v in body["variants"]}
    assert "custom_voice:0.6B" in ids and "base:0.6B" in ids
    assert "chatterbox:v3" in ids
    families = {f["family"] for f in body["families"]}
    assert "chatterbox" in families


def test_avatar_from_description(client):
    r = client.post("/api/avatars", data={"name": "Design", "prompt": "warm calm voice", "language": "English"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "clone"
    assert body["ref_text"]  # fingerprint transcript stored for ICL cloning


def test_fingerprint_endpoint(client):
    r = client.post("/api/avatars/fingerprint", data={"prompt": "deep male voice", "language": "English"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"


def test_clone_backend_switch(client):
    r = client.put("/api/settings", json={"models": {"clone_backend": "chatterbox",
                                                     "chatterbox": {"watermark": True}}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["models"]["clone_family"] == "chatterbox"
    assert body["models"]["chatterbox"]["watermark"] is True


def test_models_select_variant(client):
    r = client.post("/api/models/select", json={"variant": "base:0.6B"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["family"] == "base"
    assert "0.6B" in body["repo"]
    # The chosen repo becomes the registry entry for that family.
    reg = client.get("/api/config").json()
    assert reg["models"]["variants"]  # still listed
    assert client.post("/api/models/select", json={"variant": "nope"}).status_code == 400


def test_settings_update_default_size(client):
    r = client.put("/api/settings", json={"models": {"default_size": "0.6B"}})
    assert r.status_code == 200, r.text
    assert r.json()["models"]["default_size"] == "0.6B"
    assert client.get("/api/models").json()["default_size"] == "0.6B"


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


def test_avatar_clone_x_vector_defaults(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.settings.paths.voices = str(tmp_path / "voices")
    app = create_app(store, DummyEngine())
    with TestClient(app) as c:
        # No transcript -> speaker embedding only.
        r = c.post("/api/avatars", data={"name": "A", "kind": "clone"},
                   files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")})
        aid = r.json()["id"]
        assert r.json()["x_vector_only"] is True
        # Providing a transcript switches to ICL mode automatically.
        r = c.put(f"/api/avatars/{aid}", data={"ref_text": "hello world"})
        assert r.status_code == 200, r.text
        assert r.json()["x_vector_only"] is False
        # Explicit override back to x-vector only is honored.
        r = c.put(f"/api/avatars/{aid}", data={"x_vector_only": "true"})
        assert r.json()["x_vector_only"] is True


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

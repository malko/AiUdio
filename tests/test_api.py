import io
import json
import logging

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
    # The UI needs the registry to show which variant each family is set to.
    assert body["models"]["registry"]["base"]
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
    # `default_size` is a label now: the registry is what the service loads.
    r = client.put("/api/settings", json={"models": {"registry": {"base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}}})
    assert r.status_code == 200, r.text
    assert r.json()["models"]["registry"]["base"].endswith("1.7B-Base")
    # The size chosen for one family does not leak to the others.
    assert r.json()["models"]["registry"]["custom_voice"].endswith("CustomVoice")
    assert client.get("/api/config").json()["models"]["registry"]["base"].endswith("1.7B-Base")


def test_settings_update_vllm(client):
    # The experimental vLLM-Omni streaming backend is switchable from the API
    # (there is no UI field for it).
    r = client.put(
        "/api/settings",
        json={"vllm": {"enabled": True, "base_url": "http://tts-vllm:8091"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["vllm"]["enabled"] is True
    assert client.get("/api/config").json()["vllm"]["enabled"] is True


def test_settings_update_log_level(client):
    # Only `log_level` is editable in the `server` section, and it applies at once.
    root = logging.getLogger()
    previous = root.level
    try:
        r = client.put(
            "/api/settings",
            json={"server": {"log_level": "warning", "host": "1.2.3.4", "port": 9999}},
        )
        assert r.status_code == 200, r.text
        assert r.json()["server"]["log_level"] == "warning"
        assert root.level == logging.WARNING
        assert client.get("/api/config").json()["server"]["log_level"] == "warning"
        # Other server fields are ignored (the API does not rebind the listener).
        assert client.get("/api/config").json()["version"]  # still serving
    finally:
        root.setLevel(previous)
    # A payload with no editable section is rejected.
    assert client.put("/api/settings", json={"server": {}}).status_code == 400
    r = client.put(
        "/api/settings",
        json={"vllm": {"enabled": True, "base_url": "http://tts-vllm:8091"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["vllm"]["enabled"] is True
    assert client.get("/api/config").json()["vllm"]["enabled"] is True


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


def test_tts_rejects_out_of_range_generation(client):
    """Degenerate sampling values would assert in the sampler CUDA kernel.

    (`NaN` is covered at the schema level: JSON has no NaN literal, and echoing it
    back in a FastAPI 422 body would itself raise.)
    """
    for bad in ({"temperature": 0}, {"top_p": 0}, {"min_p": 2},
                {"cfg_weight": -1}, {"exaggeration": 5}):
        r = client.post("/api/tts", json={"text": "Hello", "generation": bad})
        assert r.status_code == 422, (bad, r.text)
    assert client.post(
        "/api/tts", json={"text": "Hello", "generation": {"temperature": 0.7}}
    ).status_code == 200


def test_tts_rejects_out_of_range_max_chars(client):
    """Beyond 600 chars one generation would truncate the audio silently."""
    assert client.post("/api/tts", json={"text": "Hello", "max_chars": 601}).status_code == 422
    assert client.post("/api/tts", json={"text": "Hello", "max_chars": 39}).status_code == 422
    assert client.post("/api/tts", json={"text": "Hello", "max_chars": 600}).status_code == 200


def test_tts_engine_value_error_is_a_400(client, monkeypatch):
    """Bad request content (unusable reference clip...) is not a 503."""
    def boom(family, items):  # noqa: ARG001
        raise ValueError("Reference audio is silent")

    monkeypatch.setattr(client.app.state.engine, "synthesize_batch", boom)
    r = client.post("/api/tts", json={"text": "Hello"})
    assert r.status_code == 400, r.text
    assert "silent" in r.json()["detail"]


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


def test_settings_clamp_max_chars(client):
    """The settings API cannot store a chunk size one generation cannot produce."""
    store = client.app.state.store
    r = client.put("/api/settings", json={"chunking": {"max_chars": 5000}})
    assert r.status_code == 200, r.text
    assert r.json()["chunking"]["max_chars"] == 600
    assert store.settings.chunking.max_chars == 600
    assert client.put(
        "/api/settings", json={"chunking": {"max_chars": 10}}
    ).json()["chunking"]["max_chars"] == 40


def test_text_cleanup_setting_and_fallbacks(client):
    """`text` is editable, and an unknown level falls back to the default."""
    r = client.put("/api/settings", json={"text": {"cleanup": "llm", "emoji": "name"}})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == {"cleanup": "llm", "emoji": "name", "code_blocks": "drop"}
    r = client.put("/api/settings", json={"text": {"cleanup": "nonsense"}})
    assert r.status_code == 200, r.text
    assert r.json()["text"]["cleanup"] == "off"     # unknown level -> default + warning


def test_api_key_text_policy(auth_client, monkeypatch):
    """A key can carry its own cleanup policy; PATCH changes it without rotating it."""
    calls = _spy_synth(monkeypatch, auth_client)
    admin = {"auth": ("admin", "s3cret")}
    raw = "## Titre 😀 **gras**"

    # The default policy leaves the text untouched.
    auth_client.post("/api/tts", json={"text": raw}, headers={"X-API-Key": "k-123"})
    assert calls[-1]["text"] == raw

    # The global setting applies to every client…
    r = auth_client.put(
        "/api/settings", json={"text": {"cleanup": "llm", "emoji": "name"}}, **admin
    )
    assert r.status_code == 200, r.text
    calls.clear()
    auth_client.post(
        "/api/tts", json={"text": raw, "language": "French"}, headers={"X-API-Key": "k-123"}
    )
    assert calls[-1]["text"] == "Titre sourire gras"

    # …unless the key overrides it (here: a machine client that wants it verbatim).
    created = auth_client.post(
        "/api/api-keys", json={"name": "batch", "text": {"cleanup": "off"}}, **admin
    ).json()
    assert created["text"] == {"cleanup": "off"}
    calls.clear()
    auth_client.post("/api/tts", json={"text": raw}, headers={"X-API-Key": created["key"]})
    assert calls[-1]["text"] == raw

    # PATCH: change the policy without invalidating the secret.
    r = auth_client.patch(
        f"/api/api-keys/{created['id']}", json={"text": {"emoji": "strip"}}, **admin
    )
    assert r.status_code == 200, r.text
    assert r.json()["text"] == {"emoji": "strip"}
    calls.clear()
    auth_client.post("/api/tts", json={"text": "Bonjour 😀"}, headers={"X-API-Key": created["key"]})
    assert calls[-1]["text"] == "Bonjour"      # inherits the global `llm` level

    # `text: null` clears the override again.
    assert auth_client.patch(
        f"/api/api-keys/{created['id']}", json={"text": None}, **admin
    ).json()["text"] is None


def test_api_key_patch_is_admin_only_and_validated(auth_client):
    admin = {"auth": ("admin", "s3cret")}
    created = auth_client.post("/api/api-keys", json={"name": "x"}, **admin).json()
    client_headers = {"X-API-Key": "k-123"}
    assert auth_client.patch(
        f"/api/api-keys/{created['id']}", json={"text": {"emoji": "strip"}},
        headers=client_headers,
    ).status_code == 403
    assert auth_client.patch(
        "/api/api-keys/nope", json={"name": "y"}, **admin
    ).status_code == 404
    assert auth_client.patch(
        f"/api/api-keys/{created['id']}", json={"name": "  "}, **admin
    ).status_code == 400
    assert auth_client.patch(
        f"/api/api-keys/{created['id']}", json={"text": "nonsense"}, **admin
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


ADMIN = ("admin", "s3cret")


def _new_session(auth_client):
    r = auth_client.post("/api/session", auth=ADMIN)
    assert r.status_code == 200, r.text
    return r.json()["key"]


def test_session_key_is_admin_bearer(auth_client):
    key = _new_session(auth_client)
    assert auth_client.get("/api/config", headers={"X-API-Key": key}).status_code == 200
    # The session key is admin-scoped: it may manage client keys.
    assert auth_client.get("/api/api-keys", headers={"X-API-Key": key}).status_code == 200


def test_session_requires_basic_not_api_key(auth_client):
    assert auth_client.post("/api/session", headers={"X-API-Key": "k-123"}).status_code == 401


def test_session_logout_revokes(auth_client):
    key = _new_session(auth_client)
    assert auth_client.post("/api/session/logout", headers={"X-API-Key": key}).json()["revoked"] is True
    assert auth_client.get("/api/config", headers={"X-API-Key": key}).status_code == 401


def test_api_keys_crud(auth_client):
    r = auth_client.post("/api/api-keys", json={"name": "mobile"}, auth=ADMIN)
    assert r.status_code == 200, r.text
    created = r.json()
    key = created["key"]
    assert key.startswith("aiud_")

    listing = auth_client.get("/api/api-keys", auth=ADMIN).json()
    assert listing["enabled"] is True
    assert any(k["id"] == created["id"] and k["name"] == "mobile" for k in listing["keys"])
    # The secret and the hash are never returned by the list.
    blob = json.dumps(listing)
    assert key not in blob
    assert '"hash"' not in blob

    assert auth_client.get("/api/config", headers={"X-API-Key": key}).status_code == 200
    assert auth_client.delete(f"/api/api-keys/{created['id']}", auth=ADMIN).status_code == 200
    assert auth_client.get("/api/config", headers={"X-API-Key": key}).status_code == 401


def test_expired_key_is_rejected(auth_client):
    r = auth_client.post(
        "/api/api-keys", json={"name": "old", "expires_in_days": -1}, auth=ADMIN
    )
    key = r.json()["key"]
    assert auth_client.get("/api/config", headers={"X-API-Key": key}).status_code == 401


def test_client_key_cannot_manage_keys(auth_client):
    headers = {"X-API-Key": "k-123"}
    assert auth_client.get("/api/api-keys", headers=headers).status_code == 403
    assert auth_client.post("/api/api-keys", json={"name": "x"}, headers=headers).status_code == 403


def test_api_keys_disabled(client):
    body = client.get("/api/api-keys").json()
    assert body == {"enabled": False, "keys": []}


def test_capabilities(client):
    body = client.get("/api/capabilities").json()
    assert body["name"] == "AiUdio Server"
    assert body["tts"]["websocket"] == "/ws/tts"
    assert body["chat"]["websocket"] == "/ws/chat"
    assert any(f["family"] == "chatterbox" for f in body["tts"]["families"])


def test_capabilities_expose_ws_protocols(client):
    body = client.get("/api/capabilities").json()
    # The WS protocols are not in OpenAPI: capabilities carries their message names.
    assert "session.config" in body["tts"]["ws"]["client"]
    assert "session.done" in body["tts"]["ws"]["server"]
    realtime = body["stt"]["realtime"]
    assert realtime["sample_rate"] == 16000 and realtime["encoding"] == "pcm16le"
    assert "input_audio_buffer.commit" in realtime["client"]
    assert "turn.cancelled" in body["chat"]["ws"]["server"]


def test_pcm_samples_announce_the_rate(client):
    # Raw PCM has no header: the rate is in X-Sample-Rate (streaming and not).
    r = client.post("/api/tts", json={"text": "Bonjour.", "format": "pcm"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/pcm")
    assert r.headers["x-sample-rate"] == "24000"
    s = client.post("/api/tts", json={"text": "Bonjour. Salut.", "stream": True})
    assert s.headers["x-sample-rate"] == "24000"


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_openapi_is_tagged(client):
    spec = client.get("/openapi.json").json()
    names = {t["name"] for t in spec.get("tags", [])}
    assert {"Service", "Access", "TTS", "STT", "Chat", "Models", "Avatars"} <= names
    assert spec["paths"]["/api/tts"]["post"]["tags"] == ["TTS"]


def test_streaming_tts_pcm(client):
    r = client.post("/api/tts", json={"text": "One. Two.", "stream": True, "format": "pcm"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/pcm")
    assert r.content


def test_metrics(client):
    client.get("/health")
    client.post("/api/tts", json={"text": "One. Two.", "stream": True, "format": "pcm"})
    body = client.get("/api/metrics").json()
    assert "ttfa_seconds" in body and "rtf" in body
    assert body["scheduler"]["batch_size"] >= 1
    assert body["counters"].get("tts_stream_sessions", 0) >= 1


def test_metrics_requires_admin(auth_client):
    assert auth_client.get("/api/metrics").status_code == 401
    assert auth_client.get("/api/metrics", auth=ADMIN).status_code == 200


def test_vllm_streaming_used(client, monkeypatch):
    from app import vllm_backend

    seen = {}

    async def fake_stream(base_url, payload, **kwargs):
        seen["payload"] = payload
        yield b"VLLMPCM1"
        yield b"VLLMPCM2"

    monkeypatch.setattr(vllm_backend, "stream_pcm", fake_stream)
    client.app.state.settings.vllm.enabled = True
    r = client.post("/api/tts", json={"text": "Hello there.", "stream": True, "format": "pcm"})
    assert r.status_code == 200
    assert r.content == b"VLLMPCM1VLLMPCM2"
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["response_format"] == "pcm"


def test_vllm_unavailable_is_a_503(client, monkeypatch):
    from app import vllm_backend

    async def failing_stream(base_url, payload, **kwargs):
        raise RuntimeError("down")
        yield b""  # pragma: no cover

    monkeypatch.setattr(vllm_backend, "stream_pcm", failing_stream)
    client.app.state.settings.vllm.enabled = True
    r = client.post("/api/tts", json={"text": "One. Two.", "stream": True, "format": "pcm"})
    # The selected engine is not swapped silently for the local one: the failure is
    # reported before the response starts.
    assert r.status_code == 503, r.text
    assert "vLLM-Omni" in r.json()["detail"]


def test_vllm_unavailable_falls_back_when_explicitly_enabled(client, monkeypatch):
    from app import vllm_backend

    async def failing_stream(base_url, payload, **kwargs):
        raise RuntimeError("down")
        yield b""  # pragma: no cover

    monkeypatch.setattr(vllm_backend, "stream_pcm", failing_stream)
    client.app.state.settings.vllm.enabled = True
    client.app.state.settings.vllm.fallback_to_local = True
    r = client.post("/api/tts", json={"text": "One. Two.", "stream": True, "format": "pcm"})
    assert r.status_code == 200, r.text
    assert r.content  # local engine, the documented escape hatch


def test_openai_voices_includes_avatars(client):
    r = client.post("/api/avatars", data={"name": "Narrator", "prompt": "calm", "language": "English"})
    assert r.status_code == 200, r.text
    body = client.get("/v1/audio/voices").json()
    assert any(a["name"] == "Narrator" for a in body["avatars"])


def test_vllm_health_probe_is_safe_when_unreachable(client):
    r = client.get("/api/vllm/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False          # nothing listening at tts-vllm:8091
    assert body["base_url"].startswith("http")


def test_vllm_service_control_is_off_by_default(client):
    r = client.get("/api/vllm/service")
    assert r.status_code == 200
    body = r.json()
    assert body["control"] is False and body["container"] == "tts-vllm"
    assert "error" in body  # explains how to enable it
    # Acting on a container without the control enabled is refused, not attempted.
    assert client.post("/api/vllm/service/start").status_code == 409
    assert client.post("/api/vllm/service/explode").status_code == 400
    r = client.put("/api/vllm/service/model", json={"model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"})
    assert r.status_code == 409


def test_vllm_model_must_be_a_qwen3_tts_checkpoint(client):
    r = client.put("/api/vllm/service/model", json={"model": "meta-llama/Llama-3"})
    assert r.status_code == 400
    assert "CustomVoice" in r.json()["detail"]
    assert client.put("/api/vllm/service/model", json={}).status_code == 400


def test_voice_bank_hygiene_reports_and_cleans_orphans(client, tmp_path):
    import json
    from pathlib import Path

    root = Path(client.app.state.settings.paths.voices) / "avatars"
    root.mkdir(parents=True, exist_ok=True)

    # Two voices that share a name (a duplicate), each with its clip…
    for avatar_id in ("bugatti", "bugatti-3f2a1c"):
        (root / f"{avatar_id}.json").write_text(
            json.dumps({"id": avatar_id, "name": "Bugatti", "kind": "clone"}), encoding="utf-8"
        )
        (root / f"{avatar_id}.wav").write_bytes(b"RIFF....WAVE")
    # …one stranded clip and one stranded photo (their entry is gone)…
    (root / "vieux-test.wav").write_bytes(b"RIFF....WAVE")
    (root / "vieux-test.png").write_bytes(b"\x89PNG")
    # …and an entry whose clip disappeared.
    (root / "casse.json").write_text(json.dumps({"id": "casse", "name": "Cassé"}), encoding="utf-8")

    hygiene = client.get("/api/avatars/hygiene").json()
    assert hygiene["avatars"] == 3
    assert hygiene["files"] == 7  # 3 .json, 3 .wav, 1 .png
    assert hygiene["by_kind"] == {"json": 3, "wav": 3, "png": 1}
    assert [d["name"] for d in hygiene["duplicates"]] == ["bugatti"]
    assert sorted(o["name"] for o in hygiene["orphans"]) == ["vieux-test.png", "vieux-test.wav"]
    assert hygiene["broken"] == ["casse"]  # no clip any more
    assert hygiene["total_bytes"] > 0

    cleaned = client.post("/api/avatars/hygiene/clean").json()
    assert sorted(cleaned["removed"]) == ["vieux-test.png", "vieux-test.wav"]
    # Voices are never touched, duplicates included.
    after = cleaned["hygiene"]
    assert after["avatars"] == 3 and after["files"] == 5
    assert after["orphans"] == []
    assert (root / "bugatti.wav").is_file() and (root / "bugatti-3f2a1c.json").is_file()


def test_multi_text_wav_has_a_single_header(client):
    r = client.post("/api/tts", json={"text": ["One. Two.", "Three."], "format": "wav"})
    assert r.status_code == 200, r.text
    assert r.content[:4] == b"RIFF"
    # One WAV file, not several concatenated blobs (which only play the first).
    assert r.content.count(b"RIFF") == 1


def _spy_synth(monkeypatch, client):
    """Count the texts that reach the engine for one request."""
    calls: list = []
    engine = client.app.state.engine
    real = engine.synthesize_batch

    def spy(family, items):
        calls.extend(items)
        return real(family, items)

    monkeypatch.setattr(engine, "synthesize_batch", spy)
    return calls


def test_non_streaming_chunks_long_text(client, monkeypatch):
    """`chunking` is honoured outside streaming too (one engine call per chunk)."""
    calls = _spy_synth(monkeypatch, client)
    text = ("a" * 30) + ". " + ("b" * 30) + "."

    r = client.post("/api/tts", json={"text": text, "max_chars": 40, "format": "wav"})
    assert r.status_code == 200, r.text
    assert r.content[:4] == b"RIFF" and r.content.count(b"RIFF") == 1
    assert len(calls) == 2                      # two sentences, split by max_chars
    audio, sr = sf.read(io.BytesIO(r.content))
    # DummyEngine returns 100 ms per chunk; the chunks are joined with a silence.
    assert len(audio) == 2 * int(0.1 * sr) + int(0.12 * sr)

    calls.clear()
    r2 = client.post(
        "/api/tts", json={"text": text, "max_chars": 40, "chunking": False, "format": "wav"}
    )
    assert r2.status_code == 200 and len(calls) == 1   # opt out keeps it in one piece


def test_non_streaming_chunking_keeps_pcm_header(client, monkeypatch):
    _spy_synth(monkeypatch, client)
    text = ("a" * 30) + ". " + ("b" * 30) + "."
    r = client.post("/api/tts", json={"text": text, "max_chars": 40, "format": "pcm"})
    assert r.status_code == 200, r.text
    assert r.headers["X-Sample-Rate"] == "24000"
    assert len(r.content) > 0


def test_non_admin_key_cannot_change_settings(auth_client):
    headers = {"X-API-Key": "k-123"}  # a plain (non-admin) client key
    assert auth_client.put(
        "/api/settings", json={"llm": {"base_url": "http://x/v1"}}, headers=headers
    ).status_code == 403
    assert auth_client.post(
        "/api/avatars", data={"name": "x", "prompt": "p"}, headers=headers
    ).status_code == 403
    assert auth_client.post(
        "/api/models/select", json={"variant": "base:0.6B"}, headers=headers
    ).status_code == 403
    assert auth_client.post(
        "/api/voices/load", json={"family": "custom_voice"}, headers=headers
    ).status_code == 403
    # The admin (HTTP Basic here) still can.
    assert auth_client.put(
        "/api/settings", json={"llm": {"base_url": "http://x/v1"}}, auth=ADMIN
    ).status_code == 200


def test_cors_preflight_is_not_blocked_by_auth(auth_client):
    # A browser preflight carries no credentials: CORS (outermost) answers it.
    r = auth_client.options(
        "/api/tts",
        headers={"Origin": "http://app.example", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code < 400, r.text
    assert "access-control-allow-origin" in {k.lower() for k in r.headers}


def test_enabling_vllm_frees_every_local_model(client):
    # The vLLM stages profile their KV pool against the *free* memory: selecting the
    # service must not leave a local family (e.g. a Base avatar) holding VRAM.
    client.post("/api/models/load", json={"family": "custom_voice"})
    client.post("/api/models/load", json={"family": "base"})
    assert set(client.app.state.engine.loaded_families()) == {"custom_voice", "base"}
    r = client.put("/api/settings", json={"vllm": {"enabled": True}})
    assert r.status_code == 200, r.text
    assert client.app.state.engine.loaded_families() == []

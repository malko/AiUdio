"""STT input language: the configured default, its normalization and the WS merge."""

import io

from conftest import DummyEngine

from app.config import ConfigStore, Settings
from app.main import (
    _merge_stt_session,
    _stt_language_code,
    _stt_session_defaults,
    create_app,
)


def test_stt_language_code_normalizes_names_and_codes():
    assert _stt_language_code("French") == "fr"
    assert _stt_language_code("fr") == "fr"
    assert _stt_language_code("  EN ") == "en"
    assert _stt_language_code("") == ""
    assert _stt_language_code(None) == ""
    # Unknown values are passed through (the service decides).
    assert _stt_language_code("fr-FR") == "fr-FR"
    assert _stt_language_code("Klingon") == "Klingon"


def test_stt_session_defaults_only_carry_a_configured_language():
    settings = Settings()
    assert _stt_session_defaults(settings) == {}
    settings.stt.language = "French"
    assert _stt_session_defaults(settings) == {"language": "fr"}


def test_merge_stt_session_fills_but_never_overrides_the_client():
    defaults = {"language": "fr"}
    event = {"type": "session.update", "session": {"sample_rate": 16000}}
    merged = _merge_stt_session(event, defaults)
    assert merged["session"] == {"sample_rate": 16000, "language": "fr"}
    # The client is explicit: it wins.
    mine = {"type": "session.update", "session": {"language": "en"}}
    assert _merge_stt_session(mine, defaults)["session"]["language"] == "en"
    # No default configured and nothing to normalize: the event is untouched.
    assert _merge_stt_session(event, {}) is event
    # A name sent by the client is normalized to a code, even without a default.
    named = {"type": "session.update", "session": {"language": "French"}}
    assert _merge_stt_session(named, {})["session"]["language"] == "fr"


class _FakeResponse:
    status_code = 200
    content = b'{"text": "ok"}'
    headers = {"content-type": "application/json"}


def _stt_client(tmp_path, monkeypatch, **stt):
    from fastapi.testclient import TestClient

    from app import main as main_mod

    captured = {}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, files=None, data=None, headers=None, **kwargs):
            captured.update(data or {})
            return _FakeResponse()

    monkeypatch.setattr(main_mod.httpx, "AsyncClient", lambda *a, **k: _FakeClient())

    store = ConfigStore(tmp_path / "config.yaml")
    settings = Settings()
    settings.paths.voices = str(tmp_path / "voices")
    settings.paths.reference_audio = str(tmp_path / "ref")
    settings.stt.enabled = True
    for key, value in stt.items():
        setattr(settings.stt, key, value)
    store.settings = settings
    return TestClient(create_app(store, DummyEngine())), captured


def _post(client, data=None):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", io.BytesIO(b"RIFF....WAVE"), "audio/wav")},
        data=data or {},
    )


def test_transcription_uses_the_configured_language_default(tmp_path, monkeypatch):
    client, captured = _stt_client(tmp_path, monkeypatch, language="French")
    with client:
        assert _post(client).status_code == 200
    assert captured["language"] == "fr"  # name normalized to a code


def test_transcription_client_language_wins(tmp_path, monkeypatch):
    client, captured = _stt_client(tmp_path, monkeypatch, language="French")
    with client:
        _post(client, {"language": "English"})
    assert captured["language"] == "en"


def test_transcription_without_any_language_sends_none(tmp_path, monkeypatch):
    client, captured = _stt_client(tmp_path, monkeypatch)  # language="" → auto
    with client:
        _post(client)
    assert "language" not in captured
    assert captured["response_format"] == "json"

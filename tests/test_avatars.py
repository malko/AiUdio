import pathlib

import pytest
from fastapi import HTTPException

from app.avatars import Avatar, AvatarStore
from app.config import Settings
from app.main import _item_from_request
from app.schemas import TTSRequest


def test_avatar_store_roundtrip(tmp_path):
    store = AvatarStore(tmp_path / "avatars")
    avatar = Avatar(id=store.new_id("Narrator"), name="Narrator", kind="design", prompt="calm voice")
    store.save(avatar)
    store.wav_path(avatar.id).write_bytes(b"RIFF")

    listed = store.list()
    assert [a.name for a in listed] == ["Narrator"]
    assert store.get(avatar.id).prompt == "calm voice"

    store.delete(avatar.id)
    assert store.get(avatar.id) is None


def test_item_from_request_uses_avatar(tmp_path):
    settings = Settings()
    settings.paths.voices = str(tmp_path / "voices")
    store = AvatarStore(pathlib.Path(settings.paths.voices) / "avatars")
    avatar = Avatar(id="narrator", name="Narrator", kind="clone", ref_text="hello world")
    store.save(avatar)
    store.wav_path(avatar.id).write_bytes(b"RIFF....WAVE")

    item = _item_from_request(settings, TTSRequest(text="hi", avatar="narrator"), "hi")
    assert item["family"] == "base"
    assert item["ref_audio"].endswith("narrator.wav")
    assert item["ref_text"] == "hello world"


def test_item_from_request_unknown_avatar(tmp_path):
    settings = Settings()
    settings.paths.voices = str(tmp_path / "voices")
    with pytest.raises(HTTPException):
        _item_from_request(settings, TTSRequest(text="hi", avatar="missing"), "hi")


def test_item_from_request_uses_x_vector_mode(tmp_path):
    settings = Settings()
    settings.paths.voices = str(tmp_path / "voices")
    store = AvatarStore(pathlib.Path(settings.paths.voices) / "avatars")
    store.save(Avatar(id="narrator", name="Narrator", kind="clone", x_vector_only=True))
    store.wav_path("narrator").write_bytes(b"RIFF....WAVE")
    item = _item_from_request(settings, TTSRequest(text="hi", avatar="narrator"), "hi")
    assert item["x_vector_only_mode"] is True


def test_item_from_request_uses_chatterbox_clone_backend(tmp_path):
    settings = Settings()
    settings.models.clone_backend = "chatterbox"
    settings.paths.voices = str(tmp_path / "voices")
    store = AvatarStore(pathlib.Path(settings.paths.voices) / "avatars")
    store.save(Avatar(id="narrator", name="Narrator", kind="clone", ref_text="hello world"))
    store.wav_path("narrator").write_bytes(b"RIFF....WAVE")
    item = _item_from_request(settings, TTSRequest(text="hi", avatar="narrator"), "hi")
    assert item["family"] == "chatterbox"


def test_avatar_explicit_family_overrides_clone_backend(tmp_path):
    settings = Settings()  # clone_backend defaults to qwen3
    settings.paths.voices = str(tmp_path / "voices")
    store = AvatarStore(pathlib.Path(settings.paths.voices) / "avatars")
    store.save(Avatar(id="narrator", name="Narrator", kind="clone", ref_text="hello world"))
    store.wav_path("narrator").write_bytes(b"RIFF....WAVE")
    item = _item_from_request(
        settings, TTSRequest(text="hi", avatar="narrator", family="chatterbox"), "hi"
    )
    assert item["family"] == "chatterbox"


def test_voice_can_be_selected_by_id(tmp_path):
    # The voice-bank id works through `voice`/`speaker` too (OpenAI route).
    settings = Settings()
    settings.paths.voices = str(tmp_path / "voices")
    store = AvatarStore(pathlib.Path(settings.paths.voices) / "avatars")
    store.save(Avatar(id="narrator", name="Narrator", kind="clone", ref_text="hi"))
    store.wav_path("narrator").write_bytes(b"RIFF....WAVE")
    item = _item_from_request(settings, TTSRequest(text="hi", voice="narrator"), "hi")
    assert item["family"] == "base"
    assert item["ref_audio"].endswith("narrator.wav")


def test_per_request_clone_backend_overrides_default(tmp_path):
    settings = Settings()  # admin default = qwen3
    settings.paths.voices = str(tmp_path / "voices")
    store = AvatarStore(pathlib.Path(settings.paths.voices) / "avatars")
    store.save(Avatar(id="narrator", name="Narrator", kind="clone", ref_text="hi"))
    store.wav_path("narrator").write_bytes(b"RIFF....WAVE")
    item = _item_from_request(
        settings, TTSRequest(text="hi", avatar="narrator", clone_backend="chatterbox"), "hi"
    )
    assert item["family"] == "chatterbox"
    # `engine` is accepted as an alias, and qwen aliases map back to base.
    item = _item_from_request(
        settings, TTSRequest(text="hi", avatar="narrator", engine="qwen3"), "hi"
    )
    assert item["family"] == "base"


def test_ref_audio_without_family_uses_clone_engine(tmp_path):
    settings = Settings()
    item = _item_from_request(
        settings, TTSRequest(text="hi", ref_audio="data:audio/wav;base64,AAAA",
                             clone_backend="chatterbox"), "hi"
    )
    assert item["family"] == "chatterbox"


def test_avatar_personality_roundtrip(tmp_path):
    store = AvatarStore(tmp_path / "avatars")
    avatar = Avatar(id="ai", name="AI", personality="You are a witty assistant.")
    store.save(avatar)
    assert store.get("ai").personality == "You are a witty assistant."

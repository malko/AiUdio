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


def test_avatar_personality_roundtrip(tmp_path):
    store = AvatarStore(tmp_path / "avatars")
    avatar = Avatar(id="ai", name="AI", personality="You are a witty assistant.")
    store.save(avatar)
    assert store.get("ai").personality == "You are a witty assistant."

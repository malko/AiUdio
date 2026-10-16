import os

from app.config import ConfigStore, Settings


def test_settings_roundtrip(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.settings.generation.temperature = 0.42
    store.settings.ui.title = "Test UI"
    store.save()

    reloaded = ConfigStore(tmp_path / "config.yaml")
    assert reloaded.settings.generation.temperature == 0.42
    assert reloaded.settings.ui.title == "Test UI"


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.local/v1")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_API_KEYS", "key1,key2")
    store = ConfigStore(tmp_path / "config.yaml")
    assert store.settings.llm.base_url == "http://llm.local/v1"
    assert store.settings.auth.enabled is True
    assert store.settings.auth.api_keys == ["key1", "key2"]


def test_update_section(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(chunking={"max_chars": 321})
    assert store.settings.chunking.max_chars == 321
    # persisted to disk
    assert ConfigStore(tmp_path / "config.yaml").settings.chunking.max_chars == 321


def test_update_nested_section_preserves_other_fields(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.settings.llm.api_key = "secret"
    store.settings.llm.temperature = 0.42
    store.save()

    store.update(llm={"base_url": "http://llm.local/v1"})

    assert store.settings.llm.base_url == "http://llm.local/v1"
    assert store.settings.llm.api_key == "secret"        # not reset
    assert store.settings.llm.temperature == 0.42        # not reset


def test_update_invalid_section_raises(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    try:
        store.update(generation={"temperature": "not-a-number"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid settings")

import os

from app.config import ConfigStore, Settings
from app.languages import to_iso, to_name


def test_language_mapping():
    assert to_iso("French") == "fr"
    assert to_iso("fr") == "fr"
    assert to_iso("Auto") is None
    assert to_iso(None) is None
    assert to_name("zh") == "Chinese"


def test_backend_for_family():
    models = Settings().models
    assert models.backend_for("custom_voice") == "qwen3"
    assert models.backend_for("chatterbox") == "chatterbox"
    assert models.backend_for("chatterbox_turbo") == "chatterbox"


def test_clone_backend_persisted(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(models={"clone_backend": "chatterbox"})
    assert ConfigStore(tmp_path / "config.yaml").settings.models.clone_backend == "chatterbox"


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
    store = ConfigStore(tmp_path / "config.yaml")
    assert store.settings.llm.base_url == "http://llm.local/v1"
    assert store.settings.auth.enabled is True


def test_legacy_plaintext_keys_are_hashed(tmp_path):
    from app.apikeys import hash_key

    path = tmp_path / "config.yaml"
    path.write_text("auth:\n  api_keys:\n  - legacy-key\n", encoding="utf-8")
    store = ConfigStore(path)
    (record,) = store.settings.auth.api_keys
    assert record.hash == hash_key("legacy-key")
    assert record.prefix == "legacy-key"[:12]


def test_api_key_persisted_hashed(tmp_path):
    from app.apikeys import new_key

    store = ConfigStore(tmp_path / "config.yaml")
    raw, record = new_key(name="mobile")
    store.update(auth={"api_keys": [record.model_dump()]})
    reloaded = ConfigStore(tmp_path / "config.yaml")
    (saved,) = reloaded.settings.auth.api_keys
    assert saved.hash and raw not in reloaded.path.read_text(encoding="utf-8")


def test_seeds_runtime_config_from_defaults(tmp_path, monkeypatch):
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text("generation:\n  temperature: 0.5\n", encoding="utf-8")
    target = tmp_path / "state" / "config.yaml"
    monkeypatch.setenv("CONFIG_DEFAULTS", str(defaults))

    store = ConfigStore(target)
    assert target.exists()
    assert store.settings.generation.temperature == 0.5

    # Once the state file exists, it wins (defaults are only a seed).
    target.write_text("generation:\n  temperature: 0.9\n", encoding="utf-8")
    assert ConfigStore(target).settings.generation.temperature == 0.9


def test_no_defaults_no_seed(tmp_path, monkeypatch):
    monkeypatch.delenv("CONFIG_DEFAULTS", raising=False)
    store = ConfigStore(tmp_path / "config.yaml")
    assert not store.path.exists()
    assert store.settings.generation.temperature == Settings().generation.temperature


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


def test_default_size_sets_registry(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(models={"default_size": "0.6B"})
    assert store.settings.models.registry["custom_voice"].endswith("0.6B-CustomVoice")
    assert store.settings.models.registry["base"].endswith("0.6B-Base")
    # VoiceDesign has no 0.6B variant -> falls back to its current repo.
    assert store.settings.models.registry["voice_design"].endswith("1.7B-VoiceDesign")

    reloaded = ConfigStore(tmp_path / "config.yaml")
    assert reloaded.settings.models.default_size == "0.6B"
    assert reloaded.settings.models.registry["base"].endswith("0.6B-Base")

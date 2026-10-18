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


def test_residency_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_RESIDENCY", "multi")
    assert ConfigStore(tmp_path / "a.yaml").settings.models.residency == "multi"
    # An unknown value is ignored (the file/default wins), not an error.
    monkeypatch.setenv("TTS_RESIDENCY", "bogus")
    assert ConfigStore(tmp_path / "b.yaml").settings.models.residency == "single"


def test_text_cleanup_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXT_CLEANUP", "llm")
    monkeypatch.setenv("TEXT_EMOJI", "name")
    monkeypatch.setenv("TEXT_CODE_BLOCKS", "announce")
    settings = ConfigStore(tmp_path / "a.yaml").settings
    assert settings.text.cleanup == "llm"
    assert settings.text.emoji == "name"
    assert settings.text.code_blocks == "announce"
    # An unknown value is ignored (the file/default wins), not an error.
    monkeypatch.setenv("TEXT_CLEANUP", "bogus")
    assert ConfigStore(tmp_path / "b.yaml").settings.text.cleanup == "off"


def test_text_cleanup_is_persisted(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(text={"cleanup": "light", "emoji": "strip"})
    reloaded = ConfigStore(tmp_path / "config.yaml").settings
    assert (reloaded.text.cleanup, reloaded.text.emoji) == ("light", "strip")


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


def test_per_family_size_survives_reload(tmp_path):
    # The registry is authoritative: picking 1.7B for the clone family must not be
    # overridden by the (informational) global `default_size`.
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(models={"registry": {"base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}})
    assert store.settings.models.registry["base"].endswith("1.7B-Base")
    # Other families are untouched by the partial registry patch.
    assert store.settings.models.registry["custom_voice"].endswith("CustomVoice")

    reloaded = ConfigStore(tmp_path / "config.yaml")
    assert reloaded.settings.models.registry["base"].endswith("1.7B-Base")
    assert reloaded.settings.models.registry["voice_design"].endswith("VoiceDesign")


def test_update_merges_nested_sections(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(models={"chatterbox": {"python": "/opt/cb/venv/bin/python", "t3_model": "v3"}})
    store.update(models={"chatterbox": {"watermark": True}})
    cb = store.settings.models.chatterbox
    assert cb.watermark is True
    assert cb.python == "/opt/cb/venv/bin/python"  # not reset by the partial update
    assert cb.t3_model == "v3"


def test_vllm_family_selects_the_served_checkpoint():
    from app.config import Settings

    assert Settings().vllm.family == "custom_voice"          # preset speakers
    assert Settings(vllm={"family": "Base"}).vllm.family == "base"
    assert Settings(vllm={"family": "clone"}).vllm.family == "base"
    # Anything else falls back to the preset (CustomVoice) family.
    assert Settings(vllm={"family": "bogus"}).vllm.family == "custom_voice"


def test_stt_language_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("STT_LANGUAGE", "French")
    monkeypatch.setenv("STT_LANGUAGES", "fr, en ,")
    store = ConfigStore(tmp_path / "config.yaml")
    assert store.settings.stt.language == "French"
    assert store.settings.stt.languages == ["fr", "en"]

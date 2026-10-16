from app.config import Settings
from app.engine import ModelHandle, TTSEngine


class _FakeModel:
    def get_supported_speakers(self):
        return ["Vivian", "Ryan"]

    def get_supported_languages(self):
        return ["English", "French"]


def test_default_speaker_is_stable():
    engine = TTSEngine(Settings())
    handle = ModelHandle(family="custom_voice", repo_id="dummy", model=_FakeModel())
    assert engine._default_speaker(handle) == "Vivian"
    assert engine._default_speaker(handle) == "Vivian"  # cached, no recompute


def test_speakers_empty_when_model_not_loaded():
    engine = TTSEngine(Settings())
    # Querying voices must not trigger a model load (and thus a download).
    assert engine.speakers("custom_voice") == []
    assert engine.languages("custom_voice") == []

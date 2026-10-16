import numpy as np

from app.config import Settings
from app.engine import ModelHandle, TTSEngine


class _FakeModel:
    def get_supported_speakers(self):
        return ["Vivian", "Ryan"]

    def get_supported_languages(self):
        return ["English", "French"]


class _FakeDesign:
    def __init__(self):
        self.calls = []

    def generate_voice_design(self, text, language, instruct, **kwargs):
        self.calls.append({"text": text, "language": language, "instruct": instruct})
        return [np.zeros(8, dtype=np.float32)], 24000


class _FakeBase:
    def __init__(self):
        self.prompts_built = 0
        self.used_prompt = None

    def create_voice_clone_prompt(self, ref_audio, ref_text):
        self.prompts_built += 1
        return [{"ref_text": ref_text}]

    def generate_voice_clone(self, text, language, voice_clone_prompt=None, **kwargs):
        self.used_prompt = voice_clone_prompt
        return [np.zeros(8, dtype=np.float32) for _ in text], 24000


def test_default_speaker_is_stable():
    engine = TTSEngine(Settings())
    handle = ModelHandle(family="custom_voice", repo_id="dummy", model=_FakeModel())
    assert engine._default_speaker(handle) == "Vivian"
    assert engine._default_speaker(handle) == "Vivian"  # cached, no recompute


def test_speakers_defaults_when_model_not_loaded():
    engine = TTSEngine(Settings())
    # Querying voices must not trigger a model load (and thus a download), but
    # the built-in CustomVoice speakers are always known.
    speakers = engine.speakers("custom_voice")
    assert "Vivian" in speakers and "Ryan" in speakers
    assert engine.speakers("voice_design") == []
    assert engine.languages("custom_voice") == []


def test_voice_design_stabilized_via_clone_and_cached():
    engine = TTSEngine(Settings())
    models = {"voice_design": _FakeDesign(), "base": _FakeBase()}
    engine._load = lambda family: ModelHandle(family=family, repo_id="x", model=models[family])

    item = {"text": "One.", "instruct": "warm voice", "language": "English"}
    out = engine.synthesize_batch("voice_design", [item, {**item, "text": "Two."}])
    assert len(out) == 2
    design = models["voice_design"]
    base = models["base"]
    assert design.calls and design.calls[0]["instruct"] == "warm voice"
    assert base.prompts_built == 1
    assert base.used_prompt is not None

    # Same description -> prompt reused (no new reference clip).
    engine.synthesize_batch("voice_design", [{**item, "text": "Three."}])
    assert base.prompts_built == 1

    # Changed description -> new reference clip and prompt.
    engine.synthesize_batch("voice_design", [{**item, "instruct": "angry voice"}])
    assert base.prompts_built == 2

import base64
import io
import json
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from app import chatterbox_backend
from app.chatterbox_backend import ChatterboxClient, resolve_python
from app.chatterbox_worker import (
    _accepts,
    _char_budget,
    _detect_t3,
    _filter_kwargs,
    _is_fatal_error,
    _split_for_model,
    _split_text,
)
from app.config import Settings
from app.engine import ModelHandle, TTSEngine
from app.schemas import GenerationParams

# Minimal stand-in for the real worker: serves the JSON protocol and dies with a
# fatal (poisoned-context) reply on `synth`, like a CUDA device-side assert.
_FAKE_WORKER = '''
import json
import sys


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    cmd = req.get("cmd")
    if cmd == "load":
        reply({"ok": True})
    elif cmd == "synth":
        reply({"ok": False, "error": "CUDA error: device-side assert triggered",
               "trace": "", "fatal": True})
        sys.exit(1)
    elif cmd == "shutdown":
        reply({"ok": True})
        sys.exit(0)
    else:
        reply({"ok": True, "languages": ["en"]})
'''


def _valid_clip(seconds=1.0, sr=24000, freq=220.0):
    n = int(seconds * sr)
    return (0.5 * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)


def _data_url(samples, sr=24000):
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _fake_client(tmp_path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER)
    monkeypatch.setattr(chatterbox_backend, "_WORKER", script)
    return ChatterboxClient(
        sys.executable, "chatterbox", "ResembleAI/chatterbox", device="cpu",
    )


class _FakeIds:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class _FakeTokenizer:
    """One token per word, like a grapheme tokenizer is ~one per character."""

    def text_to_tokens(self, text, language_id=None):  # noqa: ARG002
        return _FakeIds(len(text.split()))


class _CountingTokenizer(_FakeTokenizer):
    """Counts how many times the (potentially heavy) tokenizer is invoked."""

    def __init__(self):
        self.calls = 0

    def text_to_tokens(self, text, language_id=None):
        self.calls += 1
        return super().text_to_tokens(text, language_id=language_id)


class _FakeModel:
    """Stands in for ChatterboxMultilingualTTS (no torch, no GPU)."""

    def __init__(self, limit=10):
        self.t3 = types.SimpleNamespace(
            text_pos_emb=object(),  # a learned positional table -> a limit exists
            hp=types.SimpleNamespace(max_text_tokens=limit),
        )
        self.tokenizer = _FakeTokenizer()
        self.sr = 24000
        self.calls = []

    def generate(self, text, **kwargs):  # noqa: ARG002
        self.calls.append(text)
        return np.zeros(len(text.split()) * 100, dtype=np.float32)


class _SpyClient:
    """Records the reference clip the engine hands to the (absent) worker."""

    def __init__(self):
        self.ref_path = None

    def synthesize(self, texts, *, language, audio_prompt, params):  # noqa: ARG002
        self.ref_path = audio_prompt
        assert audio_prompt and os.path.isfile(audio_prompt)
        return [(np.zeros(240, dtype=np.float32), 24000) for _ in texts]


def test_accepts_adapts_to_signature():
    def without_t3(device):  # noqa: ARG001
        return None

    def with_t3(device, t3_model=None):  # noqa: ARG001
        return None

    def with_kwargs(device, **kwargs):  # noqa: ARG001
        return None

    assert _accepts(without_t3, "t3_model") is False
    assert _accepts(with_t3, "t3_model") is True
    assert _accepts(with_kwargs, "anything") is True
    assert _filter_kwargs(without_t3, {"device": "cpu", "t3_model": "v3"}) == {"device": "cpu"}
    assert _filter_kwargs(with_kwargs, {"t3_model": "v3"}) == {"t3_model": "v3"}


def test_detect_t3(tmp_path):
    (tmp_path / "t3_mtl23ls_v2.safetensors").write_bytes(b"x")
    assert _detect_t3(tmp_path, "v3") == "t3_mtl23ls_v2.safetensors"
    (tmp_path / "t3_mtl23ls_v3.safetensors").write_bytes(b"x")
    assert _detect_t3(tmp_path, "v3") == "t3_mtl23ls_v3.safetensors"


def test_resolve_python_missing():
    assert resolve_python("") in (None, "/opt/cb/venv/bin/python")
    assert resolve_python("/nonexistent/python") in (None, "/opt/cb/venv/bin/python")


def test_chatterbox_params_selection():
    engine = TTSEngine(Settings())
    params = engine._chatterbox_params(
        {"temperature": 0.8, "top_p": 0.9, "top_k": 50, "exaggeration": 0.6,
         "cfg_weight": 0.4, "min_p": 0.02, "max_new_tokens": 100}
    )
    assert params["temperature"] == 0.8
    assert params["exaggeration"] == 0.6
    assert "top_k" not in params        # Qwen-only
    assert "max_new_tokens" not in params


def test_chatterbox_params_are_clamped():
    """Degenerate knobs would make the sampler assert; they are clamped instead."""
    engine = TTSEngine(Settings())
    params = engine._chatterbox_params({
        "temperature": 0.0, "top_p": 0.0, "min_p": 2.0, "cfg_weight": -1.0,
        "exaggeration": 10.0, "repetition_penalty": float("inf"),
    })
    assert params["temperature"] > 0
    assert params["top_p"] > 0
    assert params["min_p"] == 1.0
    assert params["cfg_weight"] == 0.0
    assert params["exaggeration"] == 2.0
    assert params["repetition_penalty"] == 1.2     # non-finite -> Chatterbox default


def test_generation_params_reject_degenerate_sampling():
    GenerationParams(temperature=0.8, top_p=0.9, min_p=0.02, exaggeration=0.6)
    for bad in ({"temperature": 0}, {"temperature": float("nan")}, {"top_p": 0},
                {"top_p": 1.5}, {"min_p": 1.5}, {"exaggeration": 3},
                {"cfg_weight": -0.1}, {"repetition_penalty": 0}):
        with pytest.raises(ValidationError):
            GenerationParams(**bad)


def test_is_fatal_error():
    assert _is_fatal_error(RuntimeError("CUDA error: device-side assert triggered"))
    assert _is_fatal_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not _is_fatal_error(ValueError("Unsupported language_id 'xx'"))
    assert not _is_fatal_error(RuntimeError("Torch not compiled with CUDA enabled"))


def test_split_text_keeps_every_piece_within_budget():
    text = "one two three four five six seven eight nine ten"
    pieces = _split_text(text, 12)
    assert len(pieces) > 1
    assert all(len(p) <= 12 for p in pieces)
    # Nothing lost or duplicated (a character budget may cut mid-word).
    assert "".join(pieces).replace(" ", "") == "".join(text.split())


def test_split_text_prefers_boundaries():
    """The worker must not cut inside a word either (same rules as the gateway)."""
    words = " ".join(f"mot{i}" for i in range(60))
    pieces = _split_text(words, 40)
    assert all(len(p) <= 40 for p in pieces)
    assert " ".join(pieces).split() == words.split()      # words intact

    cjk = "这是一个测试。" * 20
    pieces = _split_text(cjk, 40)
    assert all(len(p) <= 40 for p in pieces)
    assert "".join(pieces) == cjk                         # nothing lost
    assert all(p.endswith("。") for p in pieces)          # cut on punctuation


def test_char_budget_is_conservative_per_language():
    """The guard never counts tokens: it bounds the text in characters."""
    model = _FakeModel(limit=2048)
    assert _char_budget(model, "en") == (2048 - 2) // 2     # plain Latin: NFKD at worst
    assert _char_budget(model, "zh") == (2048 - 2) // 8     # Cangjie: measured max
    assert _char_budget(model, "ja") == _char_budget(model, "zh")
    # Every budget stays above the gateway default (240), so a chunk produced by
    # `chunking.max_chars` is never split a second time here.
    assert min(_char_budget(model, c) for c in ("en", "zh", "ja", "ko", "he", "ru")) >= 240


def test_split_for_model_uses_the_character_budget():
    model = _FakeModel(limit=64)                 # en budget: (64 - 2) // 2 = 31 chars
    text = " ".join(f"w{i}" for i in range(20))
    pieces = _split_for_model(model, text, "en")
    assert len(pieces) > 1
    assert all(len(p) <= 31 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")
    # A text that fits is returned untouched (no pointless prosody break).
    assert _split_for_model(model, "hello there", "en") == ["hello there"]


def test_tokenizer_is_never_called():
    """The guard must not pay for a tokenizer pass on the time-to-first-audio path."""
    model = _FakeModel(limit=2048)
    model.tokenizer = _CountingTokenizer()
    long_text = "a" * 5000
    pieces = _split_for_model(model, long_text, "en")
    assert len(pieces) > 1
    assert model.tokenizer.calls == 0


def test_split_is_skipped_without_a_positional_table():
    """Turbo/Nano have no learned positional table (their tokenizer truncates)."""
    model = types.SimpleNamespace(t3=types.SimpleNamespace(text_pos_emb=None), tokenizer=None)
    text = "a " * 5000
    assert _split_for_model(model, text, "en") == [text]


def test_synth_splits_and_concatenates_long_text(tmp_path, monkeypatch):
    from app import chatterbox_worker as worker

    monkeypatch.setattr(worker, "MODEL", _FakeModel(limit=64))   # en budget: 31 chars
    monkeypatch.setattr(worker, "FAMILY", "chatterbox")
    monkeypatch.setattr(worker, "SR", 24000)

    text = " ".join(f"w{i}" for i in range(20))
    items = worker._synth({"texts": [text], "language": "English", "out_dir": str(tmp_path)})

    assert len(items) == 1
    model = worker.MODEL
    assert items[0]["parts"] == len(model.calls) > 1
    assert all(len(c) <= 31 for c in model.calls)
    assert " ".join(model.calls).split() == text.split()

    audio, sr = sf.read(items[0]["path"])
    gap = int(24000 * worker._SPLIT_SILENCE_MS / 1000)
    expected = sum(len(c.split()) * 100 for c in model.calls) + gap * (len(model.calls) - 1)
    assert sr == 24000
    assert len(audio) == expected

    # A short text stays a single piece.
    short = worker._synth(
        {"texts": ["hello there"], "language": "English", "out_dir": str(tmp_path)}
    )
    assert short[0]["parts"] == 1


def test_worker_loop_exits_on_fatal_error(monkeypatch):
    """The real worker replies `fatal` and exits instead of serving a dead context."""
    from app import chatterbox_worker as worker

    class Boom:
        def generate(self, text, **kwargs):  # noqa: ARG002
            raise RuntimeError("CUDA error: device-side assert triggered")

    out = io.StringIO()
    monkeypatch.setattr(worker, "_OUT", out)
    monkeypatch.setattr(worker, "MODEL", Boom())
    monkeypatch.setattr(worker, "FAMILY", "chatterbox")
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"cmd": "synth", "texts": ["hi"]}) + "\n")
    )

    assert worker.main() == 1
    reply = json.loads(out.getvalue().strip())
    assert reply["ok"] is False
    assert reply["fatal"] is True
    assert "device-side assert" in reply["error"]


def test_chatterbox_ref_data_url():
    """An inline payload is written to a temp file the caller owns."""
    engine = TTSEngine(Settings())
    path, owned = engine._chatterbox_ref(_data_url(_valid_clip()))
    assert path and owned and os.path.isfile(path)
    os.unlink(path)


def test_chatterbox_ref_local_path(tmp_path):
    engine = TTSEngine(Settings())
    good = tmp_path / "ref.wav"
    sf.write(good, _valid_clip(), 24000)
    assert engine._chatterbox_ref(str(good)) == (str(good), False)
    assert engine._chatterbox_ref(None) == (None, False)
    with pytest.raises(ValueError, match="not found"):
        engine._chatterbox_ref(str(tmp_path / "missing.wav"))


def test_chatterbox_inline_ref_is_cleaned_up():
    """The temp file written for a data-URL reference must not be left behind."""
    engine = TTSEngine(Settings())
    spy = _SpyClient()
    engine._handles["chatterbox"] = ModelHandle(
        family="chatterbox", repo_id="ResembleAI/chatterbox",
        model=spy, backend="chatterbox",
    )
    engine.synthesize_batch("chatterbox", [
        {"text": "hi", "language": "English", "generation": {},
         "ref_audio": _data_url(_valid_clip())},
    ])
    # The engine handed a real file to the worker, then deleted it.
    assert spy.ref_path and not os.path.exists(spy.ref_path)


def test_chatterbox_local_ref_is_not_deleted(tmp_path):
    """A reference file the engine did not create is never removed."""
    engine = TTSEngine(Settings())
    good = tmp_path / "ref.wav"
    sf.write(good, _valid_clip(), 24000)
    spy = _SpyClient()
    engine._handles["chatterbox"] = ModelHandle(
        family="chatterbox", repo_id="ResembleAI/chatterbox",
        model=spy, backend="chatterbox",
    )
    engine.synthesize_batch("chatterbox", [
        {"text": "hi", "language": "English", "generation": {}, "ref_audio": str(good)},
    ])
    assert spy.ref_path == str(good) and good.is_file()


def test_chatterbox_ref_rejects_degenerate_clips():
    engine = TTSEngine(Settings())
    with pytest.raises(ValueError, match="too short"):
        engine._chatterbox_ref(_data_url(np.zeros(1000, dtype=np.float32), 16000))
    with pytest.raises(ValueError, match="silent"):
        engine._chatterbox_ref(_data_url(np.zeros(16000, dtype=np.float32), 16000))
    nan_clip = _valid_clip(sr=16000)
    nan_clip[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        engine._chatterbox_ref(_data_url(nan_clip, 16000))


def test_worker_fatal_error_retires_the_client(tmp_path, monkeypatch):
    client = _fake_client(tmp_path, monkeypatch)
    try:
        assert client.is_alive()
        with pytest.raises(RuntimeError):
            client.synthesize(["hi"], language="en", audio_prompt=None, params={})
        # The fatal reply must have torn the poisoned worker down.
        assert not client.is_alive()
    finally:
        client.close()


def test_engine_unloads_dead_chatterbox_worker(tmp_path, monkeypatch):
    engine = TTSEngine(Settings())
    engine._handles["chatterbox"] = ModelHandle(
        family="chatterbox",
        repo_id="ResembleAI/chatterbox",
        model=_fake_client(tmp_path, monkeypatch),
        backend="chatterbox",
    )
    with pytest.raises(RuntimeError):
        engine.synthesize_batch(
            "chatterbox", [{"text": "hi", "language": "English", "generation": {}}]
        )
    # Unloaded, so the next request starts a clean worker instead of failing again.
    assert "chatterbox" not in engine.loaded_families()

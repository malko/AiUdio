"""Streaming helpers: vLLM-Omni routing and the fallback to the local engine."""

import asyncio

from app.config import Settings
from app.main import _stream_pcm
from app.schemas import TTSRequest


def _collect(agen):
    async def run():
        return [item async for item in agen]

    return asyncio.run(run())


def test_vllm_serving_custom_voice_unloads_the_local_copy(monkeypatch):
    from app import main as main_mod

    unloaded = []

    class Engine:
        def loaded_families(self):
            return ["custom_voice"]

        def unload(self, family=None):
            unloaded.append(family)

    class Scheduler:
        engine = Engine()

        async def submit_streaming(self, family, item):  # pragma: no cover
            raise AssertionError("the local engine must not be used when vLLM streams")

    async def fake_vllm(settings, req, text):
        yield ("", b"\x00\x00" * 16, 24000)

    monkeypatch.setattr(main_mod, "_stream_vllm", fake_vllm)
    settings = Settings()
    settings.vllm.enabled = True
    req = TTSRequest(text="Bonjour tout le monde.", family="custom_voice", stream=True)

    out = _collect(_stream_pcm(settings, Scheduler(), req, req.text))

    assert out and out[0][1]  # PCM forwarded from the vLLM stream
    assert unloaded == ["custom_voice"]  # the local duplicate was dropped


def test_vllm_failure_falls_back_to_the_local_engine(monkeypatch):
    from app import main as main_mod

    unloaded = []
    used = []

    class Engine:
        def loaded_families(self):
            return ["custom_voice"]

        def unload(self, family=None):
            unloaded.append(family)

    class Scheduler:
        engine = Engine()

        async def submit_streaming(self, family, item):
            used.append(item["text"])
            return [0.0] * 240, 24000

    async def empty_vllm(settings, req, text):
        return
        yield  # pragma: no cover - an async generator that yields nothing

    monkeypatch.setattr(main_mod, "_stream_vllm", empty_vllm)
    settings = Settings()
    settings.vllm.enabled = True
    req = TTSRequest(text="Bonjour tout le monde.", family="custom_voice", stream=True)

    out = _collect(_stream_pcm(settings, Scheduler(), req, req.text))

    assert used == ["Bonjour tout le monde."]  # local engine produced the audio
    assert out
    assert unloaded == []  # nothing was dropped: vLLM never delivered


def test_note_cuda_oom_counts_and_logs(caplog):
    from app.metrics import metrics
    from app.scheduler import note_cuda_oom

    before = metrics.snapshot()["counters"].get("tts_cuda_oom", 0)
    note_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB."),
                  "base", [{"text": "Bonjour tout le monde"}])
    assert metrics.snapshot()["counters"]["tts_cuda_oom"] == before + 1
    # A different error is left alone.
    note_cuda_oom(RuntimeError("boom"), "base", [{"text": "x"}])
    assert metrics.snapshot()["counters"]["tts_cuda_oom"] == before + 1


def test_speech_url_accepts_a_custom_path():
    from app.vllm_backend import speech_url

    assert speech_url("http://tts-vllm:8091") == "http://tts-vllm:8091/v1/audio/speech"
    assert speech_url("http://tts-vllm:8091/", "/v1/audio/speech/stream") == \
        "http://tts-vllm:8091/v1/audio/speech/stream"
    assert speech_url("http://tts-vllm:8091", "") == "http://tts-vllm:8091/v1/audio/speech"

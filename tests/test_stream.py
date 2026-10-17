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


def test_speech_payload_asks_for_raw_pcm():
    from app.schemas import TTSRequest
    from app.vllm_backend import speech_payload

    req = TTSRequest(text="Bonjour", family="custom_voice", speaker="Vivian")
    payload = speech_payload(req, "Bonjour", sample_rate=24000)
    # Without stream_format='audio' the service answers SSE events, not PCM.
    assert payload["stream_format"] == "audio"
    assert payload["response_format"] == "pcm"
    assert payload["stream"] is True
    assert payload["sample_rate"] == 24000
    assert payload["voice"] == "vivian"  # served speakers are lowercase


def test_warm_vllm_is_a_noop_when_disabled(monkeypatch):
    from app.main import _warm_vllm

    called = []

    async def fake_stream(*a, **kw):  # pragma: no cover - must not be called
        called.append(1)
        yield b""

    monkeypatch.setattr("app.vllm_backend.stream_pcm", fake_stream)
    settings = Settings()  # vllm.enabled is False
    asyncio.run(_warm_vllm(settings))
    assert called == []


def test_warm_vllm_retries_then_gives_up_quietly(monkeypatch):
    from app.main import _warm_vllm

    calls = []

    async def boom(*a, **kw):
        calls.append(1)
        raise OSError("connection refused")
        yield b""  # pragma: no cover - makes it an async generator

    monkeypatch.setattr("app.vllm_backend.stream_pcm", boom)
    settings = Settings()
    settings.vllm.enabled = True
    ok = asyncio.run(_warm_vllm(settings, attempts=3, delay=0))  # must not raise
    assert ok is False
    assert len(calls) == 3  # the service may still be starting: retry


def test_warm_vllm_succeeds_after_a_retry(monkeypatch):
    from app.main import _warm_vllm

    calls = []

    async def flaky(*a, **kw):
        calls.append(1)
        if len(calls) < 2:
            raise OSError("connection refused")
        yield b"\x00\x00"

    monkeypatch.setattr("app.vllm_backend.stream_pcm", flaky)
    settings = Settings()
    settings.vllm.enabled = True
    assert asyncio.run(_warm_vllm(settings, attempts=3, delay=0)) is True
    assert len(calls) == 2


def test_release_vllm_family_frees_the_local_copy():
    import asyncio as _asyncio

    from app.main import _release_vllm_family

    dropped = []

    class Engine:
        def loaded_families(self):
            return ["custom_voice", "base"]

        def unload(self, family=None):
            dropped.append(family)

    engine = Engine()
    assert _asyncio.run(_release_vllm_family(engine, "test")) is True
    assert dropped == ["custom_voice"]  # only the family vLLM serves

    class Idle:
        def loaded_families(self):
            return ["chatterbox"]

        def unload(self, family=None):  # pragma: no cover - nothing to drop
            raise AssertionError("nothing to unload")

    assert _asyncio.run(_release_vllm_family(Idle(), "test")) is False

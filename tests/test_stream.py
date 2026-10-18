"""Streaming helpers: vLLM-Omni routing and the fallback to the local engine."""

import asyncio

from fastapi import HTTPException

from app.config import Settings
from app.main import _stream_pcm
from app.schemas import TTSRequest


def _collect(agen):
    async def run():
        return [item async for item in agen]

    return asyncio.run(run())


def test_vllm_serving_custom_voice_unloads_the_local_copy(monkeypatch, tmp_path):
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
    settings.paths.voices = str(tmp_path / "voices")
    settings.vllm.enabled = True
    req = TTSRequest(text="Bonjour tout le monde.", family="custom_voice", stream=True)

    out = _collect(_stream_pcm(settings, Scheduler(), req, req.text))

    assert out and out[0][1]  # PCM forwarded from the vLLM stream
    assert unloaded == ["custom_voice"]  # the local duplicate was dropped


def test_vllm_failure_raises_instead_of_falling_back(monkeypatch, tmp_path):
    """The selected engine must not be swapped silently for the local one."""
    from fastapi import HTTPException

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

    async def failing_vllm(settings, req, text):
        raise main_mod._vllm_unavailable(settings, "the service is unreachable")
        yield  # pragma: no cover - never reached

    monkeypatch.setattr(main_mod, "_stream_vllm", failing_vllm)
    settings = Settings()
    settings.paths.voices = str(tmp_path / "voices")
    settings.vllm.enabled = True
    req = TTSRequest(text="Bonjour tout le monde.", family="custom_voice", stream=True)

    try:
        _collect(_stream_pcm(settings, Scheduler(), req, req.text))
        raise AssertionError("a 503 was expected")
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "vLLM-Omni" in exc.detail

    assert used == []  # the local engine was *not* used
    assert unloaded == []


def test_vllm_failure_falls_back_when_explicitly_enabled(monkeypatch, tmp_path):
    from app import main as main_mod

    used = []

    class Scheduler:
        async def submit_streaming(self, family, item):
            used.append(item["text"])
            return [0.0] * 240, 24000

    async def failing_vllm(settings, req, text):
        raise main_mod._vllm_unavailable(settings, "the service is unreachable")
        yield  # pragma: no cover - never reached

    monkeypatch.setattr(main_mod, "_stream_vllm", failing_vllm)
    settings = Settings()
    settings.paths.voices = str(tmp_path / "voices")
    settings.vllm.enabled = True
    settings.vllm.fallback_to_local = True  # the documented escape hatch
    req = TTSRequest(text="Bonjour tout le monde.", family="custom_voice", stream=True)

    out = _collect(_stream_pcm(settings, Scheduler(), req, req.text))

    assert used == ["Bonjour tout le monde."]
    assert out


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
    assert _asyncio.run(_release_vllm_family(engine, "custom_voice", "test")) is True
    assert dropped == ["custom_voice"]  # only the family vLLM serves

    class Idle:
        def loaded_families(self):
            return ["chatterbox"]

        def unload(self, family=None):  # pragma: no cover - nothing to drop
            raise AssertionError("nothing to unload")

    assert _asyncio.run(_release_vllm_family(Idle(), "custom_voice", "test")) is False


def _settings_with_avatar(tmp_path):
    """Settings whose voice bank holds one cloned voice."""
    from app.avatars import Avatar, AvatarStore

    settings = Settings()
    settings.paths.voices = str(tmp_path)
    store = AvatarStore(tmp_path / "avatars")
    store.save(Avatar(id="bugatti", name="Bugatti", kind="clone",
                      ref_text="Bonjour, je suis Bugatti."))
    store.wav_path("bugatti").write_bytes(b"RIFF....WAVE")
    return settings


def test_vllm_serves_clones_when_configured_for_base(tmp_path):
    from app.main import _vllm_eligible, _vllm_payload

    settings = _settings_with_avatar(tmp_path)
    settings.vllm.enabled = True
    settings.vllm.family = "base"

    clone = TTSRequest(text="Bonjour", avatar="bugatti")
    assert _vllm_eligible(settings, clone) is True
    payload = _vllm_payload(settings, clone, "Bonjour")
    assert payload["task_type"] == "Base"
    # The clip travels inline: the service cannot read our voice directory.
    assert payload["ref_audio"].startswith("data:audio/wav;base64,")
    assert payload["ref_text"] == "Bonjour, je suis Bugatti."
    assert payload["x_vector_only_mode"] is False
    assert payload["stream_format"] == "audio"
    assert "voice" not in payload

    # A preset voice belongs to the other checkpoint: the local engine keeps it.
    assert _vllm_eligible(settings, TTSRequest(text="Bonjour", speaker="vivian")) is False


def test_vllm_serves_presets_when_configured_for_custom_voice(tmp_path):
    from app.main import _vllm_eligible, _vllm_payload

    settings = _settings_with_avatar(tmp_path)  # vllm.family defaults to custom_voice
    settings.vllm.enabled = True

    preset = TTSRequest(text="Bonjour", speaker="Vivian")
    assert _vllm_eligible(settings, preset) is True
    assert _vllm_payload(settings, preset, "Bonjour")["voice"] == "vivian"

    clone = TTSRequest(text="Bonjour", avatar="bugatti")
    assert _vllm_eligible(settings, clone) is False


def test_clone_routing_requires_a_reference_clip(tmp_path):
    from app.main import _vllm_eligible, _vllm_payload

    settings = _settings_with_avatar(tmp_path)
    settings.vllm.enabled = True
    settings.vllm.family = "base"
    # A Base task with nothing to clone from has to stay on the local engine.
    assert _vllm_eligible(settings, TTSRequest(text="Bonjour", family="base")) is False
    # A client-supplied local path is refused by default (SSRF/file disclosure),
    # even on the vLLM route.
    blocked = TTSRequest(text="Bonjour", ref_audio=str(tmp_path / "avatars" / "bugatti.wav"))
    try:
        _vllm_payload(settings, blocked, "Bonjour")
        raise AssertionError("a local ref_audio path must be refused by default")
    except HTTPException as exc:
        assert exc.status_code == 400
    # Chatterbox is a local clone engine: the vLLM service cannot serve it.
    settings.models.clone_backend = "chatterbox"
    assert _vllm_eligible(settings, TTSRequest(text="Bonjour", avatar="bugatti")) is False
    # With local paths explicitly allowed (confined to reference_audio), a clip
    # without a transcript clones from the speaker embedding only.
    settings.models.clone_backend = "qwen3"
    settings.audio.allow_local_ref_paths = True
    settings.paths.reference_audio = str(tmp_path)
    payload = _vllm_payload(
        settings,
        TTSRequest(text="Bonjour", ref_audio=str(tmp_path / "avatars" / "bugatti.wav")),
        "Bonjour",
    )
    assert payload["x_vector_only_mode"] is True
    assert "ref_text" not in payload


def test_family_for_model_reads_the_served_checkpoint():
    from app.vllm_backend import family_for_model

    assert family_for_model("Qwen/Qwen3-TTS-12Hz-0.6B-Base") == "base"
    assert family_for_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice") == "custom_voice"
    assert family_for_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign") is None


def test_shared_voice_bank_is_sent_as_a_file_uri(tmp_path):
    from app.main import _vllm_payload

    settings = _settings_with_avatar(tmp_path)
    settings.vllm.enabled = True
    settings.vllm.family = "base"
    # The service container mounts the gateway's `voices/` read-only there.
    settings.vllm.ref_root = "/svc/voices"

    payload = _vllm_payload(settings, TTSRequest(text="Bonjour", avatar="bugatti"), "Bonjour")
    # No audio travels: the service reads the very same file and reloads it when
    # it changes (the locator is the path).
    assert payload["ref_audio"] == "file:///svc/voices/avatars/bugatti.wav"
    assert "base64" not in payload["ref_audio"]

    # Without the shared mount, the clip is inlined instead.
    settings.vllm.ref_root = ""
    inline = _vllm_payload(settings, TTSRequest(text="Bonjour", avatar="bugatti"), "Bonjour")
    assert inline["ref_audio"].startswith("data:audio/wav;base64,")


def test_engine_toggle_starts_and_stops_the_container(monkeypatch):
    from app import main as main_mod
    from app.main import _vllm_lifecycle

    calls = []
    state = {"running": False}  # the daemon's truth, moved by start/stop

    async def fake_start(socket, name, timeout=None):
        calls.append(("start", socket, name))
        state["running"] = True
        return {"running": True}

    async def fake_stop(socket, name):
        calls.append(("stop", socket, name))
        state["running"] = False
        return {"running": False}

    async def fake_status(socket, name):
        return {"exists": True, "running": state["running"]}

    async def fake_warm(settings, *a, **kw):
        calls.append(("warm",))
        return True

    monkeypatch.setattr(main_mod.docker_ctl, "start", fake_start)
    monkeypatch.setattr(main_mod.docker_ctl, "stop", fake_stop)
    monkeypatch.setattr(main_mod.docker_ctl, "status", fake_status)
    monkeypatch.setattr(main_mod, "_warm_vllm", fake_warm)

    settings = Settings()
    settings.docker.control = True
    settings.vllm.enabled = True

    # vLLM-Omni selected: the service is started and primed.
    asyncio.run(_vllm_lifecycle(settings, start=True))
    assert calls[0] == ("start", "/var/run/docker.sock", "tts-vllm")
    assert ("warm",) in calls

    # Local engine selected: it is stopped, freeing the VRAM.
    asyncio.run(_vllm_lifecycle(settings, start=False))
    assert calls[-1] == ("stop", "/var/run/docker.sock", "tts-vllm")


def test_lifecycle_only_primes_an_external_service(monkeypatch):
    from app import main as main_mod
    from app.main import _vllm_lifecycle

    calls = []

    async def fake_start(*a, **kw):  # pragma: no cover - control is off
        calls.append("start")

    async def fake_warm(settings, *a, **kw):
        calls.append("warm")
        return True

    monkeypatch.setattr(main_mod.docker_ctl, "start", fake_start)
    monkeypatch.setattr(main_mod, "_warm_vllm", fake_warm)

    settings = Settings()  # docker.control is False
    settings.vllm.enabled = True
    asyncio.run(_vllm_lifecycle(settings, start=True))
    asyncio.run(_vllm_lifecycle(settings, start=False))
    # Nothing to pilot: only the warm-up runs, and stopping is a no-op.
    assert calls == ["warm"]


def test_lifecycle_reports_a_missing_container(monkeypatch, caplog):
    from app import main as main_mod
    from app.main import _vllm_lifecycle

    async def fake_status(*a, **kw):
        return {"exists": False, "running": False}

    async def fake_start(*a, **kw):  # pragma: no cover - must not be called
        raise AssertionError("nothing to start")

    monkeypatch.setattr(main_mod.docker_ctl, "status", fake_status)
    monkeypatch.setattr(main_mod.docker_ctl, "start", fake_start)

    settings = Settings()
    settings.docker.control = True
    with caplog.at_level("WARNING"):
        asyncio.run(_vllm_lifecycle(settings, start=True))
    assert "does not exist" in caplog.text


def test_idle_service_is_put_to_sleep_and_woken_on_demand(monkeypatch):
    from app import main as main_mod

    calls = []

    async def fake_sleep(base_url, level=1, stage_ids=(0, 1)):
        calls.append(("sleep", base_url, level))
        return True

    async def fake_wakeup(base_url, stage_ids=(0, 1)):
        calls.append(("wake", base_url))
        return True

    monkeypatch.setattr(main_mod.vllm_backend, "omni_sleep", fake_sleep)
    monkeypatch.setattr(main_mod.vllm_backend, "omni_wakeup", fake_wakeup)

    settings = Settings()
    settings.vllm.enabled = True
    settings.vllm.base_url = "http://tts-vllm:8091"
    settings.vllm.sleep_after = 0.0  # never, by default
    main_mod._forget_vllm_sleep()
    main_mod._vllm_mark_used()

    assert asyncio.run(main_mod._vllm_sleep_if_idle(settings)) is False

    settings.vllm.sleep_after = 1.0
    main_mod._VLLM_IDLE_SINCE -= 5.0  # idle for longer than the threshold
    assert asyncio.run(main_mod._vllm_sleep_if_idle(settings)) is True
    assert calls == [("sleep", "http://tts-vllm:8091", 1)]
    assert main_mod._VLLM_SLEEPING is True

    # The next request wakes it before streaming.
    assert asyncio.run(main_mod._vllm_wake_if_sleeping(settings)) is True
    assert calls[-1] == ("wake", "http://tts-vllm:8091")
    assert main_mod._VLLM_SLEEPING is False


def test_sleep_mode_is_reported_once_when_the_service_does_not_support_it(
    monkeypatch, caplog
):
    from app import main as main_mod

    calls = []

    async def unsupported(base_url, level=1, stage_ids=(0, 1)):
        calls.append(1)
        raise RuntimeError("404 Not Found")

    monkeypatch.setattr(main_mod.vllm_backend, "omni_sleep", unsupported)
    settings = Settings()
    settings.vllm.enabled = True
    settings.vllm.sleep_after = 1.0
    main_mod._forget_vllm_sleep()
    main_mod._VLLM_IDLE_SINCE = 0.0  # long idle

    with caplog.at_level("INFO"):
        assert asyncio.run(main_mod._vllm_sleep_if_idle(settings)) is False
        # Told once: no repeated attempts, no log spam.
        assert asyncio.run(main_mod._vllm_sleep_if_idle(settings)) is False
    assert len(calls) == 1
    assert "--enable-sleep-mode" in caplog.text


def test_wake_failure_is_reported(monkeypatch):
    from app import main as main_mod

    async def boom(base_url, stage_ids=(0, 1)):
        raise RuntimeError("network down")

    monkeypatch.setattr(main_mod.vllm_backend, "omni_wakeup", boom)
    settings = Settings()
    settings.vllm.enabled = True
    main_mod._forget_vllm_sleep()
    main_mod._VLLM_SLEEPING = True
    try:
        assert asyncio.run(main_mod._vllm_wake_if_sleeping(settings)) is False
    finally:
        main_mod._forget_vllm_sleep()


def test_budget_scales_the_stage_reservation_with_the_checkpoint(monkeypatch):
    from app import main as main_mod

    # A 15.48 GiB card, like the one this was tuned on.
    monkeypatch.setattr(
        main_mod, "_gpu_memory",
        lambda: {"total_bytes": int(15.48 * 1024 ** 3), "free_bytes": int(9 * 1024 ** 3),
                 "name": "NVIDIA GeForce RTX 5060 Ti"},
    )
    settings = Settings()

    big = main_mod._vllm_budget(settings, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    small = main_mod._vllm_budget(settings, "Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    # The talker's share must cover its weights (4.1 GiB for the 1.7B, 1.4 for the
    # 0.6B): the fraction follows the model, the codec's is the same for both.
    assert big["weights_gib"]["talker"] > small["weights_gib"]["talker"]
    assert big["suggested"]["0"] > small["suggested"]["0"]
    assert big["suggested"]["1"] == small["suggested"]["1"]
    assert small["suggested"]["0"] == 0.19  # (1.45 + 1.2 ctx + 0.22 kv) / 15.48
    assert small["suggested"]["1"] == 0.11  # (0.6 + 1.0) / 15.48
    assert big["suggested"]["0"] == 0.36  # (4.12 + 1.2 ctx + 0.22 kv) / 15.48
    assert big["suggested"]["reserved_gib"] == round(
        (big["suggested"]["0"] + big["suggested"]["1"]) * 15.48, 2
    )
    # The ready-to-paste flag is what the compose/command needs.
    assert big["suggested"]["stage_overrides"] == (
        '{"0": {"gpu_memory_utilization": 0.36}, "1": {"gpu_memory_utilization": 0.11}}'
    )
    assert big["total_gib"] == 15.48 and big["free_gib"] == 9.0


def test_budget_says_so_when_it_cannot_guess(monkeypatch):
    from app import main as main_mod

    settings = Settings()
    monkeypatch.setattr(main_mod, "_gpu_memory", lambda: None)
    assert "CUDA" in main_mod._vllm_budget(settings, "Qwen/Qwen3-TTS-12Hz-0.6B-Base")["error"]

    monkeypatch.setattr(
        main_mod, "_gpu_memory",
        lambda: {"total_bytes": 16 * 1024 ** 3, "free_bytes": 8 * 1024 ** 3, "name": "x"},
    )
    assert "Qwen3-TTS" in main_mod._vllm_budget(settings, "meta-llama/Llama-3-8B")["error"]
    assert "guess" in main_mod._vllm_budget(settings, "Qwen/Qwen3-TTS-12Hz-Base")["error"]
    assert main_mod._params_from_repo("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice") == 1.7
    assert main_mod._params_from_repo("Qwen/Qwen3-TTS-12Hz-0.6B-Base") == 0.6
    assert main_mod._params_from_repo("Qwen/Qwen3-TTS-12Hz-Base") is None


def test_manual_budget_overrides_the_computation(monkeypatch):
    from app import main as main_mod

    monkeypatch.setattr(
        main_mod, "_gpu_memory",
        lambda: {"total_bytes": int(15.48 * 1024 ** 3), "free_bytes": int(9 * 1024 ** 3), "name": "x"},
    )
    settings = Settings()
    model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    auto = main_mod._effective_budget(settings, model)
    assert auto["source"] == "computed" and auto["stages"] == {"0": 0.19, "1": 0.11}

    # Explicit values always win: our numbers are a suggestion, not a verdict.
    settings.vllm.stage_memory = {"0": 0.40, "1": 0.20}
    manual = main_mod._effective_budget(settings, model)
    assert manual["source"] == "manual"
    assert manual["stages"] == {"0": 0.4, "1": 0.2}

    # A single stage can be pinned, the other keeps being computed.
    settings.vllm.stage_memory = {"1": 0.25}
    mixed = main_mod._effective_budget(settings, model)
    assert mixed["stages"] == {"0": 0.19, "1": 0.25}

    # Without CUDA nothing is computed, but an explicit budget still applies.
    monkeypatch.setattr(main_mod, "_gpu_memory", lambda: None)
    settings.vllm.stage_memory = {}
    assert main_mod._effective_budget(settings, model)["stages"] == {}
    settings.vllm.stage_memory = {"0": 0.3}
    assert main_mod._effective_budget(settings, model)["stages"] == {"0": 0.3}


def test_launch_profile_becomes_the_container_flags(monkeypatch):
    from app import main as main_mod

    monkeypatch.setattr(
        main_mod, "_gpu_memory",
        lambda: {"total_bytes": int(15.48 * 1024 ** 3), "free_bytes": int(9 * 1024 ** 3), "name": "x"},
    )
    settings = Settings()
    settings.vllm.stage_memory = {"0": 0.30, "1": 0.12}
    settings.vllm.max_model_len = 4096
    settings.vllm.enforce_eager = False

    values, switches = main_mod._vllm_flag_plan(settings, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    assert values["--max-model-len"] == "4096"
    assert values["--stage-overrides"] == (
        '{"0": {"gpu_memory_utilization": 0.3}, "1": {"gpu_memory_utilization": 0.12}}'
    )
    assert switches == {"--enforce-eager": False}


def test_reconcile_recreates_only_what_differs(monkeypatch):
    from app import main as main_mod

    calls = []

    async def fake_status(socket, name):
        return {
            "exists": True, "running": True, "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "command": ["vllm", "serve", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "--omni",
                        "--max-model-len", "2048"],
        }

    async def fake_set_model(socket, name, model=None, **kw):
        calls.append({"model": model, **kw})
        return {"running": True}

    async def fake_status_start(*a, **kw):
        return {"exists": True, "running": True}

    monkeypatch.setattr(main_mod, "_gpu_memory", lambda: None)
    monkeypatch.setattr(main_mod.docker_ctl, "status", fake_status)
    monkeypatch.setattr(main_mod.docker_ctl, "set_model", fake_set_model)
    monkeypatch.setattr(main_mod.docker_ctl, "start", fake_status_start)
    monkeypatch.setattr(main_mod, "_warm_vllm", lambda *a, **kw: _noop())

    settings = Settings()
    settings.docker.control = True
    settings.vllm.enabled = True
    settings.vllm.stage_memory = {"0": 0.30, "1": 0.12}
    settings.vllm.enforce_eager = False
    settings.vllm.max_model_len = 2048

    # The flags differ (no --stage-overrides, a stray --enforce-eager default), the
    # checkpoint does not: one recreate, same model.
    asyncio.run(main_mod._vllm_reconcile(settings))
    assert len(calls) == 1
    assert calls[0]["model"] is None
    assert calls[0]["switches"] == {"--enforce-eager": False}
    assert calls[0]["values"]["--stage-overrides"].startswith('{"0"')

    # Now the container matches: nothing is recreated.
    calls.clear()

    async def matching_status(socket, name):
        return {
            "exists": True, "running": True, "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "command": ["vllm", "serve", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "--omni",
                        "--max-model-len", "2048",
                        "--stage-overrides", '{"0": {"gpu_memory_utilization": 0.3}, "1": {"gpu_memory_utilization": 0.12}}'],
        }

    monkeypatch.setattr(main_mod.docker_ctl, "status", matching_status)
    asyncio.run(main_mod._vllm_reconcile(settings))
    assert calls == []


async def _noop():
    return None


def test_container_profile_is_read_from_its_command():
    from app.main import _vllm_container_flags

    cmd = [
        "vllm", "serve", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "--omni",
        "--max-model-len", "2048", "--enforce-eager",
        "--stage-overrides", '{"0": {"gpu_memory_utilization": 0.19}, "1": {"gpu_memory_utilization": 0.11}}',
    ]
    profile = _vllm_container_flags(cmd)
    assert profile["model"] == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert profile["stage_memory"] == {"0": 0.19, "1": 0.11}
    assert profile["max_model_len"] == 2048
    assert profile["enforce_eager"] is True

    # A container started straight from the compose: a budget, no context override.
    bare = _vllm_container_flags(["vllm", "serve", "m", "--omni"])
    assert bare["stage_memory"] == {} and bare["max_model_len"] is None
    assert bare["enforce_eager"] is False
    # Garbage in the flag is ignored, never fatal.
    broken = _vllm_container_flags(["vllm", "serve", "m", "--stage-overrides", "not json"])
    assert broken["stage_memory"] == {}


def test_service_profile_reports_divergence(monkeypatch):
    from app import main as main_mod

    monkeypatch.setattr(
        main_mod, "_gpu_memory",
        lambda: {"total_bytes": int(15.48 * 1024 ** 3), "free_bytes": int(9 * 1024 ** 3), "name": "x"},
    )
    settings = Settings()
    settings.vllm.model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    settings.vllm.max_model_len = 2048
    settings.vllm.enforce_eager = True

    matching = main_mod._vllm_container_flags([
        "vllm", "serve", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "--omni",
        "--max-model-len", "2048", "--enforce-eager",
        "--stage-overrides", '{"0": {"gpu_memory_utilization": 0.19}, "1": {"gpu_memory_utilization": 0.11}}',
    ])
    assert main_mod._vllm_profile_status(settings, matching)["matches"] is True

    # Same container, but the settings now ask for another checkpoint: divergence.
    settings.vllm.model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    status = main_mod._vllm_profile_status(settings, matching)
    assert status["matches"] is False

    # An uncomputable budget must not be reported as a divergence on its own: the
    # container keeps the flag it was started with.
    monkeypatch.setattr(main_mod, "_gpu_memory", lambda: None)
    settings.vllm.model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert main_mod._vllm_profile_status(settings, matching)["matches"] is True
    # …but a context/eager difference still is one.
    loose = main_mod._vllm_container_flags([
        "vllm", "serve", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "--omni",
        "--stage-overrides", '{"0": {"gpu_memory_utilization": 0.19}, "1": {"gpu_memory_utilization": 0.11}}',
    ])
    assert main_mod._vllm_profile_status(settings, loose)["matches"] is False


def test_flag_plan_leaves_an_uncomputable_budget_alone(monkeypatch):
    from app import main as main_mod

    monkeypatch.setattr(main_mod, "_gpu_memory", lambda: None)
    settings = Settings()
    values, _ = main_mod._vllm_flag_plan(settings, "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    assert "--stage-overrides" not in values  # nothing to say: do not strip the flag
    assert values["--max-model-len"] == "2048"


def test_local_engine_stops_a_container_started_behind_our_back(monkeypatch):
    from app import main as main_mod

    calls = []

    async def fake_status(socket, name):
        return {"exists": True, "running": True, "name": name}

    async def fake_stop(socket, name):
        calls.append(("stop", name))
        return {"exists": True, "running": False, "name": name}

    monkeypatch.setattr(main_mod.docker_ctl, "status", fake_status)
    monkeypatch.setattr(main_mod.docker_ctl, "stop", fake_stop)

    settings = Settings()
    settings.docker.control = True
    settings.vllm.enabled = False  # local engine

    # Docker (restart policy, a compose up, a manual Start) brought it up anyway: the
    # gateway takes its VRAM back.
    assert asyncio.run(main_mod._vllm_enforce_engine(settings)) is True
    assert calls == [("stop", "tts-vllm")]

    # vLLM is the engine: never stop it.
    calls.clear()
    settings.vllm.enabled = True
    assert asyncio.run(main_mod._vllm_enforce_engine(settings)) is False
    assert calls == []

    # No Docker control: nothing to enforce.
    settings.vllm.enabled = False
    settings.docker.control = False
    assert asyncio.run(main_mod._vllm_enforce_engine(settings)) is False
    assert calls == []


def test_nothing_is_logged_when_the_service_is_already_stopped(monkeypatch, caplog):
    from app import main as main_mod

    async def fake_status(socket, name):
        return {"exists": True, "running": False, "name": name}

    async def fake_stop(*a, **kw):  # pragma: no cover - must not be called
        raise AssertionError("nothing to stop")

    monkeypatch.setattr(main_mod.docker_ctl, "status", fake_status)
    monkeypatch.setattr(main_mod.docker_ctl, "stop", fake_stop)

    settings = Settings()
    settings.docker.control = True
    with caplog.at_level("INFO"):
        assert asyncio.run(main_mod._vllm_enforce_engine(settings)) is False
    assert caplog.text == ""


def test_startup_does_not_preload_local_models_when_vllm_is_selected(monkeypatch):
    """With the vLLM engine selected, no local model may be loaded at boot.

    The service's two stages profile against free VRAM: preloading a local family
    (even one it does not serve) leaves stage 0 with no KV cache.
    """
    from types import SimpleNamespace

    from app import main as main_mod

    loaded: list[str] = []

    class Engine:
        def ensure(self, family):
            loaded.append(family)

    settings = Settings()
    settings.vllm.enabled = True
    settings.startup.voice = "base"
    settings.startup.tts_family = "custom_voice"
    app = SimpleNamespace(state=SimpleNamespace(settings=settings, engine=Engine()))

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(main_mod, "_vllm_lifecycle", noop)
    asyncio.run(main_mod._apply_startup(app))
    assert loaded == []

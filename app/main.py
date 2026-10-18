"""FastAPI application: REST/OpenAI API, TTS/chat WebSocket, STT and LLM proxies.

The server is a gateway: it runs Qwen3-TTS locally, delegates STT to the
NeMo-Speech.cpp service and conversation to an external LLM (OpenAI-compatible).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, apikeys, docker_ctl, vllm_backend
from .audio import to_pcm16, wav_bytes
from .avatars import Avatar, AvatarStore
from .chatterbox_backend import is_available as is_chatterbox_available
from .chunking import chunk_text, concat_audio
from .config import ConfigStore, Settings, get_store
from .engine import TTSEngine, design_ref_text
from .llm import LLMClient, build_messages, trim_history
from .metrics import metrics
from .scheduler import BatchScheduler
from .schemas import TTSRequest
from .security import (
    is_admin,
    is_basic_admin,
    presented_key,
    request_authorized,
    validate_media_ref,
)
from .stt_models import iter_download, load_index, mark_active, read_active

log = logging.getLogger(__name__)
UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def _configure_logging(level: str = "info") -> None:
    """Route the application's own logs to stderr.

    Uvicorn only configures its own loggers: without a root handler every
    ``log.info`` from the engine and the chat pipeline was silently dropped, so
    ``server.log_level`` had no effect on them either.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    root.setLevel(getattr(logging, (level or "info").strip().upper(), logging.INFO))


def _fmt_s(seconds: Optional[float]) -> str:
    """Format an optional duration for the chat timing logs."""
    return "-" if seconds is None else f"{seconds:.2f}s"


def _avatar_dir(settings: Settings) -> Path:
    return Path(settings.paths.voices) / "avatars"


def _parse_expiry(value: Any, days: Any):
    """Parse an API-key expiry from an ISO date or a number of days."""
    from datetime import datetime, timedelta, timezone

    if value:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(400, f"Invalid expires_at: {value}") from exc
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if days not in (None, "", 0, "0"):
        try:
            return datetime.now(timezone.utc) + timedelta(days=int(days))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "Invalid expires_in_days") from exc
    return None


async def _apply_startup(app: FastAPI) -> None:
    """Preload the STT model / voice configured in Settings (best effort)."""
    settings: Settings = app.state.settings
    engine: TTSEngine = app.state.engine

    if settings.docker.control and not settings.vllm.enabled:
        # The local engine is selected: a vLLM-Omni container that is running holds
        # VRAM we are about to need. Docker may have started it on its own (and there
        # is nothing to do when it is already stopped), so the choice is enforced here
        # — *before* loading our own models.
        try:
            await _vllm_enforce_engine(settings)
        except Exception:  # pragma: no cover - defensive
            log.debug("vLLM-Omni stop at startup skipped", exc_info=True)

    if settings.startup.stt_model:
        try:
            index = await asyncio.to_thread(load_index, settings.stt.model_index)
            model = index.find(settings.startup.stt_model) if index else None
            artifact = model.asr_artifact() if model else None
            if artifact and model.local_path(artifact, settings.stt.models_dir).is_file():
                await asyncio.to_thread(mark_active, settings.stt.models_dir, artifact.filename)
                log.info("Startup: activated STT model '%s'", settings.startup.stt_model)
            else:
                log.info("Startup: STT model '%s' not downloaded yet", settings.startup.stt_model)
        except Exception:
            log.exception("Startup STT activation failed")

    voice = settings.startup.voice
    if voice and settings.vllm.enabled and _preload_is_served_by_vllm(settings, voice):
        # Both vLLM-Omni stages live on the same GPU and reserve their share at
        # startup: preloading the same family here is what makes its second stage
        # fail with CUDA OOM on a tight card.
        log.info(
            "Startup: not preloading '%s' — vLLM-Omni serves that family (keeping "
            "the VRAM for the service's two stages)",
            voice,
        )
    elif voice:
        try:
            if voice.startswith("avatar:"):
                store = AvatarStore(_avatar_dir(settings))
                avatar = store.get(voice.split(":", 1)[1])
                if avatar:
                    if _clone_family(settings) == "chatterbox":
                        await asyncio.to_thread(engine.ensure, "chatterbox")
                    else:
                        handle = await asyncio.to_thread(engine.ensure, "base")
                        await asyncio.to_thread(
                            engine._cached_clone_prompt, handle,
                            str(store.wav_path(avatar.id)), avatar.ref_text or None,
                            bool(avatar.x_vector_only),
                        )
                    log.info("Startup: loaded voice '%s'", avatar.name)
            else:
                await asyncio.to_thread(engine.ensure, voice)
                log.info("Startup: loaded model '%s'", voice)
        except Exception:
            log.exception("Startup voice load failed")

    if settings.vllm.enabled:
        try:
            # The engine switch drives the container: start it if `docker.control`
            # lets us, prime it otherwise.
            await _vllm_lifecycle(settings, start=True)
        except Exception:  # pragma: no cover - defensive
            log.debug("vLLM-Omni warm-up skipped", exc_info=True)

    family = settings.startup.tts_family
    if family and settings.vllm.enabled and family == _vllm_family(settings):
        log.info(
            "Startup: not preloading '%s' — vLLM-Omni serves it (keeping the VRAM "
            "for the service's two stages)",
            family,
        )
        family = ""
    if family:
        try:
            await asyncio.to_thread(engine.ensure, family)
            log.info("Startup: preloaded TTS family '%s'", family)
            if settings.startup.warmup and hasattr(engine, "warmup"):
                await asyncio.to_thread(engine.warmup, family)
                log.info("Startup: warmed up '%s'", family)
        except Exception:
            log.exception("Startup TTS preload failed")


def _public_config(settings: Settings, engine: TTSEngine) -> Dict[str, Any]:
    return {
        "version": __version__,
        "ui": settings.ui.model_dump(),
        "startup": settings.startup.model_dump(),
        "docker": {
            "control": settings.docker.control,
            "container": settings.docker.container,
        },
        "auth_required": settings.auth.enabled,
        "models": {
            "families": engine.available(),
            "variants": [v.model_dump() for v in settings.models.variants],
            "registry": dict(settings.models.registry),
            "default_family": settings.models.default_family,
            "default_size": settings.models.default_size,
            "residency": settings.models.residency,
            "clone_backend": settings.models.clone_backend,
            "clone_family": _clone_family(settings),
            "chatterbox": {
                "available": is_chatterbox_available(settings),
                "watermark": settings.models.chatterbox.watermark,
                "default_language": settings.models.chatterbox.default_language,
            },
        },
        "audio": settings.audio.model_dump(),
        "chunking": settings.chunking.model_dump(),
        "generation": settings.generation.model_dump(),
        "stt": {
            "enabled": settings.stt.enabled,
            "realtime_path": settings.stt.realtime_path,
        },
        "llm": {
            "base_url": settings.llm.base_url,
            "model": settings.llm.model,
            "system_prompt": settings.llm.system_prompt,
            "temperature": settings.llm.temperature,
        },
        "vllm": {
            "enabled": settings.vllm.enabled,
            "base_url": settings.vllm.base_url,
            "path": settings.vllm.path,
            "family": _vllm_family(settings),
            "ref_root": settings.vllm.ref_root,
            "model": settings.vllm.model,
            "max_model_len": settings.vllm.max_model_len,
            "enforce_eager": settings.vllm.enforce_eager,
            "stage_memory": dict(settings.vllm.stage_memory or {}),
            "sleep_after": settings.vllm.sleep_after,
            "sleep_level": settings.vllm.sleep_level,
        },
        "server": {
            "log_level": settings.server.log_level,
        },
        "languages": [
            "Auto", "Chinese", "English", "Japanese", "Korean", "German",
            "French", "Russian", "Portuguese", "Spanish", "Italian",
        ],
    }


def _clone_family(settings: Settings, backend: Optional[str] = None) -> str:
    """Family used to synthesize cloned avatars.

    ``backend`` (per-request ``clone_backend``/``engine``) overrides the admin
    ``models.clone_backend`` default.
    """
    value = (backend or settings.models.clone_backend or "qwen3").strip().lower()
    return "chatterbox" if value in ("chatterbox", "cb") else "base"


CLONE_FAMILIES = ("base", "chatterbox", "chatterbox_turbo", "chatterbox_nano")


def _item_from_request(settings: Settings, req: TTSRequest, text: str) -> Dict[str, Any]:
    gen = req.generation.to_kwargs() if req.generation else {}

    # A voice-bank id may be passed as `avatar`, or as `voice`/`speaker` (so the
    # OpenAI route works with voice-bank ids too).
    avatar_id = req.avatar
    if not avatar_id:
        candidate = req.resolve_speaker()
        if candidate:
            store = AvatarStore(_avatar_dir(settings))
            if store.get(candidate) is not None:
                avatar_id = candidate
    if avatar_id:
        store = AvatarStore(_avatar_dir(settings))
        avatar = store.get(avatar_id)
        wav = store.wav_path(avatar_id)
        if avatar is None or not wav.is_file():
            raise HTTPException(404, f"Unknown voice: {avatar_id}")
        language = req.language if isinstance(req.language, str) else None
        if not language or language == "Auto":
            language = avatar.language or "Auto"
        # Per-request clone engine, then an explicit clone-capable family, then
        # the admin default.
        family = _clone_family(settings, req.resolve_clone_backend())
        requested = req.requested_family()
        if requested in CLONE_FAMILIES:
            family = requested
        return {
            "family": family,
            "text": text,
            "language": language,
            "speaker": None,
            "instruct": None,
            "ref_audio": str(wav),
            "ref_text": avatar.ref_text or None,
            # A clone without a transcript must use the speaker embedding only.
            "x_vector_only_mode": bool(avatar.x_vector_only) or not (avatar.ref_text or "").strip(),
            "generation": gen,
        }

    family = req.resolve_family(settings.models.default_family)
    # A reference clip without an explicit family means "clone", using the
    # requested (or default) clone engine.
    if req.ref_audio and req.requested_family() is None:
        family = _clone_family(settings, req.resolve_clone_backend())
    return {
        "family": family,
        "text": text,
        "language": req.language if isinstance(req.language, str) else None,
        "speaker": req.resolve_speaker(),
        "instruct": req.resolve_instruct(),
        "ref_audio": validate_media_ref(req.ref_audio, settings, settings.paths.reference_audio),
        "ref_text": req.ref_text,
        "x_vector_only_mode": bool(req.x_vector_only_mode),
        "generation": gen,
    }


#: Families the vLLM-Omni service can serve (it loads a single checkpoint).
_VLLM_FAMILIES = ("custom_voice", "base")
#: Cache of inline reference clips: locator key → data URL (bounded: a chat must
#: not re-encode the same clip per sentence, but an unbounded map would grow with
#: every distinct clip/voice ever used).
_REF_LOCATORS: OrderedDict[str, str] = OrderedDict()
_REF_LOCATORS_MAX = 64


def _cache_locator(key: str, value: str) -> str:
    _REF_LOCATORS[key] = value
    _REF_LOCATORS.move_to_end(key)
    while len(_REF_LOCATORS) > _REF_LOCATORS_MAX:
        _REF_LOCATORS.popitem(last=False)
    return value


def _vllm_family(settings: Settings) -> str:
    """Family the configured vLLM-Omni service serves (default: preset voices)."""
    value = str(getattr(settings.vllm, "family", "") or "").strip().lower()
    return value if value in _VLLM_FAMILIES else "custom_voice"


#: Sizing model for the vLLM-Omni stages, calibrated on the observed footprints
#: (1.7B → ~4.1 GiB of weights, 0.6B → ~1.4 GiB, i.e. bf16 plus ~30 % of buffers).
_VLLM_BYTES_PER_PARAM = 2
_VLLM_WEIGHT_FACTOR = 1.3
#: Per-stage headroom on top of the weights, in GiB: the talker's KV pool (0.22 GiB
#: at `--max-model-len 2048`) plus activations and CUDA contexts; the Code2Wav
#: decoder is small but sizes its buffers for `max_num_batched_tokens` (65536).
_VLLM_STAGE_HEADROOM = {0: 0.5, 1: 1.0}
_VLLM_CODEC_WEIGHTS = 0.6


def _gpu_memory() -> Optional[Dict[str, Any]]:
    """Total/free VRAM of the card the gateway runs on (None without CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "name": torch.cuda.get_device_name(0),
        }
    except Exception:  # pragma: no cover - no driver/CPU-only install
        return None


def _params_from_repo(repo: str) -> Optional[float]:
    """Billions of parameters read from a checkpoint id (``…-0.6B-Base`` → 0.6)."""
    match = re.search(r"(\d+(?:\.\d+)?)B(?:[-_/]|$)", repo or "")
    return float(match.group(1)) if match else None


def _vllm_budget(settings: Settings, model: str) -> Dict[str, Any]:
    """Suggested per-stage `gpu_memory_utilization` for a checkpoint on this card.

    `gpu_memory_utilization` is a fraction of the GPU's **total** VRAM — vLLM says
    so itself when it refuses: *desired GPU memory utilization (0.3, 23.78 GiB)* on
    a 79.25 GiB card — and **every stage reserves its own share**, so the two
    fractions add up. The smallest budget that fits a stage is

        (weights + activation/KV headroom) / total VRAM

    rounded up to the hundredth (on a 16 GiB card, 0.01 ≈ 160 MiB). The rest of the
    card has to hold everything else (our own models, STT, the LLM…), which is why
    the sum should stay well below 1.
    """
    name = (model or "").strip()
    gpu = _gpu_memory()
    params = _params_from_repo(name)
    out: Dict[str, Any] = {"model": name, "gpu": gpu}
    if gpu is None:
        out["error"] = "CUDA is not visible from the gateway container"
        return out
    if "qwen3-tts" not in name.lower():
        out["error"] = f"{name or 'the model'} is not a Qwen3-TTS checkpoint"
        return out
    if params is None:
        out["error"] = f"cannot guess the size of {name!r}"
        return out

    total = gpu["total_bytes"] / 1024 ** 3
    talker = params * 1e9 * _VLLM_BYTES_PER_PARAM * _VLLM_WEIGHT_FACTOR / 1024 ** 3
    fraction = lambda gib: math.ceil((gib / total) * 100) / 100  # noqa: E731
    stage0 = fraction(talker + _VLLM_STAGE_HEADROOM[0])
    stage1 = fraction(_VLLM_CODEC_WEIGHTS + _VLLM_STAGE_HEADROOM[1])
    reserved = (stage0 + stage1) * total
    out["weights_gib"] = {"talker": round(talker, 2), "codec": _VLLM_CODEC_WEIGHTS}
    out["total_gib"] = round(total, 2)
    out["free_gib"] = round(gpu["free_bytes"] / 1024 ** 3, 2)
    out["suggested"] = {
        "0": stage0,
        "1": stage1,
        "reserved_gib": round(reserved, 2),
        "stage_overrides": json.dumps(
            {"0": {"gpu_memory_utilization": stage0},
             "1": {"gpu_memory_utilization": stage1}},
            separators=(", ", ": "),
        ),
        "max_model_len": 2048,
    }
    out["note"] = (
        "each stage reserves its fraction of the card's total VRAM; keep room for the "
        "gateway's own models, STT and anything else on the card"
    )
    return out


def _effective_budget(settings: Settings, model: str) -> Dict[str, Any]:
    """Per-stage VRAM budget actually applied: explicit values, computed elsewhere.

    The operator can always overrule the computation (``vllm.stage_memory``): an
    explicit stage wins, a missing one falls back to the computed value, and when
    nothing can be computed the flags are left untouched on the container.
    """
    manual = {
        str(key): float(value)
        for key, value in (settings.vllm.stage_memory or {}).items()
        if str(key) in ("0", "1") and float(value or 0) > 0
    }
    computed = _vllm_budget(settings, model)
    suggested = computed.get("suggested") or {}
    resolved: Dict[str, Any] = {
        "model": model,
        "stages": {},
        "source": "manual" if manual else ("computed" if suggested else None),
        "gpu": computed.get("gpu"),
        "weights_gib": computed.get("weights_gib"),
        "total_gib": computed.get("total_gib"),
        "free_gib": computed.get("free_gib"),
        "error": computed.get("error"),
    }
    for stage in ("0", "1"):
        value = manual.get(stage, suggested.get(stage))
        if value:
            resolved["stages"][stage] = round(float(value), 2)
    if not resolved["stages"] and resolved["error"]:
        resolved["note"] = "no budget computed: the container keeps the flags it was started with"
    return resolved


def _vllm_flag_plan(settings: Settings, model: str) -> Tuple[Dict[str, Optional[str]], Dict[str, bool]]:
    """`--flag value` / switch changes that make the container match the settings.

    The budget key is **omitted** when nothing could be computed (no CUDA, unknown
    checkpoint): the container then keeps whatever `--stage-overrides` it was started
    with, instead of having a working budget striped off.
    """
    budget = _effective_budget(settings, model)
    stages = budget.get("stages") or {}
    values: Dict[str, Optional[str]] = {
        "--max-model-len": str(int(settings.vllm.max_model_len)),
    }
    if stages:
        values["--stage-overrides"] = json.dumps(
            {stage: {"gpu_memory_utilization": value} for stage, value in sorted(stages.items())},
            separators=(", ", ": "),
        )
    return values, {"--enforce-eager": bool(settings.vllm.enforce_eager)}


def _vllm_container_flags(cmd: Optional[List[str]]) -> Dict[str, Any]:
    """The launch profile of a container, as read from its command line."""
    overrides: Dict[str, float] = {}
    raw = docker_ctl.flag_value(cmd, "--stage-overrides")
    if raw:
        try:
            for stage, spec in (json.loads(raw) or {}).items():
                value = (spec or {}).get("gpu_memory_utilization")
                if value is not None:
                    overrides[str(stage)] = round(float(value), 2)
        except (ValueError, TypeError, AttributeError):
            log.debug("Could not read --stage-overrides %r", raw)
    length = docker_ctl.flag_value(cmd, "--max-model-len")
    return {
        "model": docker_ctl.served_model(cmd),
        "stage_memory": overrides,
        "max_model_len": int(length) if (length or "").isdigit() else None,
        "enforce_eager": docker_ctl.has_flag(cmd, "--enforce-eager"),
        "command": cmd,
    }


def desired_stage_memory(values: Dict[str, Optional[str]]) -> Dict[str, float]:
    """Stage fractions carried by a `--stage-overrides` value ({} when absent)."""
    raw = values.get("--stage-overrides")
    if not raw:
        return {}
    try:
        return {
            str(stage): round(float((spec or {}).get("gpu_memory_utilization")), 2)
            for stage, spec in (json.loads(raw) or {}).items()
            if (spec or {}).get("gpu_memory_utilization") is not None
        }
    except (ValueError, TypeError, AttributeError):
        return {}


def _vllm_desired_profile(settings: Settings, model: str) -> Dict[str, Any]:
    """The launch profile the settings ask for, with the budget resolved."""
    budget = _effective_budget(settings, model)
    return {
        "model": str(settings.vllm.model or "").strip() or model,
        "stage_memory": budget.get("stages") or {},
        "max_model_len": int(settings.vllm.max_model_len),
        "enforce_eager": bool(settings.vllm.enforce_eager),
    }


def _vllm_profile_status(settings: Settings, actual: Dict[str, Any]) -> Dict[str, Any]:
    """Compare what the container runs with to what the settings want."""
    model = actual.get("model") or str(
        (settings.models.registry or {}).get(_vllm_family(settings), "")
    )
    desired = _vllm_desired_profile(settings, model)
    matches = (
        (not desired["model"] or desired["model"] == actual.get("model"))
        # An uncomputable budget is not a requirement: leave the container alone.
        and (not desired["stage_memory"] or desired["stage_memory"] == actual.get("stage_memory"))
        and desired["max_model_len"] == actual.get("max_model_len")
        and desired["enforce_eager"] == bool(actual.get("enforce_eager"))
    )
    return {"actual": actual, "desired": desired, "matches": matches}


async def _vllm_reconcile(settings: Settings) -> None:
    """Make the container match the settings (checkpoint + launch flags).

    `PUT /api/settings` is the single source of truth once `docker.control` is on:
    the compose only says how the service is *created* for the first time. Only what
    actually differs is recreated, and the engine switch still decides whether the
    service ends up running.
    """
    cfg = settings.docker
    if cfg.control:
        wanted_model = str(settings.vllm.model or "").strip()
        try:
            state = await docker_ctl.status(cfg.socket, cfg.container)
            if not state.get("exists"):
                log.warning(
                    "vLLM-Omni container %r does not exist: create it once from the "
                    "compose (or start it by hand) — the gateway cannot create it from "
                    "scratch", cfg.container,
                )
                return
            model = wanted_model or state.get("model") or str(
                (settings.models.registry or {}).get(_vllm_family(settings), "")
            )
            values, switches = _vllm_flag_plan(settings, model)
            applied = _vllm_container_flags(state.get("command"))
            unchanged = (
                (not wanted_model or wanted_model == state.get("model"))
                and (
                    "--stage-overrides" not in values
                    or desired_stage_memory(values) == applied.get("stage_memory")
                )
                and str(applied.get("max_model_len") or "") == values["--max-model-len"]
                and bool(applied.get("enforce_eager")) == switches["--enforce-eager"]
            )
            if not unchanged:
                log.info(
                    "Applying the vLLM-Omni launch profile to %s: %s --stage-overrides %s "
                    "--max-model-len %s%s",
                    cfg.container, wanted_model or "(same checkpoint)",
                    values.get("--stage-overrides") or "(unchanged)",
                    values["--max-model-len"],
                    " --enforce-eager" if switches["--enforce-eager"] else "",
                )
                await docker_ctl.set_model(
                    cfg.socket, cfg.container, wanted_model or None,
                    values=values, switches=switches, start=bool(settings.vllm.enabled),
                    timeout=cfg.start_timeout,
                )
                _forget_vllm_sleep()
        except docker_ctl.DockerError as exc:
            log.warning("Could not apply the vLLM-Omni launch profile: %s", exc)
    # Whatever happened (and without `docker.control`), honour the engine switch:
    # start the service and prime it, or stop it to free the VRAM.
    await _vllm_lifecycle(settings, start=bool(settings.vllm.enabled))


def _preload_is_served_by_vllm(settings: Settings, voice: str) -> bool:
    """Is this startup voice a family the vLLM-Omni service will serve?"""
    served = _vllm_family(settings)
    if voice.startswith("avatar:"):
        return served == "base" and _clone_family(settings) == "base"
    return voice.strip().lower() == served


def _resolve_avatar(settings: Settings, req: TTSRequest) -> Optional[Avatar]:
    """Voice-bank entry named by the request (``avatar``, or as ``voice``)."""
    store = AvatarStore(_avatar_dir(settings))
    avatar_id = req.avatar
    if not avatar_id:
        candidate = req.resolve_speaker()
        avatar = store.get(candidate) if candidate else None
        if avatar is None:
            return None
        avatar_id = candidate
    return store.get(avatar_id)


def _effective_family(settings: Settings, req: TTSRequest) -> str:
    """Family this request is synthesized with, mirroring `_item_from_request`."""
    if _resolve_avatar(settings, req) is not None:
        family = _clone_family(settings, req.resolve_clone_backend())
        requested = req.requested_family()
        return requested if requested in CLONE_FAMILIES else family
    family = req.resolve_family(settings.models.default_family)
    if req.ref_audio and req.requested_family() is None:
        family = _clone_family(settings, req.resolve_clone_backend())
    return family


def _ref_locator(value: str) -> str:
    """Reference clip as a locator the vLLM service can resolve.

    URLs and data URLs pass through; a local file becomes an inline data URL (the
    service cannot read our filesystem). Cached by path/mtime/size so a chat does
    not re-encode the clip once per sentence.
    """
    value = (value or "").strip()
    if not value or value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value)
    try:
        stat = path.stat()
    except OSError:
        return ""
    key = f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
    cached = _REF_LOCATORS.get(key)
    if cached is None:
        cached = _cache_locator(key, vllm_backend.audio_data_url(path.read_bytes()))
    return cached


def _clip_locator(settings: Settings, path: Path) -> str:
    """Locator for a local clip, shared by path when the service can see it.

    With ``vllm.ref_root`` set (the voice bank mounted read-only inside the
    service container), clips under it are sent as ``file://`` URIs: the service
    then reads the very same file as our engine and reloads it when it changes —
    no base64 travelling on every request. Everything else is inlined.
    """
    root = str(getattr(settings.vllm, "ref_root", "") or "").strip()
    if root:
        try:
            relative = path.resolve().relative_to(Path(settings.paths.voices).resolve())
            return f"file://{root.rstrip('/')}/{relative.as_posix()}"
        except (ValueError, OSError):
            pass
    return _ref_locator(str(path))


def _vllm_reference(
    settings: Settings, req: TTSRequest
) -> Optional[Tuple[str, Optional[str], bool]]:
    """Reference clip/transcript for a clone request, or None when there is none."""
    avatar = _resolve_avatar(settings, req)
    if avatar is not None:
        wav = AvatarStore(_avatar_dir(settings)).wav_path(avatar.id)
        locator = _clip_locator(settings, wav)
        if not locator:
            return None
        ref_text = (avatar.ref_text or "").strip() or None
        return locator, ref_text, bool(avatar.x_vector_only) or not ref_text
    if req.ref_audio:
        # A client-supplied reference must obey the same policy as the local engine
        # (SSRF / local file disclosure): the chat routes reach this helper without
        # a prior `validate_media_ref`.
        ref = validate_media_ref(req.ref_audio, settings, settings.paths.reference_audio)
        if not ref:
            return None
        if ref.startswith(("http://", "https://", "data:")):
            locator = ref
        else:
            locator = _clip_locator(settings, Path(ref))
        if not locator:
            return None
        ref_text = (req.ref_text or "").strip() or None
        return locator, ref_text, bool(req.x_vector_only_mode) or not ref_text
    return None


def _vllm_eligible(settings: Settings, req: TTSRequest) -> bool:
    """Does this request belong to the family the vLLM-Omni service serves?"""
    if not settings.vllm.enabled:
        return False
    if _effective_family(settings, req) != _vllm_family(settings):
        return False
    if _vllm_family(settings) == "base":
        # Cloning needs a reference clip we can hand over (a voice-bank entry or
        # an explicit `ref_audio`); otherwise the local engine has nothing to use.
        return _vllm_reference(settings, req) is not None
    return True


async def _release_vllm_family(engine: TTSEngine, family: str, reason: str) -> bool:
    """Drop our own copy of the family vLLM-Omni serves.

    The service loads **two stages** (the talker and the `Code2Wav` decoder) and each
    one reserves its GPU memory fraction up front, on the same card as ours. On a small
    shared GPU the local copy is precisely what starves the service's second stage
    (`StageEngineCoreProc failed to start … CUDA out of memory`), so it is dropped as
    soon as vLLM is asked to serve that family. Never raises: freeing VRAM must not
    break a settings update.
    """
    if family not in _VLLM_FAMILIES:  # pragma: no cover - guarded by the callers
        return False
    try:
        if family not in engine.loaded_families():
            return False
        await asyncio.to_thread(engine.unload, family)
        log.info(
            "vLLM-Omni %s: unloaded the local '%s' copy to leave it VRAM",
            reason, family,
        )
        return True
    except Exception:
        log.exception("Could not unload the local '%s' copy", family)
        return False


def _vllm_payload(
    settings: Settings, req: TTSRequest, text: str
) -> Optional[Dict[str, Any]]:
    """Speech payload for the family the service serves (None when unusable)."""
    family = _vllm_family(settings)
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only = False
    if family == "base":
        reference = _vllm_reference(settings, req)
        if reference is None:
            return None
        ref_audio, ref_text, x_vector_only = reference
    return vllm_backend.speech_payload(
        req,
        text,
        sample_rate=settings.audio.sample_rate,
        family=family,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only,
    )


#: Sleep-mode bookkeeping: is the service hibernating, and since when it is idle?
_VLLM_SLEEPING = False
_VLLM_IDLE_SINCE = 0.0
_VLLM_SLEEP_REFUSED = False


def _vllm_mark_used() -> None:
    """Note that the service was just used (resets the idle countdown)."""
    global _VLLM_IDLE_SINCE, _VLLM_SLEEPING
    _VLLM_IDLE_SINCE = time.monotonic()
    _VLLM_SLEEPING = False


async def _vllm_wake_if_sleeping(settings: Settings) -> bool:
    """Wake the hibernating service before using it. False when it cannot."""
    global _VLLM_SLEEPING
    if not _VLLM_SLEEPING:
        _vllm_mark_used()
        return True
    try:
        started = time.perf_counter()
        await vllm_backend.omni_wakeup(settings.vllm.base_url)
        log.info(
            "vLLM-Omni woken up in %.2fs (it had released its VRAM while idle)",
            time.perf_counter() - started,
        )
    except Exception as exc:
        log.warning("Could not wake the vLLM-Omni service: %s", exc)
        return False
    _vllm_mark_used()
    return True


def _forget_vllm_sleep() -> None:
    """The service went away (stopped/restarted): forget its hibernation state."""
    global _VLLM_SLEEPING, _VLLM_SLEEP_REFUSED
    _VLLM_SLEEPING = False
    _VLLM_SLEEP_REFUSED = False


async def _vllm_sleep_if_idle(settings: Settings) -> bool:
    """Ask the service to hibernate once it has been idle long enough."""
    global _VLLM_SLEEPING, _VLLM_SLEEP_REFUSED
    cfg = settings.vllm
    if _VLLM_SLEEP_REFUSED or _VLLM_SLEEPING or not cfg.enabled or cfg.sleep_after <= 0:
        return False
    idle = time.monotonic() - _VLLM_IDLE_SINCE
    if idle < cfg.sleep_after:
        return False
    try:
        await vllm_backend.omni_sleep(cfg.base_url, level=cfg.sleep_level)
    except Exception as exc:
        # The endpoint only exists with `--enable-sleep-mode`: say it once.
        _VLLM_SLEEP_REFUSED = True
        log.info(
            "vLLM-Omni sleep mode unavailable (%s): start the service with "
            "--enable-sleep-mode to let it release its VRAM while idle", exc,
        )
        return False
    _VLLM_SLEEPING = True
    log.info(
        "vLLM-Omni asleep after %.0fs idle (level %d): its VRAM is back in the pool",
        idle, cfg.sleep_level,
    )
    return True


async def _vllm_enforce_engine(settings: Settings) -> bool:
    """Stop the service if it runs while the local engine is selected.

    Docker's `restart: unless-stopped`, a `compose up` from a manager or a manual
    Start can bring the container up although the settings say the local engine
    serves: it would then hold VRAM our own models need. Returns True when it did
    stop something. Silent when there is nothing to do (or nothing to pilot).
    """
    if not settings.docker.control or settings.vllm.enabled:
        return False
    try:
        state = await docker_ctl.status(settings.docker.socket, settings.docker.container)
    except docker_ctl.DockerError as exc:
        log.debug("Could not check %s: %s", settings.docker.container, exc)
        return False
    if not (state.get("exists") and state.get("running")):
        return False
    log.info(
        "The local engine is selected but %s is running: stopping it (its VRAM "
        "belongs to our models)", settings.docker.container,
    )
    await _vllm_lifecycle(settings, start=False)
    return True


async def _vllm_idle_loop(app_state) -> None:
    """Background ticker: hibernate the service when nothing uses it.

    It also enforces the engine choice every 30 s, so a container started behind the
    gateway's back does not keep the VRAM.
    """
    global _VLLM_IDLE_SINCE
    ticks = 0
    while True:
        try:
            settings: Settings = app_state.settings
            await _vllm_sleep_if_idle(settings)
            ticks += 1
            if ticks % 6 == 0:
                await _vllm_enforce_engine(settings)
        except Exception:  # pragma: no cover - defensive, the loop must survive
            log.debug("vLLM-Omni idle check failed", exc_info=True)
        await asyncio.sleep(5.0)


async def _vllm_lifecycle(settings: Settings, *, start: bool) -> None:
    """Keep the service container in step with the engine switch.

    The UI's engine toggle *is* the lifecycle: choosing vLLM-Omni starts the
    container (and primes it), going back to the local engine stops it so the VRAM
    returns to our models. Without `docker.control` there is nothing to pilot: an
    externally managed service is simply primed. Best effort throughout — a
    failure is logged, never raised (a save must not fail because Docker said no).
    """
    cfg = settings.docker
    if not cfg.control:
        if start:
            await _warm_vllm(settings)
        return
    try:
        if start:
            state = await docker_ctl.status(cfg.socket, cfg.container)
            if not state.get("exists"):
                log.warning(
                    "vLLM-Omni container %r does not exist: create it once from the "
                    "compose (or start it by hand) — the gateway cannot create it "
                    "from scratch", cfg.container,
                )
                return
            if not state.get("running"):
                await docker_ctl.start(cfg.socket, cfg.container, timeout=cfg.start_timeout)
                log.info("vLLM-Omni container %s started (engine selected)", cfg.container)
            await _warm_vllm(settings, attempts=20, delay=15.0)
            _vllm_mark_used()
        else:
            before = await docker_ctl.status(cfg.socket, cfg.container)
            _forget_vllm_sleep()
            if not before.get("running"):
                return  # already stopped: nothing to log
            state = await docker_ctl.stop(cfg.socket, cfg.container)
            log.info(
                "vLLM-Omni container %s stopped (local engine): its VRAM is free again",
                state.get("name") or cfg.container,
            )
    except docker_ctl.DockerError as exc:
        log.warning(
            "Could not %s the vLLM-Omni container %s: %s",
            "start" if start else "stop", cfg.container, exc,
        )


async def _stream_vllm(
    settings: Settings, req: TTSRequest, text: str
) -> AsyncIterator[Tuple[str, bytes, int]]:
    """Yield native PCM from vLLM-Omni; yields nothing (to fall back) on failure."""
    if not await _vllm_wake_if_sleeping(settings):
        return
    payload = _vllm_payload(settings, req, text)
    if payload is None:  # no reference clip to clone from: stay local
        return
    started = time.perf_counter()
    stream = vllm_backend.stream_pcm(settings.vllm.base_url, payload, path=settings.vllm.path)
    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        await stream.aclose()
        return
    except Exception:
        log.warning("vLLM-Omni streaming unavailable; using the local engine", exc_info=True)
        await stream.aclose()
        return
    metrics.observe_ttfa(time.perf_counter() - started)
    metrics.incr("tts_vllm_sessions")
    try:
        yield "", first, settings.audio.sample_rate
        async for chunk in stream:
            yield "", chunk, settings.audio.sample_rate
    finally:
        await stream.aclose()


async def _stream_pcm(
    settings: Settings, scheduler: BatchScheduler, req: TTSRequest, text: str,
    *, split_chunks: bool = True,
) -> AsyncIterator[Tuple[str, bytes, int]]:
    """Generate the text sentence by sentence and yield progressive PCM chunks.

    ``split_chunks`` is turned off by the chat pipeline, which already hands over
    sentence-aligned units: splitting them again would cut a sentence mid-way.
    """
    use_chunking = split_chunks and (
        req.chunking if req.chunking is not None else settings.chunking.enabled
    )
    max_chars = req.max_chars or settings.chunking.max_chars
    chunks = chunk_text(text, max_chars) if use_chunking and text.strip() else ([text] if text.strip() else [])

    # Optional: native codec-window streaming through a vLLM-Omni service.
    if chunks and _vllm_eligible(settings, req):
        streamed = False
        served = _vllm_family(settings)
        async for out in _stream_vllm(settings, req, text):
            if not streamed:
                streamed = True
                # vLLM-Omni serves this family from now on: holding our own copy in
                # VRAM too would double the cost for the same family (the vLLM
                # service loads its own two stages). Drop it once, and only if the
                # first bytes actually arrived (a failed stream falls back locally).
                engine = getattr(scheduler, "engine", None)
                if engine is not None:
                    await _release_vllm_family(engine, served, f"is serving '{served}'")
            yield out
        if streamed:
            return
        # Nothing came back: `_stream_vllm` already logged the reason. Fall back to
        # the local engine (the turn still works) and count it so a misconfigured or
        # overloaded vLLM service is visible in /api/metrics.
        metrics.incr("tts_vllm_fallbacks")

    started = time.perf_counter()
    first = True
    gen_seconds = 0.0
    audio_seconds = 0.0
    for sentence in chunks:
        item = _item_from_request(settings, req, sentence)
        t_gen = time.perf_counter()
        wav, sr = await scheduler.submit_streaming(item["family"], item)
        gen_seconds += time.perf_counter() - t_gen
        audio_seconds += len(wav) / float(sr or 1)
        if first:
            metrics.observe_ttfa(time.perf_counter() - started)
            metrics.incr("tts_stream_sessions")
            first = False
        yield sentence, to_pcm16(wav), sr
    if not first:
        metrics.incr("tts_stream_chunks", len(chunks))
        if audio_seconds:
            metrics.observe_rtf(gen_seconds / audio_seconds)


def create_app(
    store: Optional[ConfigStore] = None,
    engine: Optional[TTSEngine] = None,
) -> FastAPI:
    store = store or get_store()
    settings = store.settings
    _configure_logging(settings.server.log_level)
    engine = engine or TTSEngine(settings)
    scheduler = BatchScheduler(
        engine,
        batch_size=settings.generation.batch_size,
        window_ms=settings.generation.batch_window_ms,
    )
    llm = LLMClient(settings.llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Start/stop the background workers with the application."""
        await scheduler.start()
        log.info("Gateway started (auth=%s)", app.state.settings.auth.enabled)
        tasks = [
            asyncio.create_task(_apply_startup(app)),
            asyncio.create_task(_vllm_idle_loop(app.state)),
        ]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await scheduler.stop()

    app = FastAPI(
        lifespan=lifespan,
        title="AiUdio Server",
        version=__version__,
        description=(
            "Self-hosted TTS/STT gateway. The web UI is a demo/admin front; this "
            "API is the integration contract. TTS/STT/chat are also exposed with "
            "OpenAI-compatible routes."
        ),
        openapi_tags=[
            {"name": "Service", "description": "Health, readiness and capability discovery."},
            {"name": "Access", "description": "API keys and admin sessions."},
            {"name": "Models", "description": "TTS model families, versions and voices."},
            {"name": "TTS", "description": "Text-to-speech (REST, OpenAI-compatible, WebSocket)."},
            {"name": "Avatars", "description": "Reusable cloned voices."},
            {"name": "STT", "description": "Speech-to-text (file, realtime WebSocket, diarization)."},
            {"name": "Chat", "description": "STT -> external LLM -> TTS conversation."},
            {"name": "Settings", "description": "Runtime configuration."},
        ],
    )
    app.state.store = store
    app.state.settings = settings
    app.state.engine = engine
    app.state.scheduler = scheduler
    app.state.llm = llm

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        metrics.incr("http_requests")
        if not request_authorized(request, app.state.settings):
            return JSONResponse(
                {"detail": "Authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="aiudio"'},
            )
        return await call_next(request)

    if settings.server.cors_origins:
        # Added *after* the auth middleware so CORS remains the outermost layer:
        # a browser preflight (OPTIONS) carries no credentials and must be answered
        # by CORS rather than rejected by the auth check.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------ pages
    @app.get("/health", tags=["Service"])
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "engine": engine.backend,
            "active_family": engine.active_family(),
            "loaded": engine.loaded_families(),
        }

    @app.get("/ready", tags=["Service"])
    async def ready() -> Response:
        """Readiness: the scheduler is running and the app finished starting up."""
        ready_ok = getattr(scheduler, "_task", None) is not None
        return JSONResponse(
            {"ready": ready_ok, "version": __version__},
            status_code=200 if ready_ok else 503,
        )

    @app.get("/api/metrics", tags=["Service"])
    async def api_metrics(request: Request) -> Dict[str, Any]:
        """Real-time metrics (admin-only when auth is enabled)."""
        if not is_admin(
            app.state.settings,
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
            request.query_params.get("api_key"),
        ):
            raise HTTPException(403, "Admin authentication required")
        snap = metrics.snapshot()
        snap["scheduler"] = {
            "depth": scheduler.depth(),
            "batch_size": scheduler.batch_size,
            "window_ms": round(scheduler.window * 1000, 1),
        }
        snap["models"] = {
            "loaded": engine.loaded_families(),
            "active_family": engine.active_family(),
            "residency": app.state.settings.models.residency,
        }
        return snap

    @app.get("/", tags=["Service"])
    async def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    # ------------------------------------------------------------- config API
    @app.get("/api/config", tags=["Service"])
    async def api_config() -> Dict[str, Any]:
        return _public_config(app.state.settings, engine)

    @app.get("/api/capabilities", tags=["Service"])
    async def api_capabilities() -> Dict[str, Any]:
        """Machine-readable capabilities, for dynamic clients and AI agents."""
        s = app.state.settings
        families = [
            {"family": f["family"], "backend": f["backend"], "repo": f["repo_id"]}
            for f in engine.available()
        ]
        return {
            "name": "AiUdio Server",
            "version": __version__,
            "auth": {"required": s.auth.enabled, "scheme": "basic+api_key"},
            "api": {
                "openapi": "/openapi.json",
                "docs": "/docs",
                "health": "/health",
                "ready": "/ready",
            },
            "tts": {
                "families": families,
                "clone_backend": s.models.clone_backend,
                "formats": ["wav", "pcm"],
                "sample_rate": s.audio.sample_rate,
                "streaming": "sentence-level PCM",
                "http": "/api/tts",
                "openai": "/v1/audio/speech",
                "websocket": "/ws/tts",
                "voices": "/api/voices",
            },
            "avatars": {
                "rest": "/api/avatars",
                "fingerprint": "/api/avatars/fingerprint",
            },
            "stt": {
                "enabled": s.stt.enabled,
                "http": "/v1/audio/transcriptions",
                "diarization": "/api/stt/diarize",
                "websocket": "/ws/stt",
            },
            "chat": {
                "websocket": "/ws/chat",
                "sse": "/api/chat",
                "llm": "external OpenAI-compatible",
            },
        }

    @app.get("/api/models", tags=["Models"])
    async def api_models() -> Dict[str, Any]:
        return {
            "families": engine.available(),
            "variants": [v.model_dump() for v in app.state.settings.models.variants],
            "default_size": app.state.settings.models.default_size,
            "active_family": engine.active_family(),
            "loaded": engine.loaded_families(),
        }

    @app.post("/api/models/select", tags=["Models"])
    async def api_models_select(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        """Point a family at one of its variants (e.g. 0.6B / 1.7B)."""
        _admin_ok(request)
        wanted = str(payload.get("variant") or payload.get("id") or payload.get("repo") or "").strip()
        variant = next(
            (v for v in app.state.settings.models.variants if wanted in (v.id, v.repo)),
            None,
        )
        if variant is None:
            raise HTTPException(400, f"Unknown model variant: {wanted}")
        models_update: Dict[str, Any] = {"registry": {variant.family: variant.repo}}
        if variant.label:
            # Recorded as a label only: `models.registry` is the source of truth (a
            # global size would otherwise override every family).
            models_update["default_size"] = variant.label
        try:
            updated = store.update(models=models_update)
        except Exception as exc:
            raise HTTPException(400, f"Invalid settings: {exc}") from exc
        app.state.settings = updated
        engine.settings = updated
        await asyncio.to_thread(engine.unload, variant.family)
        return {
            "family": variant.family,
            "variant": variant.id,
            "repo": variant.repo,
            "loaded": engine.loaded_families(),
        }

    @app.post("/api/models/load", tags=["Models"])
    async def api_models_load(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        _admin_ok(request)
        family = str(payload.get("family", "")).strip()
        if family not in app.state.settings.models.registry:
            raise HTTPException(400, f"Unknown family: {family}")
        try:
            await asyncio.to_thread(engine.ensure, family)
        except Exception as exc:  # pragma: no cover - requires GPU
            log.exception("Model load failed (family=%s)", family)
            raise HTTPException(503, f"Load failed: {exc}") from exc
        return {"loaded": engine.loaded_families(), "active_family": engine.active_family()}

    @app.post("/api/models/unload", tags=["Models"])
    async def api_models_unload(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        _admin_ok(request)
        family = payload.get("family")
        await asyncio.to_thread(engine.unload, family)
        return {"loaded": engine.loaded_families()}

    @app.get("/api/voices", tags=["Models"])
    async def api_voices(family: str = "custom_voice") -> Dict[str, Any]:
        refs = sorted(
            p.name for p in Path(app.state.settings.paths.reference_audio).glob("*") if p.is_file()
        )
        return {
            "speakers": await asyncio.to_thread(engine.speakers, family),
            "languages": await asyncio.to_thread(engine.languages, family),
            "reference_files": refs,
        }

    # ------------------------------------------------------------- avatars
    @app.post("/api/voices/load", tags=["Models"])
    async def api_voices_load(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        """Warm a conversation voice: load the model and prepare its clone prompt."""
        _admin_ok(request)
        avatar_id = str(payload.get("avatar") or "").strip()
        try:
            if avatar_id:
                store = AvatarStore(_avatar_dir(app.state.settings))
                avatar = store.get(avatar_id)
                wav = store.wav_path(avatar_id)
                if avatar is None or not wav.is_file():
                    raise HTTPException(404, f"Unknown voice: {avatar_id}")
                if _clone_family(app.state.settings) == "chatterbox":
                    await asyncio.to_thread(engine.ensure, "chatterbox")
                else:
                    handle = await asyncio.to_thread(engine.ensure, "base")
                    await asyncio.to_thread(
                        engine._cached_clone_prompt, handle, str(wav), avatar.ref_text or None,
                        bool(avatar.x_vector_only),
                    )
            else:
                family = str(payload.get("family") or "").strip()
                if family not in app.state.settings.models.registry:
                    raise HTTPException(400, f"Unknown family: {family}")
                await asyncio.to_thread(engine.ensure, family)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - requires GPU
            raise HTTPException(503, f"Load failed: {exc}") from exc
        return {"loaded": engine.loaded_families(), "active_family": engine.active_family()}

    @app.get("/api/avatars", tags=["Avatars"])
    async def list_avatars() -> Dict[str, Any]:
        store = AvatarStore(_avatar_dir(app.state.settings))
        return {"avatars": [asdict(a) for a in store.list()]}

    @app.post("/api/avatars/fingerprint", tags=["Avatars"])
    async def create_fingerprint(
        prompt: str = Form(...),
        language: str = Form("Auto"),
    ) -> Response:
        """Render a ~10s reference clip from a VoiceDesign description.

        All avatars are clones; this lets the user preview a designed voice and
        then save it as a clone (see ``POST /api/avatars`` with ``prompt``).
        """
        if not prompt.strip():
            raise HTTPException(400, "A description is required")
        try:
            arr, sr = await asyncio.to_thread(engine.design_reference, prompt, language)
        except Exception as exc:  # pragma: no cover - requires GPU
            raise HTTPException(503, f"Fingerprint generation failed: {exc}") from exc
        return Response(content=wav_bytes(arr, sr), media_type="audio/wav")

    @app.post("/api/avatars", tags=["Avatars"])
    async def create_avatar(
        request: Request,
        name: str = Form(...),
        prompt: str = Form(""),
        language: str = Form("Auto"),
        ref_text: str = Form(""),
        personality: str = Form(""),
        x_vector_only: bool = Form(False),
        ref_audio: Optional[UploadFile] = File(None),
        image: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
        _admin_ok(request)
        store = AvatarStore(_avatar_dir(app.state.settings))
        aid = store.new_id(name)
        avatar = Avatar(id=aid, name=name, kind="clone",
                        language=language or "Auto", prompt=prompt, ref_text=ref_text,
                        personality=personality)
        wav = store.wav_path(aid)
        try:
            if ref_audio is not None:
                data = await ref_audio.read()
                try:
                    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                except Exception as exc:
                    raise HTTPException(400, f"Unreadable reference audio: {exc}") from exc
                sf.write(str(wav), audio, int(sr))
            elif prompt.strip():
                # No reference clip: render a fingerprint from the description.
                arr, sr = await asyncio.to_thread(engine.design_reference, prompt, language)
                sf.write(str(wav), arr, int(sr))
                avatar.ref_text = design_ref_text(language)
            else:
                raise HTTPException(400, "A reference audio or a description is required")
            avatar.x_vector_only = bool(x_vector_only) or not avatar.ref_text.strip()
            if image is not None:
                store.photo_path(aid).write_bytes(await image.read())
        except HTTPException:
            store.delete(aid)
            raise
        store.save(avatar)
        await asyncio.to_thread(engine.clear_clone_prompts)
        return asdict(store.get(aid))

    @app.put("/api/avatars/{avatar_id}", tags=["Avatars"])
    async def update_avatar(
        avatar_id: str,
        request: Request,
        name: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        personality: Optional[str] = Form(None),
        ref_text: Optional[str] = Form(None),
        x_vector_only: Optional[bool] = Form(None),
        ref_audio: Optional[UploadFile] = File(None),
        image: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
        _admin_ok(request)
        store = AvatarStore(_avatar_dir(app.state.settings))
        avatar = store.get(avatar_id)
        if avatar is None:
            raise HTTPException(404, f"Unknown voice: {avatar_id}")
        if name is not None:
            avatar.name = name
        if prompt is not None:
            avatar.prompt = prompt
        if language is not None:
            avatar.language = language
        if personality is not None:
            avatar.personality = personality
        if ref_text is not None:
            avatar.ref_text = ref_text
        if x_vector_only is not None:
            avatar.x_vector_only = bool(x_vector_only)
        elif ref_text is not None:
            # Supplying a transcript implies ICL mode; clearing it falls back to x-vector.
            avatar.x_vector_only = not bool(ref_text.strip())
        if ref_audio is not None:
            data = await ref_audio.read()
            try:
                audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            except Exception as exc:
                raise HTTPException(400, f"Unreadable reference audio: {exc}") from exc
            sf.write(str(store.wav_path(avatar_id)), audio, int(sr))
        if image is not None:
            store.photo_path(avatar_id).write_bytes(await image.read())
        if not avatar.ref_text.strip():
            avatar.x_vector_only = True  # embedding-only; nothing to condition ICL on
        store.save(avatar)
        await asyncio.to_thread(engine.clear_clone_prompts)
        return asdict(store.get(avatar_id))

    @app.post("/api/avatars/{avatar_id}/regenerate", tags=["Avatars"])
    async def regenerate_avatar(avatar_id: str, request: Request) -> Dict[str, Any]:
        _admin_ok(request)
        store = AvatarStore(_avatar_dir(app.state.settings))
        avatar = store.get(avatar_id)
        if avatar is None:
            raise HTTPException(404, f"Unknown voice: {avatar_id}")
        if not avatar.prompt.strip():
            raise HTTPException(400, "This voice has no description to regenerate from")
        arr, sr = await asyncio.to_thread(engine.design_reference, avatar.prompt, avatar.language)
        sf.write(str(store.wav_path(avatar_id)), arr, int(sr))
        avatar.ref_text = design_ref_text(avatar.language)
        avatar.x_vector_only = False
        store.save(avatar)
        await asyncio.to_thread(engine.clear_clone_prompts)
        return asdict(store.get(avatar_id))

    @app.delete("/api/avatars/{avatar_id}", tags=["Avatars"])
    async def delete_avatar(avatar_id: str, request: Request) -> Dict[str, Any]:
        _admin_ok(request)
        store = AvatarStore(_avatar_dir(app.state.settings))
        store.delete(avatar_id)
        await asyncio.to_thread(engine.clear_clone_prompts)
        return {"deleted": avatar_id}

    @app.get("/api/avatars/hygiene", tags=["Avatars"])
    async def avatars_hygiene() -> Dict[str, Any]:
        """Audit the voice bank: files per kind, duplicate names, orphan files."""
        return await asyncio.to_thread(AvatarStore(_avatar_dir(app.state.settings)).hygiene)

    @app.post("/api/avatars/hygiene/clean", tags=["Avatars"])
    async def avatars_clean(request: Request) -> Dict[str, Any]:
        """Delete the files no voice entry references (a voice is never touched)."""
        _admin_ok(request)
        store = AvatarStore(_avatar_dir(app.state.settings))
        removed = await asyncio.to_thread(store.clean_orphans)
        if removed:
            log.info("Voice bank: removed %d orphan file(s): %s",
                     len(removed), ", ".join(removed[:10]))
        return {"removed": removed, "hygiene": await asyncio.to_thread(store.hygiene)}

    @app.get("/api/avatars/{avatar_id}/audio", tags=["Avatars"])
    async def avatar_audio(avatar_id: str) -> FileResponse:
        path = AvatarStore(_avatar_dir(app.state.settings)).wav_path(avatar_id)
        if not path.is_file():
            raise HTTPException(404, "No audio")
        return FileResponse(str(path), media_type="audio/wav")

    @app.get("/api/avatars/{avatar_id}/photo", tags=["Avatars"])
    async def avatar_photo(avatar_id: str) -> FileResponse:
        path = AvatarStore(_avatar_dir(app.state.settings)).photo_path(avatar_id)
        if not path.is_file():
            raise HTTPException(404, "No photo")
        return FileResponse(str(path))

    @app.put("/api/settings", tags=["Settings"])
    async def api_settings_update(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        _admin_ok(request)
        allowed = {"llm", "generation", "chunking", "audio", "stt", "ui", "startup", "models",
                   "vllm", "server", "docker"}
        sections = {k: v for k, v in payload.items() if k in allowed}
        # `server` is huge (host, port, CORS...): only the log level is editable, and
        # it is applied immediately (no restart).
        if "server" in sections:
            level = str((sections.get("server") or {}).get("log_level") or "").strip()
            sections["server"] = {"log_level": level} if level else {}
            if not sections["server"]:
                del sections["server"]
        if not sections:
            raise HTTPException(400, "No editable section provided")
        try:
            updated = store.update(**sections)
        except Exception as exc:
            raise HTTPException(400, f"Invalid settings: {exc}") from exc
        app.state.settings = updated
        app.state.llm = LLMClient(updated.llm)
        engine.settings = updated
        if "server" in sections:
            _configure_logging(updated.server.log_level)
        if "vllm" in sections:
            # Hibernation state belongs to the service as it was configured before.
            _forget_vllm_sleep()
            if updated.vllm.enabled:
                # Give the service's two stages the room they need *now* (the local
                # copy of the family it serves is redundant), then bring the
                # container in line (checkpoint, launch flags) in the background:
                # the first turn must not pay for it.
                await _release_vllm_family(engine, _vllm_family(updated), "enabled in Settings")
                asyncio.create_task(_vllm_reconcile(updated))
            else:
                # Back on the local engine: the service is stopped so its VRAM
                # returns to our models (no-op without `docker.control`).
                asyncio.create_task(_vllm_lifecycle(updated, start=False))
        if "models" in sections:
            # The registry/size may have changed: drop resident models so the next
            # request loads the newly selected repos.
            await asyncio.to_thread(engine.unload)
        return _public_config(updated, engine)

    @app.get("/api/vllm/health", tags=["Settings"])
    async def api_vllm_health(request: Request) -> Dict[str, Any]:
        """Probe the configured vLLM-Omni service (admin-only when auth is enabled)."""
        if not is_admin(
            app.state.settings,
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
            request.query_params.get("api_key"),
        ):
            raise HTTPException(403, "Admin authentication required")
        cfg = app.state.settings.vllm
        out: Dict[str, Any] = {
            "enabled": cfg.enabled, "base_url": cfg.base_url, "path": cfg.path,
            "family": _vllm_family(app.state.settings), "ok": False,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{cfg.base_url.rstrip('/')}/v1/models")
            out["status"] = resp.status_code
            out["ok"] = resp.status_code < 400
            if out["ok"]:
                models = [m.get("id") for m in (resp.json().get("data") or [])]
                out["models"] = models
                # The service loads a single checkpoint: tell the operator when it
                # is not the family this gateway is routing to it.
                served = vllm_backend.family_for_model(models[0]) if models else None
                out["served_family"] = served
                if served is not None and served != out["family"]:
                    out["warning"] = (
                        f"the service serves {served!r} but the gateway routes "
                        f"{out['family']!r}: set vllm.family to {served!r} (or serve the "
                        "matching checkpoint)"
                    )
        except Exception as exc:  # unreachable / not vLLM
            out["error"] = str(exc)
        return out

    @app.get("/api/vllm/budget", tags=["Settings"])
    async def api_vllm_budget(request: Request, model: str = "") -> Dict[str, Any]:
        """Suggested per-stage VRAM budget for a checkpoint on this card (admin).

        Answers "what should `--stage-overrides` be for this model here?" so the
        numbers do not have to be guessed per GPU. Read-only: the gateway's compose
        (or the container it pilots) stays authoritative.
        """
        _admin_ok(request)
        settings = app.state.settings
        wanted = (model or "").strip() or str(settings.vllm.model or "").strip()
        if not wanted:
            registry = settings.models.registry or {}
            wanted = str(registry.get(_vllm_family(settings), "") or "")
        out = _effective_budget(settings, wanted)
        out["suggested"] = (_vllm_budget(settings, wanted).get("suggested") or {})
        return out

    @app.get("/api/vllm/service", tags=["Settings"])
    async def api_vllm_service(request: Request) -> Dict[str, Any]:
        """State of the vLLM-Omni container (needs `docker.control`)."""
        _admin_ok(request)
        settings = app.state.settings
        cfg = settings.docker
        out: Dict[str, Any] = {
            "control": cfg.control,
            "container": cfg.container,
            "family": _vllm_family(settings),
            "enabled": settings.vllm.enabled,
            "sleeping": _VLLM_SLEEPING,
            "sleep_after": settings.vllm.sleep_after,
        }
        if not cfg.control:
            out["error"] = (
                "container control is disabled — set docker.control (or DOCKER_CONTROL=1) "
                "and mount the Docker socket in the gateway"
            )
            return out
        try:
            state = await docker_ctl.status(cfg.socket, cfg.container)
            out["state"] = state
            if state.get("exists"):
                # What the container really runs with, versus what the settings ask
                # for: the divergence is what the UI has to show.
                out["profile"] = _vllm_profile_status(
                    settings, _vllm_container_flags(state.get("command"))
                )
        except docker_ctl.DockerError as exc:
            out["error"] = str(exc)
        return out

    @app.post("/api/vllm/service/{action}", tags=["Settings"])
    async def api_vllm_service_action(action: str, request: Request) -> Dict[str, Any]:
        """Start / stop / restart the vLLM-Omni container (needs `docker.control`)."""
        _admin_ok(request)
        settings = app.state.settings
        cfg = settings.docker
        action = (action or "").strip().lower()
        if action not in ("start", "stop", "restart"):
            raise HTTPException(400, "action must be start, stop or restart")
        if not cfg.control:
            raise HTTPException(409, "Container control is disabled (docker.control)")
        try:
            operation = getattr(docker_ctl, action)
            if action == "start":
                state = await operation(cfg.socket, cfg.container, timeout=cfg.start_timeout)
            else:
                state = await operation(cfg.socket, cfg.container)
        except docker_ctl.DockerError as exc:
            raise HTTPException(502, str(exc)) from exc
        if action in ("start", "restart"):
            # vLLM needs ~1 min to bring its two stages up: prime it in the
            # background so the next turn does not pay the cold start.
            asyncio.create_task(_warm_vllm(settings, attempts=20, delay=15.0))
        if action == "stop":
            # Nothing is served while it is down; free our own copy too, otherwise
            # the VRAM stays reserved for a service that is not running.
            log.info("vLLM-Omni container stopped from the UI")
        return {"control": True, "container": cfg.container, "state": state}

    @app.put("/api/vllm/service/model", tags=["Settings"])
    async def api_vllm_service_model(
        payload: Dict[str, Any], request: Request
    ) -> Dict[str, Any]:
        """Serve another checkpoint: the container is recreated with it."""
        _admin_ok(request)
        model = str((payload or {}).get("model") or "").strip()
        if not model:
            raise HTTPException(400, "model is required")
        family = vllm_backend.family_for_model(model)
        if family is None:
            raise HTTPException(
                400, f"{model!r} is not a Qwen3-TTS CustomVoice/Base checkpoint"
            )
        cfg = app.state.settings.docker
        if not cfg.control:
            raise HTTPException(409, "Container control is disabled (docker.control)")
        # The engine decides whether the recreated container should be running:
        # preparing another checkpoint while the engine is local leaves it stopped.
        updated_enabled = bool(app.state.settings.vllm.enabled)
        try:
            values, switches = _vllm_flag_plan(app.state.settings, model)
            state = await docker_ctl.set_model(
                cfg.socket, cfg.container, model, values=values, switches=switches,
                start=updated_enabled, timeout=cfg.start_timeout,
            )
        except docker_ctl.DockerError as exc:
            raise HTTPException(502, str(exc)) from exc
        # The gateway must route the family this checkpoint serves, and remember it.
        updated = store.update(vllm={"family": family, "model": model})
        app.state.settings = updated
        engine.settings = updated
        await _release_vllm_family(engine, family, "switched to another checkpoint")
        if updated.vllm.enabled:
            asyncio.create_task(_vllm_lifecycle(updated, start=True))
        log.info("vLLM-Omni now serves %s (family %s)", model, family)
        return {
            "model": model,
            "family": family,
            "state": state,
            "vllm": updated.vllm.model_dump(),
        }

    # --------------------------------------------------- admin: keys / session
    def _admin_ok(request: Request) -> None:
        if not is_admin(
            app.state.settings,
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
            request.query_params.get("api_key"),
        ):
            raise HTTPException(403, "Admin authentication required")

    def _basic_required(request: Request) -> None:
        if not is_basic_admin(app.state.settings, request.headers.get("authorization")):
            raise HTTPException(
                401,
                "Admin (Basic) authentication required",
                headers={"WWW-Authenticate": 'Basic realm="aiudio"'},
            )

    @app.post("/api/session", tags=["Access"])
    async def create_session(request: Request) -> Dict[str, Any]:
        """Mint a short-lived admin key from the HTTP Basic credentials.

        The admin UI then uses a normal bearer key for its HTTP and (browser)
        WebSocket calls, and can log out / let it expire.
        """
        _basic_required(request)
        ttl = app.state.settings.auth.session_ttl_minutes
        raw, record = apikeys.new_key(name="admin session", ttl_minutes=ttl, admin=True)
        apikeys.add_session(record)
        log.info("Admin session key issued (ttl=%s min)", ttl)
        return {"key": raw, "id": record.id, "expires_at": record.expires_at, "admin": True}

    @app.post("/api/session/logout", tags=["Access"])
    async def close_session(request: Request) -> Dict[str, Any]:
        token = presented_key(
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
            request.query_params.get("api_key"),
        )
        record = apikeys.verify(token, apikeys.sessions()) if token else None
        if record is not None:
            apikeys.drop_session(record.id)
        return {"revoked": record is not None}

    @app.get("/api/api-keys", tags=["Access"])
    async def list_api_keys(request: Request) -> Dict[str, Any]:
        if not app.state.settings.auth.enabled:
            return {"enabled": False, "keys": []}
        _admin_ok(request)
        keys = [apikeys.public(k) for k in app.state.settings.auth.api_keys]
        keys.sort(key=lambda k: k["created_at"], reverse=True)
        return {"enabled": True, "keys": keys}

    @app.post("/api/api-keys", tags=["Access"])
    async def create_api_key(request: Request) -> Dict[str, Any]:
        _admin_ok(request)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        name = str(payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "A name is required")
        expires_at = _parse_expiry(payload.get("expires_at"), payload.get("expires_in_days"))
        raw, record = apikeys.new_key(name=name, expires_at=expires_at)
        keys = list(app.state.settings.auth.api_keys) + [record]
        app.state.settings = store.update(
            auth={"api_keys": [k.model_dump() for k in keys]}
        )
        return {"key": raw, **apikeys.public(record)}

    @app.delete("/api/api-keys/{key_id}", tags=["Access"])
    async def delete_api_key(key_id: str, request: Request) -> Dict[str, Any]:
        _admin_ok(request)
        current = app.state.settings.auth.api_keys
        keys = [k for k in current if k.id != key_id]
        if len(keys) == len(current):
            raise HTTPException(404, f"Unknown key: {key_id}")
        app.state.settings = store.update(
            auth={"api_keys": [k.model_dump() for k in keys]}
        )
        return {"deleted": key_id}

    # ------------------------------------------------------------------- TTS
    @app.post("/api/tts", tags=["TTS"])
    async def api_tts(req: TTSRequest) -> Response:
        texts = req.resolve_texts()
        if not texts:
            raise HTTPException(400, "Field 'text' is required")

        # Validate the reference audio up-front (also covers the streaming path).
        validate_media_ref(req.ref_audio, app.state.settings, app.state.settings.paths.reference_audio)

        if req.stream:
            if len(texts) > 1:
                raise HTTPException(400, "Streaming only supports a single text")
            return StreamingResponse(
                _stream_http(app.state.settings, scheduler, req, texts[0]),
                media_type="audio/pcm",
                headers={"X-Sample-Rate": str(app.state.settings.audio.sample_rate)},
            )

        try:
            items = [_item_from_request(app.state.settings, req, t) for t in texts]
            results = [await scheduler.submit(item["family"], item) for item in items]
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(503, f"Generation failed: {exc}") from exc
        fmt = req.resolve_format(app.state.settings.audio.format)
        if fmt == "pcm":
            body = b"".join(to_pcm16(wav) for wav, _sr in results)
            media = "audio/pcm"
        elif len(results) == 1:
            body = wav_bytes(results[0][0], results[0][1])
            media = "audio/wav"
        else:
            # Several texts: concatenate the *audio* and emit a single WAV file.
            # Joining the per-text WAV blobs would stack RIFF headers and only the
            # first text would play.
            sample_rate = results[0][1]
            combined = concat_audio(
                [wav for wav, _sr in results],
                sample_rate,
                app.state.settings.chunking.silence_ms,
            )
            body = wav_bytes(combined, sample_rate)
            media = "audio/wav"
        return Response(content=body, media_type=media)

    async def _stream_http(settings, sched, req, text) -> AsyncIterator[bytes]:
        try:
            async for _sentence, pcm, _sr in _stream_pcm(settings, sched, req, text):
                yield pcm
        except Exception:  # pragma: no cover - requires GPU
            log.exception("Streaming error")
            yield b""

    @app.post("/v1/audio/speech", tags=["TTS"])
    async def openai_speech(req: TTSRequest) -> Response:
        return await api_tts(req)

    @app.get("/v1/audio/voices", tags=["Models"])
    async def openai_voices() -> Dict[str, Any]:
        speakers = await asyncio.to_thread(engine.speakers, "custom_voice")
        store = AvatarStore(_avatar_dir(app.state.settings))
        avatars = [
            {"id": a.id, "name": a.name, "language": a.language} for a in store.list()
        ]
        return {"voices": speakers, "avatars": avatars}

    # -------------------------------------------------------------- WS TTS
    @app.websocket("/ws/tts")
    async def ws_tts(websocket: WebSocket) -> None:
        if not await _authorize_ws(websocket, app.state.settings):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        config: Dict[str, Any] = {}
        buffer: List[str] = []
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if (data := message.get("text")) is not None:
                    event = json.loads(data)
                    etype = event.get("type")
                    if etype == "session.config":
                        config = event
                    elif etype == "input.text":
                        buffer.append(event.get("text", ""))
                    elif etype == "input.done":
                        text = " ".join(buffer).strip()
                        buffer = []
                        await _ws_synthesize(websocket, app.state.settings, scheduler, config, text)
                elif message.get("bytes") is not None:
                    continue  # audio input is not handled for TTS
        except WebSocketDisconnect:
            return
        except Exception:  # pragma: no cover
            log.exception("TTS WebSocket error")
            try:
                await websocket.send_json({"type": "error", "message": "internal error"})
            except Exception:
                pass

    # -------------------------------------------------------------- STT proxy
    @app.get("/api/stt/health", tags=["STT"])
    async def stt_health() -> Dict[str, Any]:
        cfg = app.state.settings.stt
        if not cfg.enabled:
            return {"enabled": False}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{cfg.base_url.rstrip('/')}/health", headers=_stt_headers(cfg))
            return {"enabled": True, "status": resp.status_code, "upstream": resp.json()}
        except Exception as exc:
            return {"enabled": True, "error": str(exc)}

    @app.get("/api/stt/models", tags=["STT"])
    async def stt_models() -> Dict[str, Any]:
        cfg = app.state.settings.stt
        if not cfg.enabled:
            return {"available": False, "models": []}
        index = await asyncio.to_thread(load_index, cfg.model_index)
        if index is None:
            return {"available": False, "models": []}
        active = await asyncio.to_thread(read_active, cfg.models_dir)
        return {
            "available": True,
            "active": active,
            "models": index.with_status(cfg.models_dir, active),
        }

    @app.post("/api/stt/download", tags=["STT"])
    async def stt_download(payload: Dict[str, Any], request: Request) -> StreamingResponse:
        _admin_ok(request)
        cfg = app.state.settings.stt
        if not cfg.enabled:
            raise HTTPException(501, "STT disabled")
        spec = str(payload.get("model") or payload.get("name") or "").strip()
        index = await asyncio.to_thread(load_index, cfg.model_index)
        if index is None:
            raise HTTPException(503, "STT model index unavailable (start the stt service once)")
        model = index.find(spec) if spec else index.default_asr()
        if model is None:
            raise HTTPException(400, f"Unknown STT model: {spec}")
        artifact = model.asr_artifact()
        if artifact is None:
            raise HTTPException(400, f"No ASR artifact for {model.repo}")

        async def gen() -> AsyncIterator[str]:
            try:
                async for event in iter_download(model, artifact, cfg.models_dir, cfg.hf_base_url):
                    if event.get("status") in ("done", "cached"):
                        # Point the STT container at this model.
                        await asyncio.to_thread(mark_active, cfg.models_dir, artifact.filename)
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:  # pragma: no cover - network failure
                yield f"data: {json.dumps({'status': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/audio/transcriptions", tags=["STT"])
    async def proxy_transcription(
        file: UploadFile = File(...),
        response_format: str = Form("json"),
        language: Optional[str] = Form(None),
        diarization: bool = Form(False),
        automatic_punctuation: bool = Form(True),
    ) -> Response:
        cfg = app.state.settings.stt
        if not cfg.enabled:
            raise HTTPException(501, "STT disabled")
        content = await file.read()
        data = {
            "response_format": response_format,
            "automatic_punctuation": str(automatic_punctuation).lower(),
            "diarization": str(diarization).lower(),
        }
        if language:
            data["language"] = language
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{cfg.base_url.rstrip('/')}{cfg.transcribe_path}",
                    files={"file": (file.filename or "audio.wav", content, file.content_type or "audio/wav")},
                    data=data,
                    headers=_stt_headers(cfg),
                )
        except Exception as exc:
            raise HTTPException(502, f"STT service unavailable: {exc}") from exc
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    @app.post("/api/stt/diarize", tags=["STT"])
    async def proxy_diarize(file: UploadFile = File(...)) -> Response:
        cfg = app.state.settings.stt
        if not cfg.enabled:
            raise HTTPException(501, "STT disabled")
        content = await file.read()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{cfg.base_url.rstrip('/')}{cfg.diarize_path}",
                    files={"file": (file.filename or "audio.wav", content, file.content_type or "audio/wav")},
                    headers=_stt_headers(cfg),
                )
        except Exception as exc:
            raise HTTPException(502, f"STT service unavailable: {exc}") from exc
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    @app.websocket("/ws/stt")
    async def ws_stt(websocket: WebSocket) -> None:
        if not await _authorize_ws(websocket, app.state.settings):
            await websocket.close(code=4401)
            return
        cfg = app.state.settings.stt
        if not cfg.enabled:
            await websocket.close(code=1011)
            return
        await websocket.accept()
        try:
            url = _ws_url(cfg.base_url, cfg.realtime_path, cfg.api_key)
            upstream = await _open_stt_upstream(url)
            metrics.incr("stt_ws_opened")
            try:
                async with upstream:
                    await _relay(websocket, upstream)
            finally:
                metrics.incr("stt_ws_closed")
        except Exception as exc:  # pragma: no cover - depends on the STT service
            log.exception("STT WebSocket proxy error")
            try:
                await websocket.send_json({"type": "error", "message": f"STT service unavailable: {exc}"})
            except Exception:
                pass

    # ------------------------------------------------------------- LLM / chat
    @app.get("/api/llm/ping", tags=["Chat"])
    async def llm_ping() -> Dict[str, Any]:
        return await app.state.llm.ping()

    @app.post("/api/chat", tags=["Chat"])
    async def api_chat(payload: Dict[str, Any]) -> StreamingResponse:
        text = str(payload.get("text") or payload.get("input") or "").strip()
        history = payload.get("history") or []
        messages = build_messages(
            app.state.settings.llm.system_prompt,
            trim_history(history, app.state.settings.llm.history_turns),
            text,
        )

        async def gen() -> AsyncIterator[str]:
            try:
                async for token in app.state.llm.stream(messages):
                    yield f"data: {json.dumps({'delta': token})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        if not await _authorize_ws(websocket, app.state.settings):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        history: List[Dict[str, str]] = []
        tts_config: Dict[str, Any] = {}
        personality = ""
        speak = True
        turn: Optional[asyncio.Task] = None
        send_lock = asyncio.Lock()

        async def send(obj: Dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(obj)

        async def run_turn(user_text: str, turn_id: Optional[int] = None) -> None:
            system = app.state.settings.llm.system_prompt
            if personality:
                system = f"{system}\n\n{personality}" if system else personality
            messages = build_messages(
                system,
                trim_history(history, app.state.settings.llm.history_turns),
                user_text,
            )
            try:
                reply = await _chat_turn(
                    websocket, app.state.settings, scheduler, app.state.llm,
                    tts_config, messages, speak, send_lock, turn=turn_id,
                )
            except asyncio.CancelledError:
                await send({"type": "turn.cancelled", "turn": turn_id})
                raise
            except Exception as exc:
                log.exception("Chat turn failed")
                await send({"type": "error", "message": str(exc), "turn": turn_id})
                return
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            await send({"type": "llm.done", "text": reply, "turn": turn_id})

        def cancel_turn() -> None:
            if turn is not None and not turn.done():
                turn.cancel()

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if (data := message.get("text")) is None:
                    continue
                event = json.loads(data)
                etype = event.get("type")
                if etype == "session.config":
                    tts_config = event.get("tts") or {}
                    speak = bool(event.get("speak", True))
                    personality = str(event.get("personality") or "")
                elif etype == "input.cancel":
                    cancel_turn()
                elif etype == "input.text":
                    user_text = str(event.get("text", "")).strip()
                    if not user_text:
                        continue
                    cancel_turn()
                    # Optional client turn id, echoed on every event of the turn so
                    # a late `turn.cancelled`/`error` from the previous turn cannot
                    # be mistaken for the current one.
                    turn_id = event.get("turn")
                    if not isinstance(turn_id, int):
                        turn_id = None
                    await send({"type": "transcript", "text": user_text, "turn": turn_id})
                    turn = asyncio.create_task(run_turn(user_text, turn_id))
        except WebSocketDisconnect:
            cancel_turn()
            return
        except Exception:  # pragma: no cover
            log.exception("Chat WebSocket error")
        finally:
            cancel_turn()

    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

    return app


# --------------------------------------------------------------------- helpers
async def _authorize_ws(websocket: WebSocket, settings: Settings) -> bool:
    if not settings.auth.enabled:
        return True
    from .security import _authorized

    return _authorized(
        settings,
        websocket.headers.get("authorization"),
        websocket.headers.get("x-api-key"),
        websocket.query_params.get("api_key"),
    )


def _stt_headers(cfg) -> Dict[str, str]:
    return {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}


def _ws_url(base_url: str, path: str, api_key: str = "") -> str:
    url = base_url.rstrip("/")
    url = "wss://" + url[len("https://"):] if url.startswith("https://") else url.replace("http://", "ws://", 1)
    url = f"{url}{path}"
    if api_key:
        url += f"?api_key={api_key}"
    return url


async def _warm_vllm(settings: Settings, attempts: int = 8, delay: float = 15.0) -> bool:
    """Best effort: absorb the vLLM-Omni cold start.

    The first call after a restart can take tens of seconds (kernel/graph warm-up
    and the speech caches being filled): a tiny request primes it so the first
    real turn is not the one paying for it.

    The gateway usually starts before the vLLM service is listening (vLLM needs
    ~1 min), so the probe retries instead of failing once. Returns True on success;
    an unreachable service is only worth a warning, never an error.
    """
    cfg = settings.vllm
    if not cfg.enabled:
        return False
    family = _vllm_family(settings)
    req = TTSRequest(text="a.")
    if family == "base":
        # Cloning needs a reference clip: use any voice of the bank for the probe
        # (the caches it fills are shared with the real ones).
        store = AvatarStore(_avatar_dir(settings))
        avatar = next((a for a in store.list() if store.wav_path(a.id).is_file()), None)
        if avatar is None:
            log.info(
                "vLLM-Omni warm-up skipped: the service serves 'base' and the voice "
                "bank has no clip to use as a reference"
            )
            return False
        req = TTSRequest(text="a.", avatar=avatar.id)
    payload = _vllm_payload(settings, req, "a.")
    if payload is None:  # pragma: no cover - guarded above
        return False
    for attempt in range(1, max(1, attempts) + 1):
        try:
            started = time.perf_counter()
            async for _ in vllm_backend.stream_pcm(
                cfg.base_url, payload, path=cfg.path, timeout=180.0
            ):
                pass
            metrics.incr("tts_vllm_warmups")
            log.info("vLLM-Omni warm-up done in %.2fs", time.perf_counter() - started)
            return True
        except Exception as exc:
            if attempt >= attempts:
                log.warning(
                    "vLLM-Omni warm-up gave up after %d attempts (%s): the service at %s "
                    "is not reachable; turns will fall back to the local engine. Check it "
                    "with GET /api/vllm/health.",
                    attempts, exc, cfg.base_url,
                )
                return False
            log.debug("vLLM-Omni not ready (attempt %d/%d): %s", attempt, attempts, exc)
            await asyncio.sleep(delay)
    return False


async def _open_stt_upstream(url: str):
    """Open the STT realtime socket, tolerating a slow or busy upstream.

    NeMo-Speech.cpp serves one realtime session at a time and can need a moment
    to release the previous one; the library's default 10 s handshake timeout is
    then too short, and a single attempt surfaces as an opaque backend error.
    """
    import websockets

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            return await websockets.connect(url, max_size=None, open_timeout=30.0)
        except (TimeoutError, OSError, websockets.exceptions.WebSocketException) as exc:
            last_exc = exc
            log.warning(
                "STT upstream handshake failed (attempt %d/2, url=%s): %s",
                attempt + 1, url, exc,
            )
            if attempt == 0:
                await asyncio.sleep(0.5)
    metrics.incr("stt_ws_failed")
    raise last_exc if last_exc is not None else RuntimeError("STT upstream unavailable")


async def _relay(client: WebSocket, upstream) -> None:
    async def client_to_upstream() -> None:
        while True:
            message = await client.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (text := message.get("text")) is not None:
                await upstream.send(text)
            elif (binary := message.get("bytes")) is not None:
                await upstream.send(binary)

    async def upstream_to_client() -> None:
        async for message in upstream:
            if isinstance(message, bytes):
                await client.send_bytes(message)
            else:
                await client.send_text(message)

    tasks = [asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())]
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    # Await the cancellations so the upstream session is fully released before the
    # caller closes it (a lingering reader would keep the STT service busy).
    await asyncio.gather(*pending, return_exceptions=True)


async def _ws_synthesize(websocket: WebSocket, settings, scheduler, config, text: str) -> None:
    if not text:
        await websocket.send_json({"type": "session.done", "total_sentences": 0})
        return
    req = TTSRequest(**{k: v for k, v in config.items() if k in TTSRequest.model_fields})
    req.text = text
    index = 0
    sample_rate = settings.audio.sample_rate
    async for sentence, pcm, sr in _stream_pcm(settings, scheduler, req, text):
        sample_rate = sr
        await websocket.send_json({
            "type": "audio.start", "sentence_index": index,
            "sentence_text": sentence, "format": "pcm", "sample_rate": sr,
        })
        await websocket.send_bytes(pcm)
        await websocket.send_json({
            "type": "audio.done", "sentence_index": index, "total_bytes": len(pcm),
        })
        index += 1
    await websocket.send_json({
        "type": "session.done", "total_sentences": index, "sample_rate": sample_rate,
    })


_CHAT_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_CHAT_CLAUSE_RE = re.compile(r"[,;:]\s+")

# Chat TTS releases whole sentences only. A fragment cut mid-sentence (at a
# clause, let alone a single word) is synthesized as a standalone utterance with
# its own intonation, which is heard as chopped, word-by-word speech.
_CHAT_MIN_CHARS = 40          # a complete sentence is released past this length
_CHAT_FIRST_CHARS = 30        # the first chunk may be cut earlier (comma/word end)
_CHAT_COALESCE_CHARS = 200    # later chunks: group sentences up to this length
_CHAT_MAX_CHARS = 240         # last resort for a sentence that never ends


def _last_boundary(text: str, within: int) -> int:
    """Index just past the last clause/space boundary within ``text[:within]``."""
    window = text[:within]
    cut = max(
        window.rfind(", "), window.rfind("; "), window.rfind(": "),
        window.rfind("— "), window.rfind("– "), window.rfind(" "),
    )
    return cut + 1 if cut > 0 else 0


def _pop_chat_chunks(
    text: str,
    max_chars: int = _CHAT_MAX_CHARS,
    min_chars: int = _CHAT_MIN_CHARS,
    early_first: bool = False,
) -> Tuple[List[str], str]:
    """Split streaming LLM text into sentence-aligned TTS chunks.

    Only complete sentences are released; sentences shorter than ``min_chars`` are
    merged with the following ones (a tiny first chunk such as "Bonjour !" would
    otherwise drain the playback buffer). The caller raises ``min_chars`` after the
    first chunk so the rest of the reply is grouped into fewer, longer utterances:
    the first sentence starts the audio early, the following ones keep a natural
    prosody.

    ``early_first`` (used for the *first* chunk of a turn only) allows one earlier
    release so the voice can start while the LLM is still writing: at the first
    clause (comma, semicolon, colon) past ``_CHAT_FIRST_CHARS``. Without such a
    clause the first complete sentence is used instead — a word-end cut was tried
    and the engine gives that fragment a closing intonation, which is heard as a
    chopped phrase even when it happens only once.

    A sentence that never ends (no punctuation) is cut at the last clause/space
    boundary past ``max_chars`` — a last resort, so a long paragraph cannot stall
    the audio forever.
    """
    chunks: List[str] = []
    start = 0
    for match in _CHAT_SENTENCE_RE.finditer(text):
        end = match.start()  # end of the sentence (before the separating whitespace)
        if len(text[start:end].strip()) >= min_chars:
            chunks.append(text[start:end].strip())
            start = match.end()
    text = text[start:]
    if early_first and not chunks:
        # First chunk only: a comma is a genuine prosodic break, so releasing it
        # early is free. Cutting at a word end (no comma) was tried and sounds
        # wrong: the engine gives the fragment a closing intonation, mid-sentence.
        # Without a comma we therefore wait for the first complete sentence.
        window = text[:max_chars]
        cut = next(
            (m.start() + 1 for m in _CHAT_CLAUSE_RE.finditer(window)
             if m.start() >= _CHAT_FIRST_CHARS),
            0,
        )
        if cut:
            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()
    while len(text) > max_chars:
        cut = _last_boundary(text, max_chars)
        if cut <= 0:
            break
        chunks.append(text[:cut].strip())
        text = text[cut:].lstrip()
    return [c for c in chunks if c], text


async def _chat_turn(
    websocket, settings, scheduler, llm, tts_config, messages,
    speak: bool = True, send_lock: Optional[asyncio.Lock] = None,
    turn: Optional[int] = None,
) -> str:
    req = TTSRequest(**{k: v for k, v in (tts_config or {}).items() if k in TTSRequest.model_fields})
    send_lock = send_lock or asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    reply = ""
    buffer = ""
    index = 0
    queued = False
    started = time.perf_counter()
    first_token: Optional[float] = None
    last_token: Optional[float] = None
    first_queued: Optional[float] = None
    first_audio: Optional[float] = None
    spoken = 0
    audio_total = 0.0
    worst_margin: Optional[float] = None

    async def send_json(obj: Dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(obj)

    async def tts_worker() -> None:
        """Synthesize queued sentences without blocking the LLM token stream."""
        nonlocal index, first_audio, spoken, audio_total, worst_margin
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            sentence = sentence.strip()
            # Nothing to say: the sentence splitter leaves a lone final "." (or "…")
            # in the buffer, and synthesizing it costs a full call (and can emit a
            # stray blip) for no audible content.
            if not sentence or not any(ch.isalnum() for ch in sentence):
                if sentence:
                    log.debug("chat tts: skipping punctuation-only chunk %r", sentence)
                continue
            try:
                # The chat chunker already released whole sentences: synthesize each
                # one as it is (no second sentence-splitting pass).
                t_synth = time.perf_counter()
                chunk_audio = 0.0
                async for _, pcm, sr in _stream_pcm(
                    settings, scheduler, req, sentence, split_chunks=False
                ):
                    # Keep start/pcm/done atomic so concurrent deltas cannot interleave.
                    async with send_lock:
                        await websocket.send_json({
                            "type": "audio.start", "sentence_index": index,
                            "sentence_text": sentence, "format": "pcm", "sample_rate": sr,
                            "turn": turn,
                        })
                        await websocket.send_bytes(pcm)
                        await websocket.send_json({
                            "type": "audio.done", "sentence_index": index, "total_bytes": len(pcm),
                            "turn": turn,
                        })
                    index += 1
                    chunk_audio += len(pcm) / (2.0 * (sr or 1))  # mono PCM16
                spoken += 1
                now = time.perf_counter() - started
                synth = time.perf_counter() - t_synth
                audio_total += chunk_audio
                # How much audio the client still has in reserve when this chunk
                # arrives: <= 0 means it already drained its buffer (audible gap).
                margin = audio_total - (now - (first_audio or now))
                if first_audio is not None:
                    worst_margin = margin if worst_margin is None else min(worst_margin, margin)
                if first_audio is None:
                    first_audio = now
                    log.info(
                        "chat tts: first audio %.2fs after the turn started (llm first "
                        "token %s, first chunk queued %s, synth %.2fs, %d chars, %.2fs "
                        "of audio, margin %+.2fs)",
                        now, _fmt_s(first_token), _fmt_s(first_queued), synth,
                        len(sentence), chunk_audio, margin,
                    )
                else:
                    log.info(
                        "chat tts: chunk %d ready %.2fs after the turn started (synth "
                        "%.2fs, %d chars, +%.2fs, %.2fs buffered, margin %+.2fs)",
                        spoken - 1, now, synth, len(sentence), chunk_audio,
                        audio_total, margin,
                    )
            except Exception as exc:  # TTS failure must not break the text conversation
                await send_json({"type": "error", "message": f"TTS: {exc}", "turn": turn})

    worker = asyncio.create_task(tts_worker()) if speak else None
    try:
        async for token in llm.stream(messages):
            if first_token is None:
                first_token = time.perf_counter() - started
            last_token = time.perf_counter() - started
            reply += token
            await send_json({"type": "llm.delta", "text": token, "turn": turn})
            if speak:
                buffer += token
                # Low latency first: the first chunk starts at the first sentence (or
                # a long opening clause). Afterwards, release each complete sentence
                # as it arrives *while the engine is idle*: the client plays whatever
                # it has already received, so releasing promptly is what keeps
                # playback gap-free. Only when the engine already has work queued do
                # we group sentences (`_CHAT_COALESCE_CHARS`) to limit the number of
                # synthesis calls (a fast LLM can otherwise release many small ones).
                busy = queue.qsize() > 0
                minimum = _CHAT_COALESCE_CHARS if (queued and busy) else _CHAT_MIN_CHARS
                chunks, buffer = _pop_chat_chunks(
                    buffer, min_chars=minimum, early_first=not queued
                )
                for sentence in chunks:
                    if first_queued is None:
                        first_queued = time.perf_counter() - started
                    queued = True
                    await queue.put(sentence)
        if speak and worker is not None:
            if buffer.strip():
                if first_queued is None:
                    first_queued = time.perf_counter() - started
                await queue.put(buffer)
            await queue.put(None)
            await worker
    finally:
        if worker is not None and not worker.done():
            worker.cancel()
    log.info(
        "chat turn: reply %d chars, llm first token %s, last token %s, stream closed %.2fs "
        "(%+.2fs after the last token) | first chunk queued %s, first audio %s "
        "(%+.2fs vs the last token) | %d chunks, %.2fs of audio, worst margin %+.2fs, "
        "total %.2fs",
        len(reply), _fmt_s(first_token), _fmt_s(last_token), time.perf_counter() - started,
        (time.perf_counter() - started) - (last_token or 0.0),
        _fmt_s(first_queued), _fmt_s(first_audio),
        (first_audio or 0.0) - (last_token or 0.0),
        spoken, audio_total, worst_margin or 0.0,
        time.perf_counter() - started,
    )
    return reply


app = create_app()

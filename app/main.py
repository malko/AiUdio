"""FastAPI application: REST/OpenAI API, TTS/chat WebSocket, STT and LLM proxies.

The server is a gateway: it runs Qwen3-TTS locally, delegates STT to the
NeMo-Speech.cpp service and conversation to an external LLM (OpenAI-compatible).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, apikeys, vllm_backend
from .audio import to_pcm16, wav_bytes
from .avatars import Avatar, AvatarStore
from .chatterbox_backend import is_available as is_chatterbox_available
from .chunking import chunk_text
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


def _clone_family(settings: Settings) -> str:
    """Family used to synthesize cloned avatars (global clone-engine setting)."""
    return "chatterbox" if settings.models.clone_backend == "chatterbox" else "base"


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
    if voice:
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

    family = settings.startup.tts_family
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
        "auth_required": settings.auth.enabled,
        "models": {
            "families": engine.available(),
            "variants": [v.model_dump() for v in settings.models.variants],
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


def _vllm_eligible(settings: Settings, req: TTSRequest) -> bool:
    """vLLM-Omni serves the CustomVoice family only (no clone/design)."""
    if not settings.vllm.enabled:
        return False
    if req.avatar:
        return False
    speaker = req.resolve_speaker()
    if speaker and AvatarStore(_avatar_dir(settings)).get(speaker) is not None:
        return False
    return req.resolve_family(settings.models.default_family) == "custom_voice"


async def _stream_vllm(
    settings: Settings, req: TTSRequest, text: str
) -> AsyncIterator[Tuple[str, bytes, int]]:
    """Yield native PCM from vLLM-Omni; yields nothing (to fall back) on failure."""
    payload = vllm_backend.speech_payload(req, text)
    started = time.perf_counter()
    stream = vllm_backend.stream_pcm(settings.vllm.base_url, payload)
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
        async for out in _stream_vllm(settings, req, text):
            streamed = True
            yield out
        if streamed:
            return

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

    app = FastAPI(
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

    if settings.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

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

    @app.on_event("startup")
    async def _startup() -> None:
        await scheduler.start()
        log.info("Gateway started (auth=%s)", app.state.settings.auth.enabled)
        asyncio.create_task(_apply_startup(app))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await scheduler.stop()

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
    async def api_models_select(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Set the global model size (0.6B/1.7B) from a variant id."""
        wanted = str(payload.get("variant") or payload.get("id") or payload.get("repo") or "").strip()
        variant = next(
            (v for v in app.state.settings.models.variants if wanted in (v.id, v.repo)),
            None,
        )
        if variant is None:
            raise HTTPException(400, f"Unknown model variant: {wanted}")
        models_update: Dict[str, Any] = {"registry": {variant.family: variant.repo}}
        if variant.label:
            # The size is a global setting: keep it in sync so the choice applies to
            # every family and survives a restart.
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
    async def api_models_load(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    async def api_models_unload(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    async def api_voices_load(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Warm a conversation voice: load the model and prepare its clone prompt."""
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
        name: str = Form(...),
        prompt: str = Form(""),
        language: str = Form("Auto"),
        ref_text: str = Form(""),
        personality: str = Form(""),
        x_vector_only: bool = Form(False),
        ref_audio: Optional[UploadFile] = File(None),
        image: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
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
        name: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        personality: Optional[str] = Form(None),
        ref_text: Optional[str] = Form(None),
        x_vector_only: Optional[bool] = Form(None),
        ref_audio: Optional[UploadFile] = File(None),
        image: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
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
    async def regenerate_avatar(avatar_id: str) -> Dict[str, Any]:
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
    async def delete_avatar(avatar_id: str) -> Dict[str, Any]:
        store = AvatarStore(_avatar_dir(app.state.settings))
        store.delete(avatar_id)
        await asyncio.to_thread(engine.clear_clone_prompts)
        return {"deleted": avatar_id}

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
    async def api_settings_update(payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"llm", "generation", "chunking", "audio", "stt", "ui", "startup", "models", "vllm"}
        sections = {k: v for k, v in payload.items() if k in allowed}
        if not sections:
            raise HTTPException(400, "No editable section provided")
        try:
            updated = store.update(**sections)
        except Exception as exc:
            raise HTTPException(400, f"Invalid settings: {exc}") from exc
        app.state.settings = updated
        app.state.llm = LLMClient(updated.llm)
        engine.settings = updated
        if "models" in sections:
            # The registry/size may have changed: drop resident models so the next
            # request loads the newly selected repos.
            await asyncio.to_thread(engine.unload)
        return _public_config(updated, engine)

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
        out_bytes: List[bytes] = [
            to_pcm16(wav) if fmt == "pcm" else wav_bytes(wav, sample_rate)
            for wav, sample_rate in results
        ]
        body = b"".join(out_bytes)
        media = "audio/pcm" if fmt == "pcm" else "audio/wav"
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
    async def stt_download(payload: Dict[str, Any]) -> StreamingResponse:
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

        async def run_turn(user_text: str) -> None:
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
                    tts_config, messages, speak, send_lock,
                )
            except asyncio.CancelledError:
                await send({"type": "turn.cancelled"})
                raise
            except Exception as exc:
                log.exception("Chat turn failed")
                await send({"type": "error", "message": str(exc)})
                return
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            await send({"type": "llm.done", "text": reply})

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
                    await send({"type": "transcript", "text": user_text})
                    turn = asyncio.create_task(run_turn(user_text))
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
_CHAT_MIN_CHARS = 40          # first chunk: one complete sentence past this length
_CHAT_FIRST_CLAUSE_CHARS = 40  # ... or a clause past this one, so audio starts sooner
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
    allow_first_clause: bool = False,
) -> Tuple[List[str], str]:
    """Split streaming LLM text into sentence-aligned TTS chunks.

    Only complete sentences are released; sentences shorter than ``min_chars`` are
    merged with the following ones (a tiny first chunk such as "Bonjour !" would
    otherwise drain the playback buffer). The caller raises ``min_chars`` after the
    first chunk so the rest of the reply is grouped into fewer, longer utterances:
    the first sentence starts the audio early, the following ones keep a natural
    prosody.

    ``allow_first_clause`` lets the *first* chunk of a turn start even earlier: if
    the opening sentence is still running, it is cut at the first clause (comma,
    semicolon, colon) past ``_CHAT_FIRST_CLAUSE_CHARS`` — a natural prosodic break,
    unlike the old word-boundary cut that made the voice sound chopped.

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
    if allow_first_clause and not chunks:
        for match in _CHAT_CLAUSE_RE.finditer(text[:max_chars]):
            if match.start() >= _CHAT_FIRST_CLAUSE_CHARS:
                chunks.append(text[: match.start() + 1].strip())
                text = text[match.end():]
                break
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
                        })
                        await websocket.send_bytes(pcm)
                        await websocket.send_json({
                            "type": "audio.done", "sentence_index": index, "total_bytes": len(pcm),
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
                await send_json({"type": "error", "message": f"TTS: {exc}"})

    worker = asyncio.create_task(tts_worker()) if speak else None
    try:
        async for token in llm.stream(messages):
            if first_token is None:
                first_token = time.perf_counter() - started
            last_token = time.perf_counter() - started
            reply += token
            await send_json({"type": "llm.delta", "text": token})
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
                    buffer, min_chars=minimum, allow_first_clause=not queued
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

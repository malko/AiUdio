"""FastAPI application: REST/OpenAI API, TTS/chat WebSocket, STT and LLM proxies.

The server is a gateway: it runs Qwen3-TTS locally, delegates STT to the
NeMo-Speech.cpp service and conversation to an external LLM (OpenAI-compatible).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .audio import to_pcm16, wav_bytes
from .chunking import chunk_text
from .config import ConfigStore, Settings, get_store
from .engine import TTSEngine
from .llm import LLMClient, build_messages, trim_history
from .scheduler import BatchScheduler
from .schemas import TTSRequest
from .security import request_authorized, validate_media_ref
from .stt_models import iter_download, load_index

log = logging.getLogger(__name__)
UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def _public_config(settings: Settings, engine: TTSEngine) -> Dict[str, Any]:
    return {
        "version": __version__,
        "ui": settings.ui.model_dump(),
        "auth_required": settings.auth.enabled,
        "models": {
            "families": engine.available(),
            "default_family": settings.models.default_family,
            "residency": settings.models.residency,
        },
        "audio": settings.audio.model_dump(),
        "chunking": settings.chunking.model_dump(),        "generation": settings.generation.model_dump(),
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
        "languages": [
            "Auto", "Chinese", "English", "Japanese", "Korean", "German",
            "French", "Russian", "Portuguese", "Spanish", "Italian",
        ],
    }


def _item_from_request(settings: Settings, req: TTSRequest, text: str) -> Dict[str, Any]:
    family = req.resolve_family(settings.models.default_family)
    gen = req.generation.to_kwargs() if req.generation else {}
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


async def _stream_pcm(
    settings: Settings, scheduler: BatchScheduler, req: TTSRequest, text: str
) -> AsyncIterator[Tuple[str, bytes, int]]:
    """Generate the text sentence by sentence and yield progressive PCM chunks."""
    use_chunking = req.chunking if req.chunking is not None else settings.chunking.enabled
    max_chars = req.max_chars or settings.chunking.max_chars
    chunks = chunk_text(text, max_chars) if use_chunking and text.strip() else ([text] if text.strip() else [])
    for sentence in chunks:
        item = _item_from_request(settings, req, sentence)
        wav, sr = await scheduler.submit(item["family"], item)
        yield sentence, to_pcm16(wav), sr


def create_app(
    store: Optional[ConfigStore] = None,
    engine: Optional[TTSEngine] = None,
) -> FastAPI:
    store = store or get_store()
    settings = store.settings
    engine = engine or TTSEngine(settings)
    scheduler = BatchScheduler(
        engine,
        batch_size=settings.generation.batch_size,
        window_ms=settings.generation.batch_window_ms,
    )
    llm = LLMClient(settings.llm)

    app = FastAPI(title="AiUdio Server", version=__version__)
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

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await scheduler.stop()

    # ------------------------------------------------------------------ pages
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "engine": engine.backend,
            "active_family": engine.active_family(),
            "loaded": engine.loaded_families(),
        }

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    # ------------------------------------------------------------- config API
    @app.get("/api/config")
    async def api_config() -> Dict[str, Any]:
        return _public_config(app.state.settings, engine)

    @app.get("/api/models")
    async def api_models() -> Dict[str, Any]:
        return {
            "families": engine.available(),
            "active_family": engine.active_family(),
            "loaded": engine.loaded_families(),
        }

    @app.post("/api/models/load")
    async def api_models_load(payload: Dict[str, Any]) -> Dict[str, Any]:
        family = str(payload.get("family", "")).strip()
        if family not in app.state.settings.models.registry:
            raise HTTPException(400, f"Unknown family: {family}")
        try:
            await asyncio.to_thread(engine.ensure, family)
        except Exception as exc:  # pragma: no cover - requires GPU
            raise HTTPException(503, f"Load failed: {exc}") from exc
        return {"loaded": engine.loaded_families(), "active_family": engine.active_family()}

    @app.post("/api/models/unload")
    async def api_models_unload(payload: Dict[str, Any]) -> Dict[str, Any]:
        family = payload.get("family")
        await asyncio.to_thread(engine.unload, family)
        return {"loaded": engine.loaded_families()}

    @app.get("/api/voices")
    async def api_voices(family: str = "custom_voice") -> Dict[str, Any]:
        refs = sorted(
            p.name for p in Path(app.state.settings.paths.reference_audio).glob("*") if p.is_file()
        )
        return {
            "speakers": await asyncio.to_thread(engine.speakers, family),
            "languages": await asyncio.to_thread(engine.languages, family),
            "reference_files": refs,
        }

    @app.put("/api/settings")
    async def api_settings_update(payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"llm", "generation", "chunking", "audio", "stt", "ui"}
        sections = {k: v for k, v in payload.items() if k in allowed}
        if not sections:
            raise HTTPException(400, "No editable section provided")
        updated = store.update(**sections)
        app.state.settings = updated
        app.state.llm = LLMClient(updated.llm)
        return _public_config(updated, engine)

    # ------------------------------------------------------------------- TTS
    @app.post("/api/tts")
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

    @app.post("/v1/audio/speech")
    async def openai_speech(req: TTSRequest) -> Response:
        return await api_tts(req)

    @app.get("/v1/audio/voices")
    async def openai_voices() -> Dict[str, Any]:
        speakers = await asyncio.to_thread(engine.speakers, "custom_voice")
        return {"voices": speakers}

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
    @app.get("/api/stt/health")
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

    @app.get("/api/stt/models")
    async def stt_models() -> Dict[str, Any]:
        cfg = app.state.settings.stt
        if not cfg.enabled:
            return {"available": False, "models": []}
        index = await asyncio.to_thread(load_index, cfg.model_index)
        if index is None:
            return {"available": False, "models": []}
        return {"available": True, "models": index.with_status(cfg.models_dir)}

    @app.post("/api/stt/download")
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
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:  # pragma: no cover - network failure
                yield f"data: {json.dumps({'status': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/audio/transcriptions")
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
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{cfg.base_url.rstrip('/')}{cfg.transcribe_path}",
                files={"file": (file.filename or "audio.wav", content, file.content_type or "audio/wav")},
                data=data,
                headers=_stt_headers(cfg),
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    @app.post("/api/stt/diarize")
    async def proxy_diarize(file: UploadFile = File(...)) -> Response:
        cfg = app.state.settings.stt
        if not cfg.enabled:
            raise HTTPException(501, "STT disabled")
        content = await file.read()
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{cfg.base_url.rstrip('/')}{cfg.diarize_path}",
                files={"file": (file.filename or "audio.wav", content, file.content_type or "audio/wav")},
                headers=_stt_headers(cfg),
            )
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
            import websockets

            url = _ws_url(cfg.base_url, cfg.realtime_path, cfg.api_key)
            async with websockets.connect(url, max_size=None) as upstream:
                await _relay(websocket, upstream)
        except Exception:  # pragma: no cover - depends on the STT service
            log.exception("STT WebSocket proxy error")

    # ------------------------------------------------------------- LLM / chat
    @app.get("/api/llm/ping")
    async def llm_ping() -> Dict[str, Any]:
        return await app.state.llm.ping()

    @app.post("/api/chat")
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
                    tts_config = event.get("tts") or event
                elif etype == "input.text":
                    user_text = str(event.get("text", "")).strip()
                    if not user_text:
                        continue
                    await websocket.send_json({"type": "transcript", "text": user_text})
                    messages = build_messages(
                        app.state.settings.llm.system_prompt,
                        trim_history(history, app.state.settings.llm.history_turns),
                        user_text,
                    )
                    reply = await _chat_turn(
                        websocket, app.state.settings, scheduler, app.state.llm, tts_config, messages
                    )
                    history.append({"role": "user", "content": user_text})
                    history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({"type": "llm.done", "text": reply})
        except WebSocketDisconnect:
            return
        except Exception:  # pragma: no cover
            log.exception("Chat WebSocket error")

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


async def _chat_turn(websocket, settings, scheduler, llm, tts_config, messages) -> str:
    import re

    req = TTSRequest(**{k: v for k, v in (tts_config or {}).items() if k in TTSRequest.model_fields})
    sentence_re = re.compile(r"(?<=[.!?…])\s+")
    buffer = ""
    reply = ""
    index = 0

    async def flush(sentence: str) -> None:
        nonlocal index
        sentence = sentence.strip()
        if not sentence:
            return
        async for _, pcm, sr in _stream_pcm(settings, scheduler, req, sentence):
            await websocket.send_json({
                "type": "audio.start", "sentence_index": index,
                "sentence_text": sentence, "format": "pcm", "sample_rate": sr,
            })
            await websocket.send_bytes(pcm)
            await websocket.send_json({
                "type": "audio.done", "sentence_index": index, "total_bytes": len(pcm),
            })
            index += 1

    async for token in llm.stream(messages):
        reply += token
        buffer += token
        await websocket.send_json({"type": "llm.delta", "text": token})
        parts = sentence_re.split(buffer)
        if len(parts) > 1:
            buffer = parts[-1]
            for sentence in parts[:-1]:
                await flush(sentence)
    if buffer.strip():
        await flush(buffer)
    return reply


app = create_app()

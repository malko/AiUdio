# Architecture

## Components

| Service | Image | Role |
|---|---|---|
| `aiudio` | `aiudio-server:local` (built on official PyTorch/ROCm/Python base) | Qwen3-TTS engine + REST/OpenAI API + WebSocket + UI + STT/LLM proxies |
| `stt` | official NGC `nvcr.io/nvidia/nemo-speech.cpp:0.2.0` (non-NVIDIA: built from source) | offline/streaming ASR + diarization |
| `tts-vllm` | `vllm/vllm-omni` (official, `vllm` profile) | experimental native PCM streaming |
| LLM | *outside compose* | external OpenAI-compatible endpoint |

All services share the Docker network `aiudio-net`; they are reachable by name
(`http://aiudio:8004`, `http://stt:8080`).

## Flows

### Simple TTS (non-streaming)
1. Client → `POST /api/tts` (or `/v1/audio/speech`).
2. `TTSRequest` → one `SynthItem` per text.
3. `BatchScheduler.submit()` queues the item; the background task groups requests of the
   same family within a window (`generation.batch_window_ms`, `batch_size`).
4. `TTSEngine.synthesize_batch()` (dedicated thread) dispatches by backend
   (`models.backend_for(family)`): Qwen families call `generate_custom_voice` /
   `generate_voice_design` / `generate_voice_clone` with lists; Chatterbox families call the
   isolated worker one text at a time (no batch API).
5. WAV or PCM16 encoding → HTTP response.

### Streaming TTS (perceived)
`stream: true` → `chunk_text()` splits into sentences; each sentence is synthesized and
emitted as PCM16 through a `StreamingResponse`. `format` is forced to `pcm`. No batching for
streaming (per-session ordering and latency are prioritized).

### TTS WebSocket (`/ws/tts`)
Protocol modeled on vLLM-Omni for compatibility/passthrough:
`session.config` → `input.text`* → `input.done` → for each sentence:
`audio.start` (JSON) + PCM frames (binary) + `audio.done` → `session.done`.

### Conversation (`/ws/chat`)
1. `session.config` (TTS voice).
2. `input.text` (user transcription).
3. The server streams the LLM (`llm.delta`), buffers tokens and emits a sentence as soon as a
   sentence boundary is detected (`audio.start` + PCM + `audio.done`), then `llm.done`.

## Model management

- **Hot-swap** (`residency: single`): before loading a new family, the current model is
  evicted LRU (`del` + `gc.collect()` + `torch.cuda.empty_cache()`).
- **Multi-residency** (`residency: multi`): keeps `resident` loaded; others are evicted on
  demand.
- **Lock**: inference and hot-swap are serialized by an `RLock`; blocking calls run through
  `asyncio.to_thread` so the event loop is never blocked.
- **Clone prompt cache**: `create_voice_clone_prompt` is memoized by hash (reference).

## Modules

| File | Responsibility |
|---|---|
| `app/config.py` | `config.yaml` + env overrides, persistence |
| `app/schemas.py` | request/response models, OpenAI/vLLM aliases |
| `app/chunking.py` | sentence/chunk splitting, audio concatenation |
| `app/audio.py` | PCM16/WAV/base64 |
| `app/engine.py` | model pool, backend dispatch, batch inference |
| `app/chatterbox_backend.py` | client for the isolated Chatterbox worker process |
| `app/chatterbox_worker.py` | Chatterbox inference in its own venv (JSON protocol) |
| `app/languages.py` | language name ↔ ISO code mapping |
| `app/scheduler.py` | asynchronous micro-batching |
| `app/llm.py` | OpenAI-compatible LLM client (SSE) |
| `app/security.py` | Basic + API keys |
| `app/main.py` | FastAPI application, routes, WS, proxies |
| `ui/i18n.js` | multilingual UI strings |

## "True real-time" streaming

The `qwen-tts` package does **not** expose token-level streaming; the streaming offered here
is **per sentence**. For native codec-window PCM, use the `vllm` profile (vLLM-Omni) and wire
the engine abstraction to it. See `docs/backends.md#streaming`.

# API reference

All routes are served by the gateway (`:8004`). When `auth.enabled` is true, add either
`Authorization: Bearer <key>` / `X-API-Key: <key>` or HTTP Basic authentication.

## Service

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status (always public) |
| GET | `/` | Web UI |
| GET | `/docs` | OpenAPI (Swagger) |
| GET | `/api/config` | Public configuration (models, languages, LLM, UI) |
| PUT | `/api/settings` | Update `llm`, `generation`, `chunking`, `audio`, `stt`, `ui` |

## Models & voices

| Method | Path | Description |
|---|---|---|
| GET | `/api/models` | Available families, loaded, active |
| POST | `/api/models/load` | `{"family": "custom_voice"}` — load (hot-swap) |
| POST | `/api/models/select` | `{"variant": "custom_voice:0.6B"}` — pick the model version for a family |
| POST | `/api/models/unload` | `{"family": "..."}` or `{}` to unload everything |
| GET | `/api/voices?family=custom_voice` | Speakers, languages, reference files |
| POST | `/api/voices/load` | `{"avatar": "id"}` or `{"family": "custom_voice"}` — warm a conversation voice |
| GET | `/v1/audio/voices` | OpenAI-compatible list |

## TTS

### `POST /api/tts`

Body (`TTSRequest`, aliases accepted):

| Field | Type | Default | Description |
|---|---|---|---|
| `text` / `input` | str \| str[] | — | Text(s) to synthesize |
| `family` / `mode` / `task_type` | str | `models.default_family` | `custom_voice`, `voice_design`, `base` |
| `language` | str | `Auto` | e.g. `French`, `English`, `Auto` |
| `speaker` / `voice` | str | — | Speaker (`custom_voice` family) |
| `instruct` / `instructions` | str | — | Style/emotion (`custom_voice`, `voice_design`) |
| `ref_audio` | str | — | Path / URL / data-URL (`base` family) |
| `ref_text` | str | — | Reference transcript (ICL mode) |
| `x_vector_only_mode` | bool | false | Speaker embedding only |
| `stream` | bool | false | Progressive PCM streaming (single text only) |
| `format` / `response_format` | str | `audio.format` | `wav` or `pcm` (forced to `pcm` when `stream`) |
| `chunking` | bool | `chunking.enabled` | Split by sentence |
| `max_chars` | int | `chunking.max_chars` | Chunk size |
| `generation` | object | — | `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens`... |

Response: `audio/wav` or `audio/pcm`. On error: JSON `{"detail": "..."}`.

### `POST /v1/audio/speech`
Same body (OpenAI: `input`, `voice`, `response_format`, `speed`). `speed` is not applied.

### `WS /ws/tts`
Client events: `{"type":"session.config", ...}` then `{"type":"input.text","text":"..."}`
then `{"type":"input.done"}`.
Server events: `audio.start`, binary PCM16 frames, `audio.done`, `session.done`, `error`.

## STT (proxy to NeMo-Speech.cpp)

### `POST /v1/audio/transcriptions`
Multipart: `file` (WAV), `response_format` (`json`/`verbose_json`/`text`/`srt`/`vtt`),
`language`, `diarization` (bool, requires `verbose_json`), `automatic_punctuation`.

### `POST /api/stt/diarize`
Multipart `file` → segments `{start, end, speaker}`.

### `GET /api/stt/health`
Upstream STT service status.

### `WS /ws/stt`
Passthrough to the STT service's `/v1/audio/transcriptions/realtime`. Events:
`session.created`, `session.update`, PCM16 frames, `input_audio_buffer.commit`,
`conversation.item.input_audio_transcription.delta` / `.completed`.

## LLM / chat

### `GET /api/llm/ping`
Tests the connection to the external LLM.

### `POST /api/chat`
`{"text": "...", "history": [{"role","content"}]}` → SSE stream `data: {"delta": "..."}` ... `[DONE]`.

### `WS /ws/chat`
`{"type":"session.config","speak":true,"personality":"...","tts":{...}}` then
`{"type":"input.text","text":"..."}`. Send `{"type":"input.cancel"}` to stop the
current turn (LLM stream + TTS synthesis); `personality` is appended to the system
prompt for avatars that define one.
Server events: `transcript`, `llm.delta`, `audio.start` / PCM / `audio.done`, `llm.done`.

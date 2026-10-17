# API reference

All routes are served by the gateway (`:8004`). When `auth.enabled` is true, add either
`Authorization: Bearer <key>` / `X-API-Key: <key>` or HTTP Basic authentication.

## Service

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status (always public) |
| GET | `/ready` | Readiness (always public) |
| GET | `/api/capabilities` | Machine-readable capabilities (families, routes, formats) |
| GET | `/api/metrics` | Latency/throughput metrics (admin when auth is enabled) |
| GET | `/` | Web UI |
| GET | `/docs` | OpenAPI (Swagger) |
| GET | `/api/config` | Public configuration (models, languages, LLM, UI) |
| PUT | `/api/settings` | Update `llm`, `generation`, `chunking`, `audio`, `stt`, `ui`, `startup`, `models`, `vllm` (`enabled`/`base_url`/`path`), `server.log_level` — sections merge recursively |

## Access (admin)

Authentication is off by default. When `auth.enabled` is true, machine clients use an
**API key** (`Authorization: Bearer <key>`, `X-API-Key: <key>`, or `?api_key=<key>`) and the
admin UI uses **HTTP Basic**. Keys are stored **hashed** and shown only once at creation.

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/session` | Basic | Mint a short-lived admin key (`{key, id, expires_at, admin}`) |
| POST | `/api/session/logout` | session key | Revoke the presented session key |
| GET | `/api/api-keys` | admin | List client keys (never the secret: `id, name, prefix, created_at, expires_at, expired, last_used_at`) |
| POST | `/api/api-keys` | admin | `{"name", "expires_in_days"?}` → `{key, …}` (secret shown once) |
| DELETE | `/api/api-keys/{id}` | admin | Delete (revoke) a client key |

- "admin" = HTTP Basic admin credentials or an admin-scoped session key; a plain client key
  gets `403`.
- When `auth.enabled` is false, `GET /api/api-keys` returns `{"enabled": false, "keys": []}`.

## Models & voices

| Method | Path | Description |
|---|---|---|
| GET | `/api/vllm/health` | Probe the configured vLLM-Omni service (admin) |
| GET | `/api/models` | Available families, loaded, active |
| POST | `/api/models/load` | `{"family": "custom_voice"}` — load (hot-swap) |
| POST | `/api/models/select` | `{"variant": "custom_voice:0.6B"}` — point a family at one of its variants (`models.registry`) |
| POST | `/api/models/unload` | `{"family": "..."}` or `{}` to unload everything |
| GET | `/api/voices?family=custom_voice` | Speakers, languages, reference files |
| POST | `/api/voices/load` | `{"avatar": "id"}` or `{"family": "custom_voice"}` — warm a conversation voice |
| GET | `/v1/audio/voices` | OpenAI-compatible list (`voices` + `avatars` = voice-bank ids) |

## TTS

### `POST /api/tts`

Body (`TTSRequest`, aliases accepted):

| Field | Type | Default | Description |
|---|---|---|---|
| `text` / `input` | str \| str[] | — | Text(s) to synthesize |
| `family` / `mode` / `task_type` | str | `models.default_family` | `custom_voice`, `voice_design`, `base`, `chatterbox`, `chatterbox_turbo`, `chatterbox_nano` |
| `language` | str | `Auto` | e.g. `French`, `English`, `Auto` (Chatterbox maps to ISO codes; `Auto` uses `models.chatterbox.default_language`) |
| `speaker` / `voice` | str | — | Speaker (`custom_voice` family) |
| `instruct` / `instructions` | str | — | Style/emotion (`custom_voice`, `voice_design`) |
| `ref_audio` | str | — | Path / URL / data-URL (`base`, `chatterbox*`) |
| `ref_text` | str | — | Reference transcript (ICL mode; Qwen only) |
| `avatar` | str | — | Voice-bank id (see Avatars). `voice`/`speaker` also accepts a voice-bank id. |
| `clone_backend` / `engine` | str | `models.clone_backend` | Clone engine for this request: `qwen3` or `chatterbox` |
| `x_vector_only_mode` | bool | false | Speaker embedding only (Qwen only) |
| `stream` | bool | false | Progressive PCM streaming (single text only) |
| `format` / `response_format` | str | `audio.format` | `wav` or `pcm` (forced to `pcm` when `stream`) |
| `chunking` | bool | `chunking.enabled` | Split by sentence |
| `max_chars` | int | `chunking.max_chars` | Chunk size |
| `generation` | object | — | `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens` (Qwen); `exaggeration`, `cfg_weight`, `min_p` (Chatterbox) |

Response: `audio/wav` or `audio/pcm`. On error: JSON `{"detail": "..."}`.

### `POST /v1/audio/speech`
Same body (OpenAI: `input`, `voice`, `response_format`, `speed`). `speed` is not applied.

### `WS /ws/tts`
Client events: `{"type":"session.config", ...}` then `{"type":"input.text","text":"..."}`
then `{"type":"input.done"}`.
Server events: `audio.start`, binary PCM16 frames, `audio.done`, `session.done`, `error`.

## Avatars (voice bank)

The UI calls this the **Voice Bank**; the API uses `avatars`. All avatars are **clones**. The
engine that synthesizes them defaults to `models.clone_backend` (`qwen3` or `chatterbox`) and
can be overridden per request with `clone_backend`/`engine`; an explicit clone-capable `family`
(`base`, `chatterbox`, `chatterbox_turbo`, `chatterbox_nano`) also takes precedence.

| Method | Path | Description |
|---|---|---|
| GET | `/api/avatars` | List avatars |
| POST | `/api/avatars` | Multipart: `name`, `prompt` (Voice Design description), `language`, `ref_text`, `personality`, `x_vector_only`, `ref_audio` (file), `image` (file). With `ref_audio` → clone from it; otherwise a fingerprint is rendered from `prompt` and stored. |
| PUT | `/api/avatars/{id}` | Edit name/description/language/personality/transcript/reference audio/photo |
| POST | `/api/avatars/fingerprint` | Multipart `prompt`, `language` → `audio/wav` (~10 s Voice Design reference, for preview) |
| POST | `/api/avatars/{id}/regenerate` | Re-render the fingerprint from the stored description |
| DELETE | `/api/avatars/{id}` | Delete |
| GET | `/api/avatars/{id}/audio` \| `/photo` | Reference WAV / photo |

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

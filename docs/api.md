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
| GET | `/api/config` | Public configuration (models, languages, LLM, UI, `text`) |
| PUT | `/api/settings` | Update `llm`, `generation`, `chunking`, `text`, `audio`, `stt` (`enabled`/`base_url`/`language`/`languages`/…), `ui`, `startup`, `models`, `vllm` (`enabled`/`base_url`/`path`/`family`/`ref_root`/`model`/`max_model_len`/`enforce_eager`/`stage_memory`/`sleep_after`/`sleep_level`/`fallback_to_local`), `docker` (`control`), `server.log_level` — sections merge recursively. Selecting the vLLM-Omni engine starts the container (and unloads our copy of the family it serves); going back to the local engine stops it, returning its VRAM |

> **Admin-only writes.** When `auth.enabled` is true, `PUT /api/settings`, model/voice management
> (`POST /api/models/select|load|unload`, `POST /api/voices/load`), avatar writes
> (`POST`/`PUT`/`DELETE /api/avatars…`, `POST /api/avatars/hygiene/clean`) and
> `POST /api/stt/download` require an **admin** (HTTP Basic or an admin session key); a plain
> client key gets `403`. Read-only routes stay open to any valid key.

## Access (admin)

Authentication is off by default. When `auth.enabled` is true, machine clients use an
**API key** (`Authorization: Bearer <key>`, `X-API-Key: <key>`, or `?api_key=<key>`) and the
admin UI uses **HTTP Basic**. Keys are stored **hashed** and shown only once at creation.

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/session` | Basic | Mint a short-lived admin key (`{key, id, expires_at, admin}`) |
| POST | `/api/session/logout` | session key | Revoke the presented session key |
| GET | `/api/api-keys` | admin | List client keys (never the secret: `id, name, prefix, created_at, expires_at, expired, last_used_at`, `text`) |
| POST | `/api/api-keys` | admin | `{"name", "expires_in_days"?, "text"?}` → `{key, …}` (secret shown once) |
| PATCH | `/api/api-keys/{id}` | admin | `{"name"?, "text"?}` — rename and/or change the client's `text` policy **without** rotating the secret (`"text": null` clears the override) |
| DELETE | `/api/api-keys/{id}` | admin | Delete (revoke) a client key |

**Per-key text policy.** The optional `text` block (`{"cleanup": …, "emoji": …, "code_blocks": …}`)
overrides the global `text` setting **for that client only**: only the fields you send are
overridden, the others keep following the global setting (see
[Text cleanup](#text-cleanup)). It is how behaviour is attached to a credential instead of to
the request body, so `/api/tts` and `/v1/audio/speech` stay standard.

- "admin" = HTTP Basic admin credentials or an admin-scoped session key; a plain client key
  gets `403`.
- When `auth.enabled` is false, `GET /api/api-keys` returns `{"enabled": false, "keys": []}`.

## Models & voices

| Method | Path | Description |
|---|---|---|
| GET | `/api/vllm/health` | Probe the configured vLLM-Omni service: `ok`, `models`, `served_family` and a `warning` when the served checkpoint is not the one `vllm.family` routes (admin) |
| GET | `/api/vllm/budget` | Effective per-stage VRAM budget for a checkpoint on the gateway's card (`?model=…`, defaults to the routed family): `stages` (`{"0": 0.36, "1": 0.11}`), `source` (`computed`/`manual`), the weights, the headroom (CUDA contexts + KV) and the card (admin) |
| GET | `/api/vllm/service` | State of the vLLM-Omni container: `state` (running, image, served checkpoint) and `profile` — what it really runs with (`actual`), what the settings ask for (`desired`) and whether they `matches`; reports `control: false` until `docker.control` is enabled (admin) |
| POST | `/api/vllm/service/{action}` | `start` / `stop` / `restart` the vLLM-Omni container, then warm it up (admin, needs `docker.control`) |
| PUT | `/api/vllm/service/model` | `{"model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}` — recreate the container serving another checkpoint, keeping its configuration (started only if the vLLM-Omni engine is selected); `vllm.family` follows the checkpoint (admin, needs `docker.control`) |
| GET | `/api/models` | Available families, loaded, active |
| POST | `/api/models/load` | `{"family": "custom_voice"}` — load (hot-swap) |
| POST | `/api/models/select` | `{"variant": "custom_voice:0.6B"}` — point a family at one of its variants (`models.registry`) |
| POST | `/api/models/unload` | `{"family": "..."}` or `{}` to unload everything |
| GET | `/api/voices?family=custom_voice` | Speakers, languages, reference files |
| POST | `/api/voices/load` | `{"avatar": "id"}` or `{"family": "custom_voice"}` — warm a conversation voice |
| GET | `/v1/audio/voices` | OpenAI-compatible list (`voices` + `avatars` = voice-bank ids) |

> The `/api/vllm/*` routes pilot the **experimental** vLLM-Omni streaming service (**may be
> dropped in a future release**): it buys time-to-first-audio at the cost of a larger VRAM
> reservation, a ~1 min cold start, an extra container to manage and per-stage budgets to tune.
> The local engines are the supported path — see
> [backends](backends.md#why-vllm-omni-stays-optional).

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
| `ref_audio` | str | — | Path / URL / data-URL (`base`, `chatterbox*`). Chatterbox additionally needs ≥ 0.5 s, non-silent, finite samples (`400`; with `stream: true` the stream is cut instead) |
| `ref_text` | str | — | Reference transcript (ICL mode; Qwen only) |
| `avatar` | str | — | Voice-bank id (see Avatars). `voice`/`speaker` also accepts a voice-bank id. |
| `clone_backend` / `engine` | str | `models.clone_backend` | Clone engine for this request: `qwen3` or `chatterbox` |
| `x_vector_only_mode` | bool | false | Speaker embedding only (Qwen only) |
| `stream` | bool | false | Progressive PCM streaming (single text only) |
| `format` / `response_format` | str | `audio.format` | `wav` or `pcm` (forced to `pcm` when `stream`) |
| `chunking` | bool | `chunking.enabled` | Split by sentence and join the audio back. Applies to `stream: true` *and* to a non-streaming `text`. |
| `max_chars` | int | `chunking.max_chars` | Chunk size, **40–600** (`422` outside): above 600 one call would generate more audio than an engine can, and the tail would be lost silently |
| `generation` | object | — | Sampling knobs, see below |

Generation knobs are range-checked (out-of-range or `nan`/`inf` → `422`): `temperature`
∈ (0, 5], `top_p` ∈ (0, 1], `top_k` ∈ [0, 1000], `repetition_penalty` ∈ (0, 5],
`max_new_tokens` ∈ [1, 32768] and the subtalker variants (Qwen); `exaggeration` ∈ [0, 2],
`cfg_weight` ∈ [0, 1], `min_p` ∈ [0, 1] (Chatterbox). Values set in `config.yaml`
(`generation.*`) are clamped at use instead, since that file is not validated on load.

A `text` is split according to `chunking`/`max_chars` (all engines, streaming or not) and the
pieces are joined with `chunking.silence_ms`; the response still contains one file per input
`text`. Cuts land on a **sentence** end, else a clause end, else a line break, else between
words — a chunk never ends inside a word (only a *single* token longer than `max_chars`, e.g. a
very long URL, has to be cut). As a last resort — `chunking: false`, or a chunk above what the
engine can ingest — the engine itself splits further rather than truncating, e.g. on the
**2048**-token positional table of Chatterbox (multilingual).

Response: `audio/wav` or `audio/pcm` (mono signed-16-bit little-endian). For `pcm` the sample
rate is announced in the `X-Sample-Rate` header, streaming and non-streaming alike: it is the
rate of **the samples in that response**. `audio.sample_rate` (the setting) is what is *asked* of
the vLLM-Omni service and the default shown to clients — the local engines output **24 kHz** and
nothing is resampled here, so a client that assumes the setting instead of reading the header
plays the audio at the wrong speed. On error: JSON `{"detail": "..."}`.

With `stream: true` the first chunk is produced **before** the response headers are sent (the
header must describe the data, and a failure then becomes a real status code). A failure *after*
that first chunk still can only cut the stream — treat a short/abrupt end as an error.

### `POST /v1/audio/speech`
Same body (OpenAI: `input`, `voice`, `response_format`, `speed`). `speed` is not applied.

### Text cleanup

The gateway can clean the text before it is split and spoken — Markdown and emojis are the
usual noise of an LLM answer, and some characters cannot be pronounced. It is a **setting**
(`text` in `PUT /api/settings`, and optionally **per API key**, see
[Access](#access-admin)); there is deliberately no request field, so `/api/tts` keeps its
standard shape.

| Field | Values | Default | What it does |
|---|---|---|---|
| `text.cleanup` | `off`, `light`, `llm` | `off` | `light`: remove characters that carry no sound (control codes, zero-width joiners/BOM, soft hyphens), normalize spacing and fold typographic punctuation. `llm`: also strip Markdown (`**bold**`, `#`, `>` quotes, bullets, `[text](url)`, inline code) |
| `text.emoji` | `keep`, `strip`, `name` | `keep` | `strip`: remove emojis (as a space, so neighbouring words are not glued). `name`: speak the ~20 most frequent ones in the request language (`😂` → "rires"/"laughing") and strip the rest. Families, flags, skin tones and keycaps count as one emoji |
| `text.code_blocks` | `drop`, `announce` | `drop` | With `cleanup: llm`: drop a fenced ``` block, or replace it with "bloc de code"/"code block" |

Sentence punctuation is **never** touched (the prosody depends on it), numbers are not spelled
out, and the reference transcript (`ref_text`) is left alone. In the chat WebSocket an open code
fence is remembered across chunks, so a streamed block stays silent.

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
| GET | `/api/avatars/hygiene` | Audit of the voice bank: files per kind, total size, duplicate names, orphan files (a clip/photo whose entry is gone) and voices without a clip |
| POST | `/api/avatars/hygiene/clean` | Delete the orphan files (never a voice) and return the refreshed audit |
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

`language` is an optional hint; a **name** (`French`) or an ISO-639-1 code (`fr`) is accepted
(the gateway normalizes it to a code). It defaults to the `stt.language` setting, then to the
model's own default: leave both empty for **"auto"** (the multilingual models detect the
language themselves).

> **When it matters.** The hint only changes the decoding of *prompt-conditioned* models — the
> streaming Nemotron ones (`nemotron-3.5-asr-streaming-0.6b`), which pick a language prompt from
> it. Models such as `parakeet-tdt-0.6b-v3` accept the field but ignore it (language-agnostic),
> and the English-only models are fixed. It never *prevents* another language from being
> transcribed; it is a hint, not a filter.

### `POST /api/stt/diarize`
Multipart `file` → segments `{start, end, speaker}`.

### `GET /api/stt/health`
Upstream STT service status.

### `GET /api/stt/models`
Downloadable STT models from the vendored index, with cache status and the active one:
`{available, active, models: [{repo, aliases, filename, size, cached, active, streaming}]}`.

### `POST /api/stt/download`
`{"model": "<repo|alias|filename>"}` (defaults to the index's default ASR model) — downloads the
GGUF into the shared cache (SHA-256 verified), yields SSE progress events and points the STT
container at it. Admin-only.

### `WS /ws/stt`
Passthrough to NeMo-Speech.cpp's realtime endpoint. Requires a **streaming** model
(`GET /api/stt/models` → `streaming: true`) and serves **one session per client socket**.
Connect with `ws://<host>:8004/ws/stt?api_key=<key>` when auth is enabled.

```
-> {"type":"session.update","session":{"sample_rate":16000,"language":"fr","automatic_punctuation":true}}
-> <binary PCM16 mono 16 kHz frames>        # e.g. 3200-byte (100 ms) frames
-> {"type":"input_audio_buffer.commit"}     # end of utterance
<- {"type":"session.created", ...}
<- {"type":"conversation.item.input_audio_transcription.delta","delta":"..."}
<- {"type":"conversation.item.input_audio_transcription.completed","transcript":"..."}
<- {"type":"error","message":"..."}
```

`language` in `session.update` follows the same rules as the file endpoint above (code or name,
optional; only prompt-conditioned models use it). If the client's `session.update` omits it, the
gateway fills in the configured `stt.language` (and sends its own `session.update` for a client
that sends none) — so a configured default applies to every client, code or name alike.

Audio frames are **raw binary** (not base64) at 16 kHz mono. `language` is optional;
`automatic_punctuation` defaults to `true`. The gateway retries a slow upstream handshake
once; a client that leaves its socket open keeps the single upstream session busy
(see [realtime.md](realtime.md#stt-sessions)). Runnable client: `examples/python/stt_ws.py`.

## LLM / chat

### `GET /api/llm/ping`
Tests the connection to the external LLM.

### `POST /api/chat`
`{"text": "...", "history": [{"role","content"}]}` → SSE stream `data: {"delta": "..."}` ... `[DONE]`.

### `WS /ws/chat`
`{"type":"session.config","speak":true,"personality":"...","tts":{...}}` then
`{"type":"input.text","text":"...","turn":1}` (the optional `turn` id is echoed on the turn's
`transcript`, `llm.delta`, `llm.done`, `turn.cancelled`, `error` and `audio.*` events, so a
client can ignore the tail of a turn it already cancelled). Send `{"type":"input.cancel"}` to stop
the current turn (LLM stream + TTS synthesis); `personality` is appended to the system prompt for
avatars that define one.
Server events: `transcript`, `llm.delta`, `audio.start` / PCM / `audio.done`, `llm.done`.

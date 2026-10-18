# Integration guide

AiUdio is a **service**: the web UI is a demo/admin front (create voices, test, tune
settings), while the **HTTP/WebSocket API is the integration contract**. This guide is for
developers wiring another application (backend, desktop, mobile, voice agent) to an AiUdio
instance. Full route reference: [`api.md`](api.md) · live OpenAPI: `/docs`.

Machine-oriented summary for AI agents: [`../AGENTS.md`](../AGENTS.md).

## 1. Base URL and discovery

Everything is served by the gateway (default `:8004`):

| Thing | URL |
|---|---|
| API / OpenAI-compatible | `http://<host>:8004/` |
| Web UI (demo/admin) | `http://<host>:8004/` |
| OpenAPI (Swagger) | `http://<host>:8004/docs` · schema `/openapi.json` |
| Capabilities (JSON) | `http://<host>:8004/api/capabilities` |
| Health / readiness | `/health` · `/ready` (public, no auth) |

`GET /api/capabilities` is the fastest way for a client (or an agent) to learn what a given
instance offers: engine families and their backend, supported formats/sample rate, the
WebSocket routes, whether STT is enabled and the auth mode.

```bash
curl -s http://localhost:8004/api/capabilities | jq
```

## 2. Authentication

Authentication is **disabled by default** (trusted LAN). When `auth.enabled` is true:

- **Machine clients** use an **API key**: `Authorization: Bearer <key>`,
  `X-API-Key: <key>`, or `?api_key=<key>` (the only option for browser WebSockets).
- **The admin** logs in with **HTTP Basic**; the admin UI then mints a short-lived
  **session key** (`POST /api/session`) and uses it like any client.
- Keys are created/revoked in the UI (**Settings → Access & API keys**) — see
  [`security.md`](security.md). They are stored hashed and shown once.
- **Admin-only writes** (when `auth.enabled`): `PUT /api/settings`, model/voice management,
  avatar writes and `POST /api/stt/download` require HTTP Basic or an admin session key — a
  plain client key gets `403`. Read routes (TTS/STT/voices/capabilities) accept any valid key.

```python
HEADERS = {"X-API-Key": "aiud_..."}
```

> Ask the instance owner for a key, or create one from the admin. `/health`, `/ready` and
> (depending on `auth.exempt_paths`) nothing else stay public.

CORS: by default `server.cors_origins: ['*']`. For browser clients, restrict it in
production.

## 3. Text-to-speech

### 3.1 One-shot WAV — `POST /api/tts`

```bash
curl -s http://localhost:8004/api/tts -H 'Content-Type: application/json' \
  -d '{"text":"Hello world","family":"custom_voice","speaker":"Vivian","language":"English"}' \
  -o out.wav
```

```python
import httpx
r = httpx.post("http://localhost:8004/api/tts", headers=HEADERS, json={
    "text": "Hello world", "family": "custom_voice",
    "speaker": "Vivian", "language": "English",
})
r.raise_for_status()
open("out.wav", "wb").write(r.content)
```

Key fields (`TTSRequest`): `text` (or `input`, or a list), `family`, `language`,
`speaker`/`voice`, `instruct`, `ref_audio`/`ref_text`, `format` (`wav`|`pcm`),
`chunking`, `max_chars`, `generation`.

`chunking` (default: `chunking.enabled`) splits the text and joins the audio back, for streaming
**and** non-streaming requests alike — useful for long inputs, where a model can otherwise
truncate or refuse the text. Cuts land on sentence/clause punctuation, line breaks, then word
boundaries (never inside a word). `max_chars` is bounded to **40–600**: above that one call
would generate more audio than an engine can, and the tail would be lost silently.
`chunking: false` sends it in one piece.

The operator may also let the gateway **clean** the text it receives (Markdown, emojis,
invisible characters) — globally or per API key. It is not a request field: see
[Text cleanup](api.md#text-cleanup) for the levels and the defaults (both `off`).

### 3.2 OpenAI-compatible — `POST /v1/audio/speech`

Same engine, OpenAI request shape (`input`, `voice`, `response_format`), so existing OpenAI
SDKs work by pointing `base_url` at `http://<host>:8004/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8004/v1", api_key="unused-or-api-key")
with client.audio.speech.with_streaming_response.create(
    model="aiudio", voice="Vivian", input="Hello", response_format="wav"
) as r:
    r.stream_to_file("out.wav")
```

List voices: `GET /v1/audio/voices`.

### 3.3 Low-latency streaming (PCM over HTTP)

Set `stream: true` (single text only). `format` is forced to `pcm`; chunks are emitted
**sentence by sentence** as they are synthesized, so playback can start early. The sample
rate is in the `X-Sample-Rate` response header. A non-streaming `format: "pcm"` response
carries the same header (WAV already embeds its rate).

```bash
curl -N http://localhost:8004/api/tts -H 'Content-Type: application/json' \
  -d '{"text":"Sentence one. Sentence two.","stream":true,"format":"pcm"}' \
  --output - | ffplay -f s16le -ar 24000 -ch_layout mono -
```

PCM is mono, signed 16-bit little-endian. A client should keep a small **jitter buffer**
(see §7).

### 3.4 WebSocket — `/ws/tts`

Modeled on the vLLM-Omni protocol. Send JSON text frames, receive JSON + binary PCM frames:

```
-> {"type":"session.config", <TTSRequest fields>}
-> {"type":"input.text","text":"Hello"}
-> {"type":"input.done"}
<- {"type":"audio.start","sentence_index":0,"format":"pcm","sample_rate":24000}
<- <binary PCM16 frame>
<- {"type":"audio.done","sentence_index":0,"total_bytes":N}
<- ...
<- {"type":"session.done","total_sentences":N,"sample_rate":24000}
```

Use `ws://<host>:8004/ws/tts?api_key=<key>` when auth is enabled.

### 3.5 Voices, cloning and avatars

- **Preset speakers** (`custom_voice` family): `GET /api/voices?family=custom_voice` →
  `{speakers, languages, reference_files}`.
- **Zero-shot clone per request**: send a `data:` base64 WAV as `ref_audio` (with `family`
  `base` or a `chatterbox*` family). Remote URLs/local paths are disabled by default. With
  `ref_audio` and no explicit `family`, the clone engine is chosen by `clone_backend`.
- **Reusable voices** (recommended for apps): create once from the admin
  (`POST /api/avatars`, or a Voice Design **fingerprint** via `POST /api/avatars/fingerprint`),
  then synthesize by id:

  ```bash
  curl -s http://localhost:8004/api/tts -H 'Content-Type: application/json' \
    -d '{"text":"Hello","avatar":"narrator"}' -o out.wav
  ```

  `GET /api/avatars` lists them (ids), `GET /v1/audio/voices` returns both preset `voices`
  and voice-bank `avatars`, and `GET /api/avatars/{id}/audio` returns the reference clip.
  A voice-bank id also works through `voice`/`speaker`
  (`{"text":"Hello","voice":"narrator"}`), so OpenAI SDK callers can use it directly.
  The clone engine is `models.clone_backend` by default and can be overridden **per request**
  with `clone_backend` (or `engine`): `{"avatar":"narrator","clone_backend":"chatterbox"}`.

## 4. Speech-to-text

- **File** — `POST /v1/audio/transcriptions` (multipart `file`, OpenAI-compatible
  `response_format` in `json`/`verbose_json`/`text`/`srt`/`vtt`, optional `language`).
- **Diarization** — `POST /api/stt/diarize` → segments `{start, end, speaker}`.
- **Realtime** — `WS /ws/stt` (passthrough to NeMo-Speech.cpp): send a `session.update`
  (`sample_rate`, optional `language`, `automatic_punctuation`), stream **raw binary PCM16
  mono 16 kHz** frames, then `input_audio_buffer.commit`; receive
  `conversation.item.input_audio_transcription.delta` / `.completed` events. Requires a
  **streaming** STT model (see `streaming` in `GET /api/stt/models`) and serves **one session
  per client socket** — close it when done. Runnable client: `examples/python/stt_ws.py`
  (see [api.md](api.md#ws-wsstt) for the exact frames).

`language` is an optional hint on both routes: a name (`French`) or an ISO code (`fr`) — the
gateway normalizes it. It defaults to the `stt.language` setting, then to the model's own
default, so leaving everything empty is **"auto"**: the multilingual models detect the language
themselves. The hint only changes the decoding of *prompt-conditioned* models (the streaming
Nemotron ASR); `parakeet-tdt-0.6b-v3` accepts the field but ignores it (language-agnostic), and
the English-only models are fixed. It is a hint, not a filter.

Model management (download/select) is done from the admin; `GET /api/stt/models` lists
availability and cache status.

## 5. Conversation (STT → LLM → TTS)

The LLM is **not** served by AiUdio; point it at an external OpenAI-compatible endpoint
(`config.yaml llm.base_url`). Two server-side options:

- `POST /api/chat` — SSE, text only (`data: {"delta": "..."}`).
- `WS /ws/chat` — full cascade with audio. Client events:
  `session.config` (`speak`, `personality`, `tts`), `input.text`, `input.cancel`.
  Server events: `transcript`, `llm.delta`, `audio.start`/PCM/`audio.done`, `llm.done`,
  `turn.cancelled`, `error`. Optional `turn` id: if you send `{"type":"input.text","text":…,"turn":N}`,
  the server echoes `turn: N` on that turn's events, so a client can drop the tail of a turn it
  cancelled (barge-in).

```python
import asyncio, json, websockets
async def main():
    async with websockets.connect("ws://localhost:8004/ws/chat") as ws:
        await ws.send(json.dumps({"type": "session.config", "speak": True,
                                  "tts": {"avatar": "narrator"}}))
        await ws.send(json.dumps({"type": "input.text", "text": "Hello!"}))
        async for msg in ws:
            if isinstance(msg, bytes):
                play(msg)              # PCM16
            else:
                print(json.loads(msg))
asyncio.run(main())
```

If you prefer to own the orchestration, chain the pieces yourself:
`WS /ws/stt` (or your own ASR) → your LLM → `POST /api/tts`/`/ws/tts`.

## 6. Errors, health and limits

Errors are JSON `{"detail": "..."}` with a 4xx/5xx status:

| Status | Meaning |
|---|---|
| `400` | Bad request / validation (unknown variant, unreadable `ref_audio`, disabled feature) |
| `401` | Authentication required (missing/invalid Basic or API key) |
| `403` | Valid key but **not an admin** on a write route |
| `404` | Unknown voice/avatar id |
| `409` | vLLM-Omni container control disabled (`docker.control`) |
| `501` | STT disabled (`stt.enabled: false`) |
| `502` | Upstream failure (STT service or the Docker API) |
| `503` | Model not loadable / engine unavailable (or first-time download in progress) |

`/health` (liveness) and `/ready` (readiness: scheduler up) are always public.

### Limits and behaviour per route

| Route | Limit / behaviour |
|---|---|
| `POST /api/tts` (REST) | Micro-batched; inference is **serialized** by the engine. A `text` is split per `chunking`/`max_chars` and the pieces are joined back (one file per text). Throughput scales with `generation.batch_size`; there is no server-side queue cap (pressure shows up as latency and `GET /api/metrics` queue depth). |
| `POST /api/tts` `stream:true` | Per sentence, one text at most. A mid-stream failure **ends the stream without an error frame** (HTTP headers are already sent): treat a short/abrupt end as an error and retry the request. |
| vLLM-routed family (`vllm.enabled`) | The service is authoritative for the family `vllm.family` names: if it is unreachable or not ready the request fails — `503` (`/api/tts`, `/ws/tts`) or an `error` event (`/ws/chat`) — instead of silently using the local engine. Set `vllm.fallback_to_local: true` to allow that switch. |
| `/v1/audio/speech` | Same engine and limits as `/api/tts`. |
| Chatterbox families | No batch API: requests are processed **sequentially** (one at a time). |
| `POST /v1/audio/transcriptions`, `/api/stt/diarize` | The whole upload is read into memory by the gateway and forwarded (300 s upstream timeout). Accepted audio formats are those of the loaded NeMo-Speech model. |
| `WS /ws/stt` | **One realtime session per client socket**; a leaked open socket keeps the upstream busy. Requires a streaming model. |
| `WS /ws/chat` | One turn at a time per socket; `input.cancel` stops the LLM stream and synthesis. The LLM is external, so its latency/limits are yours. |
| First call to a family | Downloads several GB of weights and loads them: expect a slow first request. Pre-warm with `POST /api/voices/load` (`{"avatar":"id"}` or `{"family":"custom_voice"}`). |
| Writes (`/api/settings`, models, avatars, STT download) | Admin only when `auth.enabled` (see §2). |

Treat the service as a **shared, bounded GPU resource**: prefer streaming, pre-warm voices,
and keep an eye on `/api/metrics`.

## 7. Real-time best practices

- **Prefer streaming** (`stream=true` or `/ws/tts`) for interactive apps; the server emits
  audio per sentence so time-to-first-audio stays low.
- **Pre-warm** the voice/model at session start (`POST /api/voices/load`) and, when the GPU
  has room, keep several families resident (`TTS_RESIDENCY=multi`; the default `single`
  hot-swaps and reloads on every family switch).
- **Jitter buffer**: buffer ~150–300 ms (or 2 chunks) before starting playback to absorb
  variation between chunks; concatenate to an `AudioContext`/`AudioTrack` at the sample
  rate from `X-Sample-Rate`.
- **Sample rate**: output is mono 24 kHz PCM16 by default; resample client-side if needed.
- **Cancel** in-flight work with `input.cancel` (`/ws/chat`) when the user speaks/aborts.
- **Transport**: PCM16 is fine on a LAN; for WAN/high-latency links, compress (Opus) on the
  client or terminate closer to the user (the roadmap is described in
  [`realtime.md`](realtime.md)).

## 8. Copy-paste examples

Runnable clients live in [`../examples/`](../examples/): `python/`, `node/`, `curl/`
(one-shot, streaming PCM, WebSocket TTS, chat, STT).

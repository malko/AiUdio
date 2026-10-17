# AGENTS.md

Guidance for AI coding agents working **on** this repository or integrating AiUdio **into
another project**. Humans: see [`README.md`](README.md).

## What this project is

AiUdio Server is a self-hosted **TTS/STT gateway** (FastAPI) with two interchangeable TTS
engines (Qwen3-TTS, Chatterbox), a streaming STT proxy (NeMo-Speech.cpp) and a
conversation loop over an external LLM. The bundled web UI is a **demo/admin** front; the
**HTTP/WebSocket API is the product**. Docker Compose runs the whole stack.

| Item | Value |
|---|---|
| Gateway | FastAPI, default `http://localhost:8004` |
| Auth | HTTP Basic for the admin; API keys for machines (`X-API-Key` / `Bearer` / `?api_key=`) |
| Discovery | `GET /api/capabilities`, OpenAPI at `/openapi.json` / `/docs` |
| Key routes | TTS `/api/tts` (streaming PCM with `stream:true`), OpenAI `/v1/audio/speech`, TTS WS `/ws/tts`, chat WS `/ws/chat`, STT `/v1/audio/transcriptions` + WS `/ws/stt` |
| Language/stack | Python 3.12, FastAPI, pydantic; static vanilla-JS UI in `ui/` |

Terminology: the UI's **Voice Bank** holds reusable cloned voices; the API calls them
`avatars` (`/api/avatars`, request field `avatar`). A voice-bank id can also be passed as
`voice`/`speaker`. The clone engine defaults to `models.clone_backend` and can be overridden
per request with `clone_backend`/`engine`.

## Run it (for manual testing)

```bash
cp .env.example .env          # set PORT, HF_TOKEN, optional AUTH_*, LLM_*
docker compose up -d --build   # gateway :8004, STT :8080
# tests / lint (no GPU needed):
python3 -m pytest -q
python3 -m ruff check app tests
```

The first request to a model family downloads several GB of weights.

`config.yaml` is mounted read-only as **defaults**; runtime settings (UI/API) persist in
the writable `aiudio_state` volume (`CONFIG_PATH=/app/state/config.yaml`). To re-apply
defaults, remove that volume (`docker compose down && docker volume rm aiudio_aiudio_state`).

## Minimal integration contract

To integrate AiUdio into a host application, you only need:

1. **A base URL** (e.g. `http://aiudio:8004` from another container on the `aiudio-net`
   network, or `http://<host>:8004`).
2. **A credential**: an API key (header `X-API-Key: aiud_...`) when auth is enabled,
   otherwise nothing.
3. **Requests**:

```python
import httpx
BASE = "http://aiudio:8004"
H = {"X-API-Key": "aiud_..."}  # omit if auth disabled

# One-shot WAV
r = httpx.post(f"{BASE}/api/tts", headers=H, json={"text": "Hello", "speaker": "Vivian"})
open("out.wav", "wb").write(r.content)

# Streaming PCM (mono s16le 24 kHz; sample rate in X-Sample-Rate)
with httpx.stream("POST", f"{BASE}/api/tts", headers=H,
                  json={"text": "Hi. How are you?", "stream": True}) as resp:
    for chunk in resp.iter_bytes():
        speaker.write(chunk)
```

Full recipes (TTS/STT/chat, WS protocols, latency tips): [`docs/integration.md`](docs/integration.md).
Runnable clients: [`examples/`](examples/).

## Integration checklist (for an agent asked to "add AiUdio to project X")

1. Read `GET /api/capabilities` (or `docs/integration.md`) to learn the available families,
   formats and routes for the target instance.
2. Add configuration to project X: `AIUDIO_BASE_URL` and `AIUDIO_API_KEY` (env/secret).
3. Add a thin client module in project X's own idioms (HTTP client + optional WebSocket),
   covering: synthesize-to-bytes, streaming PCM, and (if needed) chat/STT.
4. Wire it behind project X's feature flag/config; never hard-code keys.
5. Buffer ~150–300 ms of PCM before playback; reuse one connection for streaming.
6. Verify against `../examples/` and the live `/docs`.

Runnable probe:

```bash
curl -sf "$AIUDIO_BASE_URL/health" && curl -sf "$AIUDIO_BASE_URL/api/capabilities" | jq .
```

## Documentation must stay in sync (required)

**Any change to the API, configuration, environment variables, endpoints, protocols or
default behaviour MUST update the docs in the same change.** Treat this as part of
"done". In particular:

- `README.md` — user-facing overview, quick start, env vars, troubleshooting.
- `docs/api.md` — routes/fields; `docs/integration.md` — how clients consume the service.
- `docs/architecture.md`, `docs/backends.md`, `docs/conversation.md`, `docs/realtime.md`,
  `docs/security.md` — when the matching behaviour changes.
- `AGENTS.md` (this file) and `llms.txt` — when the integration contract or entry points change.
- `.env.example` / `config.yaml` — when a setting is added or renamed.

Prefer code that is self-describing (docstrings, OpenAPI tags) so `/docs` stays accurate.

## Repo conventions

- Source, comments and docs are in **English**; user-facing UI strings live in `ui/i18n.js`
  (English + French).
- Commits follow **[gitmoji](https://gitmoji.dev)**: `<gitmoji> <type>(<scope>): <subject>`.
  A `commit-msg` hook enforces the prefix (`make hooks`).
- Tests must pass without a GPU (`tests/conftest.py` provides a mocked engine):
  `python3 -m pytest -q` and `python3 -m ruff check app tests`.
- Keep changes focused and document any new endpoint in `docs/api.md` and
  `docs/integration.md`.

# AiUdio Server

Self-hosted **text-to-speech (TTS)** server built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
(CustomVoice / VoiceDesign / cloning), complemented by a **STT** stage (NeMo-Speech.cpp:
Parakeet TDT v3 + streaming Nemotron 3.5 + diarization) and a **conversation** mode
pluggable into an external OpenAI-compatible LLM. Everything is served by a single
`docker compose` stack, reachable on the LAN or on the Docker network.

> Status: functional skeleton (API, UI, engines, compose, tests). Hardware validation
> (GPU, model weights) must be done on the target machine.

> [!NOTE]
> - **Tested on Linux only** by the maintainer. Windows/macOS are not validated (only the
>   Linux Docker workflows are exercised).
> - **Vibe-coded** with [opencode](https://opencode.ai) and **DeepSeek V4.1 Flash**. Review
>   the code and pin versions before any production use.

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Hardware variants](#hardware-variants)
- [Models](#models)
- [API](#api)
- [STT](#stt)
- [Conversation with an external LLM](#conversation-with-an-external-llm)
- [Internationalization](#internationalization)
- [Security](#security)
- [Development](#development)
- [Commit conventions (gitmoji)](#commit-conventions-gitmoji)
- [Troubleshooting](#troubleshooting)
- [Licenses and credits](#licenses-and-credits)

## Features

- **Qwen3-TTS engine**: 3 model families (CustomVoice, VoiceDesign, Base for cloning),
  hot-swap or multi-residency, micro-batching for throughput, sentence chunking for long
  text, progressive PCM streaming.
- **API**: first-party REST (`/api/tts`), **OpenAI-compatible** (`/v1/audio/speech`,
  `/v1/audio/voices`), **WebSocket** (`/ws/tts`, vLLM-Omni-style protocol).
- **STT**: file transcription (OpenAI-compatible `/v1/audio/transcriptions`), realtime
  streaming (WebSocket), diarization.
- **Web UI**: Synthesis / Transcription / Conversation / Settings tabs, same-origin
  (no CORS), browser microphone, gapless PCM playback.
- **Conversation**: plug in an external LLM (llama.cpp, Ollama, vLLM, OpenAI...) and run
  the STT → LLM → TTS cascade in streaming mode, without serving the LLM.
- **Multilingual UI**: English and French out of the box, easily extensible.
- **Security**: HTTP Basic for the UI, API keys (Bearer / `X-API-Key`) for machines.
- **Multi-backend**: CUDA (default), ROCm, CPU, Vulkan (STT) via Compose overrides.

## Architecture

```
                 ┌──────────────────────────────────────────────┐
   Browser  ────▶│ aiudio  (gateway, :8004)                     │
   / API client  │  • Qwen3-TTS engine (hot-swap + batching)    │
                 │  • static UI, /api/*, /v1/*, /ws/*           │
                 │  • STT proxy + external LLM proxy            │
                 └───────┬──────────────────────────┬───────────┘
                         │ docker network aiudio-net
              ┌──────────▼─────────┐      ┌─────────▼──────────────┐
              │ stt (:8080)        │      │ external LLM            │
              │ NeMo-Speech.cpp    │      │ (outside compose:       │
              │ Parakeet/Nemotron  │      │  llama.cpp, Ollama...)  │
              └────────────────────┘      └─────────────────────────┘
   ("vllm" profile) tts-vllm : official vLLM-Omni image — experimental PCM streaming
```

Details: [`docs/architecture.md`](docs/architecture.md) · API: [`docs/api.md`](docs/api.md) ·
hardware backends: [`docs/backends.md`](docs/backends.md) ·
conversation: [`docs/conversation.md`](docs/conversation.md) ·
security: [`docs/security.md`](docs/security.md).

## Requirements

- Docker Engine + the `docker compose` plugin.
- **NVIDIA**: recent driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (for the default CUDA profile).
- Disk space: ~10 GB+ per 1.7B TTS model (Hugging Face cache) plus STT weights.
- `make` (optional, command shortcuts).

## Quick start

```bash
git clone <your-fork> aiudio && cd aiudio
cp .env.example .env        # set PORT, HF_TOKEN, AUTH_*, LLM_*
make hooks                  # enable the gitmoji commit-msg hook + template

# (optional) pre-download the TTS models (plain curl, no Python needed)
./scripts/download_models.sh

docker compose up -d --build
```

- UI: `http://<host-ip>:8004/`
- API: `http://<host-ip>:8004/api/tts` · OpenAPI docs: `/docs`
- Health: `http://<host-ip>:8004/health`

Useful checks:

```bash
docker compose ps
docker compose logs -f aiudio
curl -s localhost:8004/health | jq
```

> The first call to a model family triggers the weight download (several GB). The cache
> is persisted in the `hf_cache` volume.

## Hardware variants

| Target | Command |
|---|---|
| **CUDA** (default, 12.4) | `docker compose up -d --build` |
| **CUDA 13** (driver ≥ 580) | `docker compose -f docker-compose.yml -f docker-compose.cuda13.yml up -d --build` |
| **AMD ROCm** | `docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --build` |
| **CPU** | `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build` |
| **Vulkan** (STT) | `docker compose -f docker-compose.yml -f docker-compose.vulkan.yml up -d --build` |
| **Mixed** (e.g. CUDA TTS + CPU STT) | `docker compose -f docker-compose.yml -f docker-compose.mixed.yml up -d --build` |
| **vLLM-Omni** (experimental) | `docker compose --profile vllm up -d` |

Notes: **TTS has no Vulkan backend** (PyTorch/vLLM); under Vulkan the TTS falls back to the
CPU image (or point it to a remote service). On **ROCm**, FlashAttention 2 is not available
→ `sdpa` attention. On **CUDA**: a container does **not** need to match the host CUDA toolkit
— only the NVIDIA **driver** matters, so a CUDA 12.4 container runs fine on a CUDA 13.x host.
Use the CUDA 13 override if you want CUDA 13 images (newer GPUs; requires driver ≥ 580).
See [`docs/backends.md`](docs/backends.md).

### Mixing backends (TTS and STT are independent)

Yes — the TTS gateway and the STT service are **separate containers**, so each can use a
different backend. For example, run **TTS on CUDA** (fast) and **STT on CPU** (no extra VRAM)
on the same host:

```bash
# Option A: the ready-made mixed override (CUDA TTS + CPU STT)
docker compose -f docker-compose.yml -f docker-compose.mixed.yml up -d --build
```

```bash
# Option B: pick backends independently via .env (no override file)
#   TTS backend  -> TTS_DOCKERFILE (docker/tts.Dockerfile, docker/tts.cpu.Dockerfile,
#                   docker/tts.rocm.Dockerfile) + TTS_BASE_IMAGE / TTS_TORCH_* args
#   STT backend  -> STT_BACKEND (cuda | cpu | vulkan)
TTS_DOCKERFILE=docker/tts.Dockerfile
STT_BACKEND=cpu
docker compose up -d --build
```

Valid combinations: any of {CUDA, ROCm, CPU} × {CUDA, CPU, Vulkan}. Constraints to remember:
the STT `cuda` and `vulkan` backends need the matching device (`deploy.resources` GPU
reservation, or `/dev/dri` for Vulkan) — the mixed override clears the GPU reservation for a
CPU STT. Full details: [`docs/backends.md`](docs/backends.md#mixing-backends).

## Models

| Family | Model | Use |
|---|---|---|
| `custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 predefined voices + style instruction |
| `voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | voice described in natural language |
| `base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | voice cloning (3 s reference) |

Override the repositories via `config.yaml` (`models.registry`) or the environment
variables `TTS_CUSTOM_VOICE`, `TTS_VOICE_DESIGN`, `TTS_BASE`. 0.6B variants are supported.

By default the weights are downloaded automatically on first use (into the `hf_cache`
volume). To pre-fetch them instead, run `./scripts/download_models.sh` (plain **curl**, no
Python dependency; honours `HF_TOKEN` and `HF_ENDPOINT`). It saves each model under
`hf_cache/models/<name>`; to run fully offline, point `models.registry` at those directories,
e.g.:

```yaml
models:
  registry:
    custom_voice: /app/hf_cache/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
    voice_design: /app/hf_cache/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    base: /app/hf_cache/models/Qwen3-TTS-12Hz-1.7B-Base
```

Loading strategy in `config.yaml`:

- `models.residency: single` (default) → a single model in VRAM, automatic hot-swap.
- `models.residency: multi` + `models.resident: [custom_voice, base]` → keep several models
  loaded (more VRAM, no reload).

## API

Quick examples (full reference: [`docs/api.md`](docs/api.md)):

```bash
# First-party TTS (WAV)
curl -s localhost:8004/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world","family":"custom_voice","speaker":"Vivian","language":"English"}' \
  -o out.wav

# OpenAI-compatible
curl -s localhost:8004/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello there","voice":"Ryan","response_format":"wav"}' \
  -o hello.wav

# Streaming PCM (playback: ffplay -f s16le -ar 24000 -ch_layout mono -)
curl -N localhost:8004/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Sentence one. Sentence two.","stream":true,"format":"pcm"}' \
  | ffplay -f s16le -ar 24000 -ch_layout mono -
```

In Python (OpenAI SDK):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8004/v1", api_key="none")
with client.audio.speech.with_streaming_response.create(
    model="aiudio", voice="Vivian", input="Hello", response_format="wav"
) as r:
    r.stream_to_file("out.wav")
```

## STT

The `stt` service relies on **[NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)**
(official ggml-based C++ runtime). NVIDIA does not publish a registry image, so it is built
from their sources (`docker/stt.Dockerfile`).

- **Parakeet TDT 0.6B v3**: 25 European languages, automatic punctuation (offline, accuracy).
- **Nemotron 3.5 ASR Streaming 0.6B**: realtime streaming (~160 ms) — default.
- **Sortformer**: diarization (who speaks).

Endpoints (proxied through the gateway):

```bash
# File
curl -s localhost:8004/v1/audio/transcriptions -F file=@audio.wav -F response_format=json

# Diarization
curl -s localhost:8004/api/stt/diarize -F file=@meeting.wav
```

Config (`config.yaml` / `.env`): `STT_ASR_MODEL` (default `nemotron-3.5`, or
`parakeet-tdt-0.6b-v3`), `STT_MODELS_DIR` (GGUF cache mounted at `/models`).

## Conversation with an external LLM

**The LLM is not served by this compose stack.** Point it at an OpenAI-compatible endpoint
in `.env` or the Settings tab:

```bash
# Example llama.cpp (on the host):
llama-server -hf ggml-org/gemma-3-4b-it-GGUF --port 8080
# then
LLM_BASE_URL=http://host.docker.internal:8080/v1
LLM_MODEL=gemma-3-4b-it
```

The UI (**Conversation** tab) and the `WS /ws/chat` endpoint then chain STT → LLM → TTS in
streaming mode. Details: [`docs/conversation.md`](docs/conversation.md).

## Internationalization

The repository, code and documentation are in **English**. The **product UI is
multilingual**: a language selector in the header switches between English and French, and
additional locales can be added by extending `MESSAGES` in [`ui/i18n.js`](ui/i18n.js).

## Security

By default authentication is **disabled** (trusted LAN use). To enable it:

```bash
AUTH_ENABLED=true
AUTH_USERNAME=admin
AUTH_PASSWORD='strong-password'
AUTH_API_KEYS='machine-key-1,machine-key-2'
```

- UI: **HTTP Basic** (the browser prompts for credentials).
- Machines: `Authorization: Bearer <key>` or the `X-API-Key: <key>` header.
- `/health` stays public.

⚠️ **A browser cannot send an `Authorization` header over WebSocket**: use an **API key**
(`?api_key=...`) for protected WS usage. For Internet exposure, put a TLS reverse proxy in
front (see [`docs/security.md`](docs/security.md)).

Reference audio (`ref_audio`) is **restricted by default** — only base64 data-URLs are
accepted; remote URLs and local paths are rejected to prevent SSRF and local file disclosure
(`audio.allow_remote_media`, `audio.allow_local_ref_paths`). A **build/run supply-chain risk
assessment** and a dependency audit are in [`docs/security.md`](docs/security.md); run
`make audit` (pip-audit) locally.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                      # tests without GPU (mocked engine)
uvicorn app.main:app --reload  # dev server (the real TTS engine needs qwen-tts + GPU)
make lint
```

## Commit conventions (gitmoji)

Commit messages follow **[gitmoji](https://gitmoji.dev)**:
`<gitmoji> <type>(<scope>): <subject>` — e.g. `✨ feat(tts): add dynamic batching`.
A `commit-msg` hook (`.githooks/`) checks the prefix. Enable it with `make hooks`.

## Troubleshooting

| Symptom | Hint |
|---|---|
| `CDI device injection failed` | NVIDIA runtime/plugin; use `runtime: nvidia` or update the toolkit |
| `CUDA out of memory` | `models.residency: single`, rely on hot-swap, or use 0.6B |
| Very long first call | weights are downloading (check logs / `hf_cache`) |
| Authenticated WS fails | use an API key (Basic does not work over browser WebSocket) |
| STT unreachable | `docker compose logs stt`; the upstream build may need a path adjustment (`cmake --install`) |
| Choppy TTS streaming | lower `chunking.max_chars` or use the `vllm` profile |

## Licenses and credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Apache-2.0.
- [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) — Apache-2.0 (third-party components: see their `THIRD_PARTY_NOTICES`).
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — Apache-2.0.
- This project: see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

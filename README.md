# AiUdio Server

Self-hosted **text-to-speech (TTS)** server with **two selectable engines**:
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (CustomVoice / VoiceDesign / cloning) and
[Chatterbox](https://github.com/resemble-ai/chatterbox) (zero-shot multilingual cloning) —
use one or the other, per family and as the global engine for cloned voices. Complemented
by a **STT** stage (NeMo-Speech.cpp: Parakeet TDT v3 + streaming Nemotron 3.5 + diarization)
and a **conversation** mode pluggable into an external OpenAI-compatible LLM. Everything is
served by a single `docker compose` stack, reachable on the LAN or on the Docker network.

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
- [Hardware](#hardware)
- [Models](#models)
- [API](#api)
- [Integration](docs/integration.md)
- [Real-time](docs/realtime.md)
- [STT](#stt)
- [Conversation with an external LLM](#conversation-with-an-external-llm)
- [Internationalization](#internationalization)
- [Security](#security)
- [Development](#development)
- [Commit conventions (gitmoji)](#commit-conventions-gitmoji)
- [Troubleshooting](#troubleshooting)
- [Licenses and credits](#licenses-and-credits)

## Features

- **Two TTS engines**: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (CustomVoice,
  VoiceDesign, Base for cloning) and [Chatterbox](https://github.com/resemble-ai/chatterbox)
  (zero-shot multilingual cloning + Turbo/Nano English). Hot-swap or multi-residency,
  micro-batching (Qwen), sentence chunking for long text, progressive PCM streaming.
  Choose the engine per family, and globally for cloned voices.
- **API**: first-party REST (`/api/tts`), **OpenAI-compatible** (`/v1/audio/speech`,
  `/v1/audio/voices`), **WebSocket** (`/ws/tts`, vLLM-Omni-style protocol).
- **STT**: file transcription (OpenAI-compatible `/v1/audio/transcriptions`), realtime
  streaming (WebSocket), diarization.
- **Web UI**: Conversation / Synthesis / Voice Bank / Transcription / Settings tabs,
  same-origin (no CORS), browser microphone, gapless PCM playback.
- **Conversation**: plug in an external LLM (llama.cpp, Ollama, vLLM, OpenAI...) and run
  the STT → LLM → TTS cascade in streaming mode, without serving the LLM. Push-to-talk or
  hands-free **VAD** (auto-send on pause), barge-in interruption, and per-reply replay.
- **Multilingual UI**: English and French out of the box, easily extensible.
- **Security**: HTTP Basic for the UI, API keys (Bearer / `X-API-Key`) for machines.
- **Single CUDA stack**: TTS gateway on the official PyTorch image + STT on the official NVIDIA NGC NeMo-Speech.cpp image (base images overridable via `.env`).

## Architecture

```
                 ┌──────────────────────────────────────────────┐
   Browser  ────▶│ aiudio  (gateway, :8004)                     │
   / API client  │  • TTS engines: Qwen3-TTS + Chatterbox       │
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
integration: [`docs/integration.md`](docs/integration.md) ·
hardware backends: [`docs/backends.md`](docs/backends.md) ·
conversation: [`docs/conversation.md`](docs/conversation.md) ·
real-time: [`docs/realtime.md`](docs/realtime.md) ·
security: [`docs/security.md`](docs/security.md).

> Integrating AiUdio into another application? Start with
> [`docs/integration.md`](docs/integration.md) (the API is the product; the UI is a
> demo/admin front). AI coding agents: see [`AGENTS.md`](AGENTS.md).

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

## Hardware

One **CUDA** compose (the default `docker-compose.yml`, CUDA 13), works with any NVIDIA
**driver ≥ 580**:

- **TTS gateway**: `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` (official image).
- **STT**: `nvcr.io/nvidia/nemo-speech.cpp:0.2.0` (official NVIDIA NGC image, CUDA, multi-arch), wrapped with busybox for the wait entrypoint.
- Optional: `docker compose --profile vllm up -d` for the experimental vLLM-Omni streaming service.

Non-NVIDIA hardware uses **additive overrides** (the CUDA default is unchanged):

| Target | Command |
|---|---|
| **AMD ROCm** (TTS rocm/pytorch; STT built from source, CPU) | `docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --build` |
| **CPU** | `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build` |
| **Vulkan** (STT on ggml/Vulkan; TTS on CPU) | `docker compose -f docker-compose.yml -f docker-compose.vulkan.yml up -d --build` |

Reminder: a container does **not** need to match the host CUDA toolkit — only the driver
matters. Base images and torch pins can be overridden via `.env` (`TTS_BASE_IMAGE`,
`TTS_TORCH_SPEC`, `TTS_TORCH_INDEX_URL`, `STT_IMAGE`). See [`docs/backends.md`](docs/backends.md).

## Models

| Family | Model | Use |
|---|---|---|
| `custom_voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 predefined voices + style instruction |
| `voice_design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | voice described in natural language |
| `base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | voice cloning (3 s reference) |
| `chatterbox` | `ResembleAI/chatterbox` (Multilingual V3) | zero-shot cloning, 20+ languages |
| `chatterbox_turbo` | `ResembleAI/chatterbox-turbo` | low-latency English (350M) |
| `chatterbox_nano` | `ResembleAI/chatterbox-nano` | smallest English model (110M, CPU-able) |

Override the repositories via `config.yaml` (`models.registry`) or the environment
variables `TTS_CUSTOM_VOICE`, `TTS_VOICE_DESIGN`, `TTS_BASE`. 0.6B variants are supported.
Select the clone engine with `models.clone_backend` (`TTS_CLONE_BACKEND`) and point the
isolated Chatterbox venv with `models.chatterbox.python` (`TTS_CHATTERBOX_PYTHON`).

Pick a per-family version in the Synthesis tab (**Model → Version**), or set the global Qwen
size with `models.default_size` (`0.6B` or `1.7B`). Families without a matching variant keep
their repo.

### Qwen3-TTS or Chatterbox?

Each family is bound to a backend: the `chatterbox*` families are served by Chatterbox,
everything else by Qwen3-TTS. You can therefore run Chatterbox **instead of** Qwen3-TTS:

- **Plain TTS** — request a Chatterbox family. It has a built-in voice, so no reference clip
  is needed:

  ```bash
  curl -s localhost:8004/api/tts -H 'Content-Type: application/json' \
    -d '{"text":"Hello world","family":"chatterbox","language":"English"}' -o out.wav
  ```

- **Cloned voices / avatars** — set the global clone engine to `chatterbox`
  (`models.clone_backend` in Settings, or the per-conversation engine selector) so cloned voices are synthesized by Chatterbox.

Voice Design (description-based voices) and the Qwen preset speakers always use Qwen3-TTS.

### Chatterbox backend

Chatterbox (Resemble AI) is offered as an **alternative** to Qwen3-TTS. It cannot share the same
Python environment: `chatterbox-tts` pins different `torch`/`transformers` versions than
`qwen-tts` (`transformers==5.2.0` vs `==4.57.3`). AiUdio therefore installs Chatterbox in an
**isolated virtualenv** (`/opt/cb/venv`) and runs it as a child process
(`app/chatterbox_worker.py`), driven over a JSON protocol. See `docs/backends.md`.

- Built by default; disable with `INSTALL_CHATTERBOX=0` (build arg / `.env`).
- The interpreter is auto-detected (`/opt/cb/venv/bin/python`); override with
  `models.chatterbox.python` or `TTS_CHATTERBOX_PYTHON`.
- Checkpoint version via `models.chatterbox.t3_model` (`v3` default, or `v2`). **V3 is only
  published on GitHub**; the PyPI release (`chatterbox-tts==0.1.7`) is v2-only, so the image
  installs from git by default (`CHATTERBOX_SPEC`). With a PyPI install the worker falls back
  to v2 automatically.
- The venv is created with `--system-site-packages` and reuses the base image's **CUDA 13
  torch** (chatterbox pins `torch==2.6.0` / CUDA 12.4, which has **no kernels for Blackwell /
  RTX 50xx (sm_120)**); the `torch`/`torchaudio` and `nvidia-*` (cu12) wheels chatterbox pulls
  are removed so the system's CUDA 13 NCCL/libs take over.
- PerTh watermarking is **off by default** for self-hosted instances
  (`models.chatterbox.watermark`); enable it to mark generated audio.
- Chatterbox has no batch API: concurrent requests are processed **sequentially** (the
  micro-batcher only benefits Qwen families). "Auto" language is not supported — a default is
  used (`models.chatterbox.default_language`).

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

- `models.residency: multi` (default, `resident: [custom_voice, base]`) → keeps the chat
  family and the clone family loaded, so switching between preset and cloned voices does not
  reload (and re-warm up) the model.
- `models.residency: single` → a single model in VRAM, automatic hot-swap (less VRAM, but
  every family switch pays a full load).

The **web UI (Synthesis tab)** exposes a model + version selector with plain-language
descriptions and a **Load** button (models load lazily on first use anyway). The **Transcription
tab** lists the downloadable STT models with descriptions, sizes and cache status.

**Voice Bank tab**: all voices in the bank are **clones**. You can either upload a reference
clip, or generate a ~10 s **fingerprint** from a Voice Design description and save it as a clone
(`POST /api/avatars/fingerprint`). Each voice can have a photo/name, be regenerated from its
description, and is selectable in the **Conversation** tab. The **clone engine**
(`models.clone_backend`: `qwen3` or `chatterbox`, editable in Settings or per conversation and
overridable per API request with `clone_backend`) chooses which backend synthesizes cloned
voices; Voice Design always uses Qwen3-TTS. Voices are stored as `<name>.wav` + `<name>.json` (+
`<name>.png`) under `voices/avatars/` (persistent). API: `GET/POST/PUT/DELETE /api/avatars`,
`POST /api/avatars/fingerprint`, `POST /api/avatars/{id}/regenerate`,
`GET /api/avatars/{id}/audio|photo`. A voice-bank id can be used as `avatar` or as
`voice`/`speaker` in `/api/tts` and `/v1/audio/speech`.

**Conversation is the home page** and lets you pick the voice (presets or an avatar) and the
STT model. In **Settings → Startup** you can mark an STT model and a voice to preload when the
server starts (`startup.stt_model`, `startup.voice`).

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

The `stt` service uses the **official NVIDIA NGC image**
[`nvcr.io/nvidia/nemo-speech.cpp:0.2.0`](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo-speech.cpp)
(CUDA, multi-arch, signed). The image is distroless (no shell), so AiUdio wraps it in a
**tiny derived image that only adds busybox** (`docker/stt-nvcr.Dockerfile`) for the
wait-for-model entrypoint — **no compilation**. Models are provided by the gateway.

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

Model storage is **shared with the TTS cache** (the `hf_cache` named volume): the gateway
downloads GGUFs under `nemo-speech/models` and points the STT container at the chosen one.
Set `STT_MODELS_DIR` to a host path for a bind mount instead.

**Models are downloaded on demand**, not at container startup: the gateway fetches the GGUF
from Hugging Face (SHA-256 verified) when you click **Download** in the **STT tab**, writes a
pointer file, and the STT container starts automatically. Pin the image via `STT_IMAGE`.

> The official image is **CUDA-only**. For **CPU/Vulkan/ROCm**, use the corresponding
> compose overrides, which build NeMo-Speech.cpp from source (`docker/stt.Dockerfile`).

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
```

- UI / admin: **HTTP Basic** (`username` / `password`).
- Machines: **API keys** created from the admin (**Settings → Access & API keys**), stored
  **hashed**, shown once, revocable, with an optional expiration. Send them as
  `Authorization: Bearer <key>` or `X-API-Key: <key>`.
- The admin UI mints a short-lived **session key** on load (`POST /api/session`), so it
  needs no manual key — and it works for WebSocket too.
- `/health` stays public.

⚠️ **A browser cannot send an `Authorization` header over WebSocket**: the UI uses the
session key as `?api_key=...`; non-browser clients do the same. For Internet exposure, put a
TLS reverse proxy in front (see [`docs/security.md`](docs/security.md)).

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
| `Warning: flash-attn is not installed` | expected — attention falls back to PyTorch `sdpa` (no FlashAttention build in the image) |
| `Chatterbox is not installed` | build the image with `INSTALL_CHATTERBOX=1`, or point `models.chatterbox.python` at a venv with `chatterbox-tts` |
| Chatterbox slow with many callers | it has no batch API; requests are serialized (use Qwen for concurrent throughput) |

## Licenses and credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Apache-2.0.
- [Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI) — MIT (PerTh watermarking optional).
- [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) — Apache-2.0 (third-party components: see their `THIRD_PARTY_NOTICES`).
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — Apache-2.0.
- This project: see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

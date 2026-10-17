# Hardware backends

AiUdio ships a **single CUDA compose** (`docker-compose.yml`, CUDA 13), plus **additive
overrides** for non-NVIDIA hardware. The CUDA stack runs on any NVIDIA **driver ≥ 580**. A
container does **not** need to match the host CUDA toolkit version — only the driver matters.

## Default (CUDA)

| Component | Image | Notes |
|---|---|---|
| TTS gateway | `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` (built with our app) | official base image |
| STT | `nvcr.io/nvidia/nemo-speech.cpp:0.2.0` + busybox wrapper | official NGC image, CUDA, multi-arch |
| vLLM-Omni (profile `vllm`) | `vllm/vllm-omni:latest` | optional experimental streaming |

### Why CUDA 13.0 (and not 13.2)

`torchaudio` is only published for **CUDA 13.0** on PyPI. A CUDA 13.2 `torch` paired with a
CUDA 13.0 `torchaudio` fails at load with *"PyTorch and TorchAudio were compiled with
different CUDA versions"*. Keeping torch and torchaudio on the same CUDA minor (13.0) avoids
this; it still runs on 13.x host drivers. Since PyTorch 2.11, `pip install torch` pulls CUDA
13.0 wheels by default and torchaudio 2.11 is ABI-stable.

CUDA 13 supports **Turing (SM 7.5) and newer** only.

## Non-NVIDIA overrides

| Target | Override | TTS | STT |
|---|---|---|---|
| AMD ROCm (Linux) | `docker-compose.rocm.yml` | `rocm/pytorch` (sdpa, no FlashAttention) | built from source, CPU backend |
| CPU | `docker-compose.cpu.yml` | `python:3.12-slim` + CPU torch | built from source, CPU backend |
| Vulkan (Intel/AMD iGPU) | `docker-compose.vulkan.yml` | CPU image (no Vulkan TTS) | built from source, Vulkan backend |

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml   up -d --build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml    up -d --build
docker compose -f docker-compose.yml -f docker-compose.vulkan.yml up -d --build
```

The non-NVIDIA overrides build NeMo-Speech.cpp from source (`docker/stt.Dockerfile`,
`BACKEND=cpu|vulkan`), since the official image is CUDA-only. ROCm STT uses the CPU backend
(HIP/ggml is not provided). The upstream ref is set with `NEMO_SPEECH_REF` (default `main`).

## Chatterbox backend (isolated venv)

Chatterbox (Resemble AI) is a second TTS engine, selectable per family
(`chatterbox`, `chatterbox_turbo`, `chatterbox_nano`). It **cannot** be installed in the
gateway's Python environment: `chatterbox-tts` pins `transformers==5.2.0` and `torch==2.6.0`,
while `qwen-tts` pins `transformers==4.57.3` on the CUDA 13 / torch 2.14 base image. Two
`transformers` majors cannot coexist in one environment.

AiUdio therefore installs Chatterbox in a **dedicated virtualenv** inside the *same image*
(`/opt/cb/venv`, build arg `INSTALL_CHATTERBOX=1`) and runs `app/chatterbox_worker.py` as a
**child process**, driven over a line-delimited JSON protocol by `app/chatterbox_backend.py`.
No separate container is needed; only the Python runtime is isolated.

The package is installed from **GitHub** by default (`CHATTERBOX_SPEC`): the PyPI release
(`chatterbox-tts==0.1.7`) is multilingual **v2 only** and lacks the `t3_model` argument needed
for V3. Pin a tag/commit in `CHATTERBOX_SPEC` for reproducibility. The worker introspects the
installed package's signatures, so it also works against the PyPI (v2) release.

Because chatterbox pins `torch==2.6.0` (CUDA 12.4) — which ships **no kernels for Blackwell /
RTX 50xx (sm_120)** — the venv is created with `--system-site-packages` so it reuses the base
image's **CUDA 13 torch**. Chatterbox's own dependencies (e.g. `transformers==5.2.0`) are
installed into the venv and shadow the system ones; the `torch`/`torchaudio` chatterbox pulls
**and its `nvidia-*` CUDA wheels** (cu12, which would otherwise shadow the system's cu13 NCCL)
are then uninstalled so the system's GPU-enabled build takes over.

- Interpreter: auto-detected at `/opt/cb/venv/bin/python`; override with
  `models.chatterbox.python` / `TTS_CHATTERBOX_PYTHON`.
- The worker shares the `hf_cache` volume for model downloads.
- **PerTh watermarking** is off by default (`models.chatterbox.watermark`); the worker swaps in
  a no-op watermarker when disabled.
- **No batch API**: `generate()` is single-text, so the micro-batcher (`app/scheduler.py`) only
  helps Qwen families; Chatterbox requests are processed sequentially (lower throughput under
  concurrency, but identical single-stream latency profile).

## Model downloads

- **TTS**: downloaded by the gateway via `huggingface_hub` on first use
  (`HF_HUB_ENABLE_HF_TRANSFER=1` requires the `hf_transfer` package, installed in the image).
- **STT**: the official image ships **no downloader**; the gateway downloads the GGUF from
  Hugging Face into the shared volume, writes a pointer (`.active`), and the STT container
  (official image or source build) starts on it. See the STT section of the README.

## Custom images

Everything is overridable via `.env` (no extra compose files):

```bash
TTS_BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
TTS_TORCH_SPEC=                      # empty = keep the image's torch/torchaudio
TTS_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
STT_IMAGE=nvcr.io/nvidia/nemo-speech.cpp:0.2.0   # pin a tag or digest
```

## Streaming

- **Gateway (native)**: **per-sentence** streaming (PCM16). Reduced perceived latency but no
  token-level streaming (not exposed by `qwen-tts`).
- **`vllm` profile** (official vLLM-Omni image): set `vllm.enabled: true` and the gateway
  routes streaming CustomVoice requests to it (`POST {vllm.base_url}/v1/audio/speech` with
  `stream=true`, `response_format=pcm`) for **native codec-window PCM**; the local
  per-sentence engine is used as a fallback if the service is unreachable. Experimental,
  NVIDIA. See `app/vllm_backend.py`.
- **STT**: NeMo-Speech.cpp exposes a realtime WebSocket
  (`/v1/audio/transcriptions/realtime`, ~160 ms with Nemotron 3.5). Parakeet TDT v3 stays
  offline (no cache-aware streaming).

## Sizing (indicative)

| Component | VRAM |
|---|---|
| Qwen3-TTS 1.7B bf16 | ~3.5–4 GB |
| Qwen3-TTS 0.6B bf16 | ~1.5 GB |
| Chatterbox Multilingual V3 (500M, fp32) | ~2–3 GB |
| STT Nemotron 3.5 Q8 (ggml) | < 1 GB |
| vLLM-Omni (1.7B + caches) | ~8–12 GB |

With `residency: single`, only one TTS model is loaded at a time → target 8 GB+ for
TTS + STT. For multi-residency, add up the models.

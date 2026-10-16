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
- **`vllm` profile** (official vLLM-Omni image): `stream=true` + `response_format=pcm` → PCM
  chunks decoded on the fly; WebSocket `/v1/audio/speech/stream`. Experimental, NVIDIA.
- **STT**: NeMo-Speech.cpp exposes a realtime WebSocket
  (`/v1/audio/transcriptions/realtime`, ~160 ms with Nemotron 3.5). Parakeet TDT v3 stays
  offline (no cache-aware streaming).

## Sizing (indicative)

| Component | VRAM |
|---|---|
| Qwen3-TTS 1.7B bf16 | ~3.5–4 GB |
| Qwen3-TTS 0.6B bf16 | ~1.5 GB |
| STT Nemotron 3.5 Q8 (ggml) | < 1 GB |
| vLLM-Omni (1.7B + caches) | ~8–12 GB |

With `residency: single`, only one TTS model is loaded at a time → target 8 GB+ for
TTS + STT. For multi-residency, add up the models.

# Hardware backends

## Matrix

| Target | TTS Qwen3-TTS | STT NeMo-Speech.cpp | vLLM-Omni |
|---|---|---|---|
| **NVIDIA CUDA** | ✅ official (bf16, FlashAttention 2 optional) | ✅ | ✅ (official image) || **AMD ROCm** | ⚠️ ROCm torch + `sdpa` (no FlashAttention), not upstream-tested | ⚠️ HIP/ggml possible; compose uses CPU | ❌ not supported (NVIDIA first) |
| **CPU** | ✅ slow (1.7B autoregressive) | ✅ fast (~6× real time) | ❌ |
| **Vulkan** | ❌ no PyTorch/vLLM/vLLM-Omni backend | ✅ ggml (Intel/AMD iGPU) | ❌ |
| **Apple Metal** | via `qwen-tts` outside Docker | ✅ (Metal build) | ❌ |

## Docker images

- **Official images used as-is**: `pytorch/pytorch` (CUDA), `rocm/pytorch` (ROCm),
  `python:slim` (CPU), `vllm/vllm-omni` (`vllm` profile).
- **Built** (no official image available): the TTS gateway (`docker/tts*.Dockerfile`, on an
  official base) and STT (`docker/stt.Dockerfile`, from official NeMo-Speech.cpp sources,
  since NVIDIA does not publish a registry image).

## Mixing backends

The TTS gateway and the STT service are **independent containers**, so their backends can be
mixed freely. Selection is driven by environment variables (see `.env.example`):

| Service | Variable | Values |
|---|---|---|
| TTS | `TTS_DOCKERFILE` | `docker/tts.Dockerfile` (CUDA), `docker/tts.cpu.Dockerfile`, `docker/tts.rocm.Dockerfile` |
| TTS | `TTS_BASE_IMAGE`, `TTS_TORCH_VERSION`, `TTS_TORCH_INDEX_URL` | base image / torch pinning |
| STT | `STT_BACKEND` | `cuda` (default), `cpu`, `vulkan` |
| STT | `NEMO_SPEECH_REF` | upstream NeMo-Speech.cpp branch/tag |

A ready-made example (CUDA TTS + CPU STT) is provided:

```bash
docker compose -f docker-compose.yml -f docker-compose.mixed.yml up -d --build
```

Or pick backends directly via `.env` with the base compose file:

```bash
TTS_DOCKERFILE=docker/tts.Dockerfile
STT_BACKEND=cpu
docker compose up -d --build
```

Any combination of {CUDA, ROCm, CPU} TTS × {CUDA, CPU, Vulkan} STT is valid. Caveats:

- The base file reserves the GPU for **both** services. When one side is CPU-only, clear its
  reservation (the `mixed`/`cpu`/`rocm`/`vulkan` overrides already do this) so it does not
  request an unused GPU.
- A `vulkan` STT needs `/dev/dri` and the `render` group; a `cuda` STT needs the NVIDIA
  runtime.
- Independently of the compute backend, the two services always communicate over the
  `aiudio-net` Docker network.

## Backend notes

### CUDA
- Default base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`.
- FlashAttention 2: not installed by default (long build, variable compatibility). To enable
  it: set `models.attn_implementation: flash_attention_2` **and** add the `flash-attn` build
  to the image.
- NVIDIA driver + Container Toolkit required.

#### CUDA 12 vs CUDA 13
- **A container does not need to match the host CUDA toolkit** — only the NVIDIA **driver**
  matters (backward compatibility). A CUDA 12.4 container runs on a CUDA 13.x host driver.
- **TTS** (default): `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, driver ≥ 525.
- **STT** (default): compiled from source on **Ubuntu 26.04 LTS / CUDA 13.4**
  (`nvidia/cuda:13.4.2-*-ubuntu26.04`). Ubuntu 26.04 ships **CMake 4.2.3**, required by
  NeMo-Speech.cpp's `CMakePresets.json` (Ubuntu 22.04's CMake 3.22 fails with
  `Unrecognized "version" field`). This needs **driver ≥ 580**.
- **Legacy (driver < 580)**: use `docker-compose.cuda12.yml` (STT on Ubuntu 24.04 / CUDA 12.8;
  Ubuntu 24.04 ships CMake 3.28, new enough for the presets).
- **CUDA 13 TTS** (`docker-compose.cuda13.yml`, driver ≥ 580): TTS base
  `pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime`; the image already ships a compatible
  `torch`/`torchaudio`, so torch is **not** force-reinstalled (`TORCH_SPEC` empty).
- Since PyTorch 2.11, `pip install torch` pulls CUDA 13.0 wheels by default, and **torchaudio
  2.11 is ABI-stable** (works with torch ≥ 2.11, no separate CUDA wheels). CUDA 13 supports
  Turing (SM 7.5) and newer only — older GPUs must stay on CUDA 12.
- Select images/versions via `TTS_BASE_IMAGE`, `TTS_TORCH_SPEC`, `TTS_TORCH_INDEX_URL`,
  `STT_CUDA_BUILDER_IMAGE`, `STT_CUDA_RUNTIME_IMAGE`, `STT_CUDA_ARCHITECTURES`
  (see `.env.example`).
- **STT CUDA architectures**: the builder has no GPU, so ggml's `-arch=native` detection
  would be unreliable. A broad portable default is compiled (`75;80;86;89;90;100;120`); set
  `STT_CUDA_ARCHITECTURES` to your GPU (e.g. `89` Ada/RTX 40, `90` Hopper, `120` Blackwell/RTX 50)
  for a faster, smaller build.

### ROCm
- Base: `rocm/pytorch` (tag configurable via `BASE_IMAGE`).
- `HSA_OVERRIDE_GFX_VERSION` is set in the Dockerfile; adjust it to your iGPU/dGPU.
- Force `models.attn_implementation: sdpa` (default).
- Devices `/dev/kfd`, `/dev/dri` + groups `video`,`render` (see override).

### CPU
- Base: `python:3.12-slim`, CPU torch.
- TTS works but is slow; reserve it for tests or short texts.
- STT remains comfortable (0.6B Q8-quantized models).

### Vulkan
- STT only (ggml). TTS has no Vulkan backend: the override falls back to the CPU image.
- Requires `/dev/dri` and the `render` group.

## Streaming

- **Gateway (native)**: **per-sentence** streaming (PCM16). Reduced perceived latency but no
  token-level streaming (not exposed by `qwen-tts`).
- **`vllm` profile** (official vLLM-Omni image): `stream=true` + `response_format=pcm` →
  PCM chunks decoded on the fly; WebSocket `/v1/audio/speech/stream`. Experimental, NVIDIA.
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

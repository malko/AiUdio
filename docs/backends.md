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
- **`vllm` profile** (official vLLM-Omni image): set `vllm.enabled: true` for **native
  codec-window PCM** (`POST {vllm.base_url}/v1/audio/speech` with `stream=true`,
  `response_format=pcm`, `stream_format=audio`) on the family named by `vllm.family`
  (`custom_voice` for the preset speakers, `base` for voice cloning — the service serves one
  checkpoint, so both need their own deployment); the local per-sentence engine handles the
  other families and is never a silent substitute: a request routed to a selected-but-down
  service fails (503 / `error` event) unless `vllm.fallback_to_local` is enabled. Experimental,
  NVIDIA. See `app/vllm_backend.py`.
- **STT**: NeMo-Speech.cpp exposes a realtime WebSocket
  (`/v1/audio/transcriptions/realtime`, ~160 ms with Nemotron 3.5). Parakeet TDT v3 stays
  offline (no cache-aware streaming).

### Why vLLM-Omni stays optional

Measured on this project's own chat workload, one shared GPU, warm service:

| | local engine (0.6B) | vLLM-Omni (1.7B, `--enforce-eager`) |
|---|---|---|
| first audio | 2–4.5 s (per sentence) | **0.04–0.06 s** (codec-window streaming) |
| RTF | ≈0.5 | **≈0.22** (15 s of audio decoded in 3.2 s) |
| cost per call | `0.6 s + 0.42 × audio_seconds` | ≈`0.2 × audio_seconds` |

The second column is a real gain, and it is **architectural**: the talker generates codec frames
and the `Code2Wav` decoder consumes them in overlapping windows (the 12 Hz decoder is causal),
emitting a first tiny window (~80 ms) immediately — while our local path cannot emit a single
byte before a whole sentence has been synthesized. It is not something we could tune: `qwen-tts`
exposes **no streaming generation** (`non_streaming_mode` "only *simulates* streaming text input
… rather than enabling true streaming input or streaming generation"), so matching it means
re-implementing a serving engine inside the gateway — the AR loop, the windowed codec decode and
the seam handling.

What it brings beyond latency, and this deployment does not use: continuous batching across
clients, per-stage placement/tensor parallelism, prefix caching, CUDA graphs/FlashInfer, a batch
endpoint, uploaded voices and an OpenAI-compatible server. What it costs: a second container
image (several GB on its own, on top of ours);
**one checkpoint per service** (presets *or* clones, `voice_design` separately, each switch = a
container restart); a VRAM reservation that must be sized per stage (see the table above) where
the local engine simply allocates what it needs; and, to pilot it from the UI, a Docker socket —
root-equivalent power over the host, see [security](security.md).

**Decision**: vLLM-Omni stays an **opt-in profile** (`docker compose --profile vllm up -d
tts-vllm`, then `vllm.enabled`), never a default and never required. The local engines are the
reference path — Qwen3-TTS for the preset speakers, cloning and `voice_design`, Chatterbox when
clone quality matters most. The gateway keeps the coexistence cheap (the engine switch
starts/stops the service, drops our own copy of the family it serves, enforces the choice at
boot), so trying it costs one setting and turning it off frees the VRAM again.

**When to revisit**: if `qwen-tts` (or an equivalent) ever exposes incremental generation, the
latency argument disappears and the local engine wins outright; conversely, on a large GPU or a
multi-tenant deployment the serving features become the point, and vLLM-Omni should then be the
default for the families it supports.

## Sizing (indicative)

| Component | VRAM |
|---|---|
| Qwen3-TTS 1.7B bf16 | ~3.5–4 GB |
| Qwen3-TTS 0.6B bf16 | ~1.5 GB |
| Qwen3-TTS `-Base` (cloning) | same as the matching CustomVoice variant |
| Chatterbox Multilingual V3 (500M, fp32) | ~2–3 GB |
| STT Nemotron 3.5 Q8 (ggml) | < 1 GB |
| vLLM-Omni (1.7B + caches) | ~8–12 GB |

With `residency: single` (the default), only one TTS model is loaded at a time → target 8 GB+
for TTS + STT. For multi-residency (`TTS_RESIDENCY=multi`), add up the models.

### Why vLLM-Omni needs so much more than the local engine

Same weights, different contract. Our engine is a *library*: it loads the weights and
allocates everything else on demand. vLLM-Omni is a *serving engine*: it must know at startup
how much it can serve, and it optimizes the decode loop.

| | Local engine (`qwen-tts`) | vLLM-Omni |
|---|---|---|
| Weights (1.7B CustomVoice + codec) | ~4.1 GiB | ~4.1 GiB (identical, the log prints it) |
| KV cache | grown per request (a few MB for a sentence) | **pre-allocated paged pool**, sized by `max-model-len` (0.44 GiB at 4096) |
| CUDA graphs / `torch.compile` | none (eager) | capture of `cudagraph_capture_sizes` (1…128) + inductor compile (~1–2 GiB, 13 s in the logs) |
| Speaker embeddings | small Python dict | `speaker_cache` reserves 512 MiB |
| Attention / sampling kernels | `sdpa` inside torch | FlashAttention-2 + FlashInfer workspaces |
| Stage reservations | n/a (one model) | **per stage**, from the model's deploy YAML — Qwen3-TTS sets 0.3 (talker) *and* 0.3 (`Code2Wav`), i.e. 60 % of the card whatever the variant; the weights only move the weights↔KV split *inside* a stage's budget |
| Processes / CUDA contexts | one | API server + engine core(s) + stage replicas: one context each (~0.3–0.6 GiB) + NCCL buffers |
| Activation buffers | sized per request | sized for `max_num_batched_tokens` (512, and 65536 for the `Code2Wav` stage) |

That is the trade for throughput and predictable latency — and it is why a shared card that
cannot afford the reservation fails with `Available KV cache memory: 0.35 GiB` while the local
engine is perfectly happy. The levers follow from the table: `--max-model-len` (smaller pool),
`--enforce-eager` (no graphs/compile), the 0.6B variant (smaller weights), and per-stage
`gpu_memory_utilization` low enough to leave room for the gateway and STT — **per stage**
(`--stage-overrides '{"0": {...}, "1": {...}}'`, the keys are stage ids, not GPUs): the bundled
Qwen3-TTS YAML reserves 0.3 for the talker *and* 0.3 for the vocoder, and a global
`--gpu-memory-utilization` replaces both with the same value rather than lowering them. See
the vLLM-Omni recipe in [realtime](realtime.md).

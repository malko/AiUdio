# Real-time pipeline

How latency is produced today, what was already tuned, and the roadmap to make the
STT → LLM → TTS loop feel instantaneous. Metrics are exposed at `GET /api/metrics`
(admin when auth is enabled) and readiness at `GET /ready`.

## Pipeline and where time goes

```
🎤 mic ─▶ WS /ws/stt (streaming ASR) ─▶ endpointing ─▶ WS /ws/chat
                                                          │
                                              ┌───────────┴───────────┐
                                              ▼                       ▼
                                    LLM token stream         _pop_chat_chunks()
                                    (external)               sentence/clause split
                                              └───────────┬───────────┘
                                                          ▼
                                          per-chunk TTS ─▶ PCM16 ─▶ client jitter buffer ─▶ 🔊
```

Stage | Latency source | Current handling
---|---|---
STT | streaming decode | Nemotron 3.5 realtime (~160 ms); endpointing is **client-side** (RMS + ~900 ms silence)
LLM | time-to-first-token | external; streamed token by token
Chunking | wait for a boundary | whole sentences only (~40 chars min, then grouped up to ~200)
TTS | model load + per-chunk synth | **per-sentence**, not token-level (`qwen-tts` exposes no codec streaming)
Transport | PCM16 over HTTP/WS | uncompressed mono 24 kHz
Playback | scheduling | client `PCMPlayer` with a small prebuffer (~200 ms / 250 ms timeout)

Measurable targets: **time-to-first-audio (TTFA)**, real-time factor (RTF), scheduler queue
depth, and per-stage timings (`/api/metrics`).

## Implemented (Phase 1 + 2/3 quick wins)

- **Streaming bypasses micro-batching**: `_stream_pcm` calls
  `BatchScheduler.submit_streaming()`, which runs immediately instead of waiting for the
  batch window (`generation.batch_window_ms`). Batching still applies to non-streaming REST.
- **Sentence-aligned chunking**: `_pop_chat_chunks` releases **complete sentences** —
  fragments would be synthesized as standalone utterances with their own intonation, heard as
  chopped, word-by-word speech. The **first chunk** of a turn is the exception: it may be
  cut at the first comma past ~30 chars, because the engine synthesizes audio ~4-6× slower
  than the LLM writes text (a word-end cut in the absence of a comma was tried and sounds
  chopped, so the first complete sentence is used instead).
  Then each following sentence is released as soon as it is complete while the engine is idle.
  Grouping (up to ~200 chars per synthesis call) only happens while the engine still has work
  queued: sending text promptly is what prevents an audible gap between chunks (the client
  plays what it has received; a late release drains its buffer — the `margin` logged per chunk
  becomes negative). A sentence that never ends is cut at the last clause/space boundary past
  ~240 chars (last resort). Chat units are synthesized as-is: `_stream_pcm` does not re-split
  them (that pass, `chunking.*`, applies to the TTS API/WebSocket routes).
- **Preload + optional warm-up**: `startup.tts_family` preloads a family at boot;
  `startup.warmup: true` runs a tiny synthesis to absorb CUDA init. `residency: multi`
  keeps several families hot (more VRAM; `TTS_RESIDENCY=multi`, `single` by default).
- **Adaptive client endpointing**: the VAD shortens the silence window when the streaming
  transcript already ends with sentence punctuation (faster turn-taking).
- **Browser echo cancellation**: the mic is opened with `echoCancellation`,
  `noiseSuppression`, `autoGainControl`, so hands-free barge-in works without headphones.
- **vLLM-Omni streaming backend** (experimental): with `vllm.enabled`, the family named
  by `vllm.family` is served by vLLM-Omni (native codec-window PCM) via `app/vllm_backend.py`,
  the other families keep using the local engine. Two families are supported:
  `custom_voice` (preset speakers) and `base` (voice cloning from a reference clip, sent
  inline as a `data:` URL), because **the service loads a single checkpoint**. Start it with
  a Base checkpoint and select `family: base` to route your cloned voices to it. Requests
  of the other family, and every request when the service is down, fall back to the local
  per-sentence engine. Enable it without editing the state file:

  ```bash
  docker compose --profile vllm up -d tts-vllm     # the service
  curl -s -X PUT localhost:8004/api/settings -H 'Content-Type: application/json' \
       -d '{"vllm":{"enabled":true,"base_url":"http://tts-vllm:8091","family":"custom_voice"}}'
  ```

  For **cloned voices**, start the service on a Base checkpoint and point `family` at it —
  the voice bank keeps working, only the reference clip travels differently (inline
  `ref_audio` data URL + `ref_text`, `task_type=Base`):

  ```yaml
  # docker-compose.yml, tts-vllm
    command: >
      vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-Base      # or 1.7B-Base
      --omni --host 0.0.0.0 --port 8091
      --max-model-len 2048 --enforce-eager
  ```

  ```bash
  curl -s -X PUT localhost:8004/api/settings -H 'Content-Type: application/json' \
       -d '{"vllm":{"enabled":true,"family":"base"}}'
  ```

  The preset speakers (`vivian`, `serena`…) then stay local, and so do Chatterbox clones
  (`models.clone_backend: chatterbox` is not what a Base checkpoint serves). An avatar
  without a transcript clones from the speaker embedding only (`x_vector_only_mode`).

  A manager that cannot pass a profile (Web UI, Dockge, Portainer…) should **delete the
  `profiles: ["vllm"]` line** of the `tts-vllm` service instead: it then belongs to the
  default stack and can be started/stopped from the UI. Keep `shm_size: "8gb"` (vLLM needs
  a large `/dev/shm`; Docker's default is 64 MB) and remember that the service reserves its
  own VRAM (see [backends](backends.md)), so keep it stopped when unused.

  vLLM-Omni exposes both the OpenAI-compatible `/v1/audio/speech` and a dedicated
  `/v1/audio/speech/stream` (WebSocket). Our client asks for **raw PCM** with
  `stream_format: "audio"`: with `stream: true` alone the service answers the OpenAI
  `speech.audio.*` SSE events (base64 deltas), which cannot be forwarded as PCM to our
  clients. **Settings → vLLM-Omni** has the switch, the family served, the base URL, the
  route and a *Test connection* button (`GET /api/vllm/health`, which reports the checkpoint
  the service actually serves and warns when it is not the family you route to it):

  ```bash
  curl -s -X PUT localhost:8004/api/settings -H 'Content-Type: application/json' \
       -d '{"vllm":{"enabled":true,"base_url":"http://tts-vllm:8091","path":"/v1/audio/speech"}}'
  curl -s localhost:8004/api/vllm/health | jq        # ok / models / error
  ```

  A `speech.audio.error` event or a `Stage-1 has no live replica` message means the
  service is up but one of its two stages died: restart the container and check that both
  stages report *EngineCore running*. The **first call after a restart is much slower**
  (tens of seconds: kernel/graph warm-up and the speech caches); the gateway therefore
  primes it with a tiny request when vLLM is enabled or at startup. That probe retries for
  ~2 minutes (the gateway usually boots before the vLLM service listens) and only logs a
  warning if it never answers; a turn that finds vLLM unavailable **falls back to the local
  engine** and counts `tts_vllm_fallbacks` (`tts_vllm_sessions` = served, `tts_vllm_warmups`
  = primed).
  Measured once warm (1.7B CustomVoice, `--enforce-eager`, `--max-model-len 2048`):
  TTFB ≈ 0.05 s and RTF ≈ 0.22, i.e. the audio starts almost immediately and is produced
  ~4× faster than it plays.

  The model the service serves is its own `vllm serve <model>` argument — the gateway's
  `models.registry`/`default_size` only drive the *local* engine; both 0.6B and 1.7B
  CustomVoice expose the same speakers, and the `-Base` checkpoints do the cloning.
  A gateway that routes `base` to a service serving a `CustomVoice` checkpoint (or the
  reverse) gets a 400 per request and falls back locally; `/api/vllm/health` names the
  mismatch instead.

  Without Compose, the same service can be started standalone on the stack network:

  ```bash
  docker run -d --name tts-vllm --gpus all --shm-size=8gb --network aiudio-net \
    -v "$PWD/hf_cache:/root/.cache/huggingface" -p 8005:8091 \
    vllm/vllm-omni:latest \
    vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --host 0.0.0.0 --port 8091
  ```

  Check the service on its own before switching the gateway over:

  ```bash
  curl -s localhost:8005/v1/models | head -c 200
  curl -s localhost:8005/v1/audio/speech -H 'Content-Type: application/json' \
       -d '{"input":"Bonjour, ceci est un test.","voice":"Vivian","response_format":"pcm","stream":true}' \
       -o /tmp/test.pcm -w '%{http_code} %{size_download}\n'
  ffplay -f s16le -ar 24000 -ac 1 /tmp/test.pcm    # should be intelligible
  ```

  If the service fails with `ValueError: To serve at least one request with the model's max
  seq len …, (X GiB KV cache is needed, which is larger than the available KV cache memory
  (Y GiB)`, the weights fit but the KV budget does not: lower the context and, if the card is
  tight, use the smallest Base/CustomVoice model and skip the CUDA graphs. TTSS inputs are
  short sentences, so a small context is enough:

  ```yaml
    command: >
      vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
      --omni --host 0.0.0.0 --port 8091
      --max-model-len 2048        # ~0.22 GiB of KV instead of 0.44 GiB at 4096
      --enforce-eager             # no CUDA-graph capture (~1-2 GiB) and no 13 s compile
  ```

  The log line `Model loading took X GiB memory` tells you the weights footprint (≈4.1 GiB for
  the 1.7B CustomVoice, ≈1.3 GiB for the 0.6B), and the startup warning
  `Free memory on device …` how much room the card actually has.

  A *different* failure — `StageEngineCoreProc_stage1_replica0 … torch.OutOfMemoryError` while
  loading `Qwen3TTSTokenizerV2Decoder` (or requests failing with
  `speech.audio.error: Stage-1 has no live replica`) — is not about the KV budget: the card
  ran out of room **between the two stages**. Qwen3-TTS is served as two stages (`0` = talker,
  `1` = `Code2Wav`) that live on the **same** GPU and each reserves its own fraction of that
  GPU's *total* memory (its bundled deploy YAML sets **0.3 + 0.3**, i.e. 60 % of the card, for
  both the 0.6B and the 1.7B). The weights only decide the split *inside* each budget
  (weights ↔ KV pool), never the total, so a **smaller model does not need less room** — which
  is why the same card can succeed with the 1.7B and fail with the 0.6B depending on what else
  was resident at that moment. A global `--gpu-memory-utilization` sets *both* stages to the
  same value — a no-op when the YAML already matches (Qwen3-TTS), but it also *raises* a stage
  whose own value is lower, so prefer `--stage-overrides` to give them different budgets (and
  mind that stage 0 must fit the talker's weights):

  ```yaml
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # what the OOM message itself suggests
    command: >
      vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
      --omni --host 0.0.0.0 --port 8091
      --max-model-len 2048        # talker *and* codec stream; the YAML's 65536 for stage 1 is huge
      --enforce-eager             # no CUDA-graph capture (~1-2 GiB) and no 13 s compile
      --stage-overrides '{"0": {"gpu_memory_utilization": 0.30}, "1": {"gpu_memory_utilization": 0.12}}'
  ```

  The keys are **stage ids, not GPU ids** — with one visible GPU both stages land on `cuda:0`
  (the YAML's `devices: "0"`). `--stage-overrides` beats the global flag and the YAML (see its
  precedence); the per-stage defaults live in the image's
  `vllm_omni/deploy/qwen3_tts.yaml`, worth reading when sizing. Keep stage 0 large enough for
  the talker's weights (`fraction × total VRAM` ≥ ~4.1 GiB for the 1.7B, ~1.3 GiB for the
  0.6B) and let stage 1 — a vocoder of well under 1 GiB — sit far below its 0.3 default.
  Lowering `1` from 0.3 to ~0.12 returns ~2.8 GiB to the rest of the card. The flag name has
  moved between releases (`--stage-configs-path` → `--deploy-config`), so confirm it with
  `vllm serve --help | grep -i stage`.

  Anything else holding VRAM on that GPU (typically the gateway's own copy of the family, if it
  is loaded) eats into the same budget: when `vllm.enabled` the gateway now **skips the
  `custom_voice` preload** (log: *not preloading 'custom_voice' — vLLM-Omni serves it*) and
  **unloads** its local copy as soon as vLLM is switched on, so start the container while the
  gateway is idle and it sizes its pools from free memory.

  `counters.tts_vllm_sessions` confirms the route is used (0 ⇒ the local engine, e.g. the
  service is unreachable and the call fell back). When vLLM does serve a family, the
  gateway **unloads its own copy** of it (logged) so the VRAM is not paid twice —
  note that the vLLM-Omni service itself always loads **two stages** (the talker and the
  `Code2Wav` decoder) on the GPU, which its startup log mentions explicitly.
- **Observability**: `/api/metrics` (TTFA, RTF, counters, queue depth, model
  loads/evictions/hot-swap thrash) and `/ready`. The app logs its own timings at INFO
  (`server.log_level`, now actually applied — editable from **Settings → Logs** or
  `PUT /api/settings {"server":{"log_level":"warning"}}`, effective immediately): one line per
  synthesis call
  (`TTS family=… chars=… gen=… total=… rtf=…`), one per model load/eviction, and — per chat
  turn — `chat tts: first audio Xs after the turn started (llm first token …s, first chunk
  queued …s, synth …s, N chars, Ys of audio, margin +Zs)` followed by one line per following
  chunk (`… +Ys, Zs buffered, margin +Ms`) and a `chat turn:` summary. The summary separates
  the LLM's **last content token** from the moment its HTTP stream closes (with llama.cpp the
  two differ by ~2 s, which is why `llm done` alone was misleading) and reports the worst
  `margin` — the audio still buffered on the client when a chunk arrives, so a negative margin
  is an audible gap. The web UI logs the same client-side figure in the browser console
  (`[chat] first audio … ms after send`).
- **Client jitter buffer**: `PCMPlayer` buffers the first ~200 ms before starting playback
  (forced flush after 250 ms for very short clips) to avoid underruns.

### Diagnosing a slow engine (Qwen3-TTS vs Chatterbox)

A large, constant per-request TTFA on Qwen (while Chatterbox feels instant) points at the
model family, not the pipeline. Measured on a 0.6B CustomVoice clone, synthesis costs
roughly `gen ≈ 0.6 s + 0.42 × audio_seconds` per call, against ~6-9 ms per character for a
fast LLM: a whole first sentence therefore starts playing *after* the reply is printed,
which is what the ~30-char first chunk (above) compensates for. Check the chat log line,
then `GET /api/metrics`:

- `counters.model_loads` growing every turn ⇒ the family is being **reloaded** (log:
  `Loaded … in X s`). The common cause is oscillating between two families (e.g. a preset
  `custom_voice` voice and a cloned `base` avatar) under `residency: single` (the default),
  where any switch reloads. Fix: `TTS_RESIDENCY=multi` (or `models.residency: multi`) with
  every family actually used in `models.resident` — a non-resident family is then evicted
  first. On a small GPU, staying on `single` and accepting the swap is the safer trade-off
  (several families plus STT in VRAM can exhaust it).
- `counters.model_hot_swap_thrash` > 0 confirms it (a model was evicted within 120 s of its
  last use). The engine logs `Hot-swap thrash: evicting …` and says whether the victim was
  pinned by `models.resident`.
- `counters.tts_cuda_oom` counts `CUDA out of memory` raised by a synthesis call. On a shared
  card this is usually *another* process holding the memory (typically a vLLM-Omni service
  whose reservation grew), not our own footprint: the engine logs the family/chars and the
  action to take. Note that vLLM sizes its KV pool from the **free** memory at startup, so
  start it before the gateway loads its models, and cap it with
  `--gpu-memory-utilization`/`--max-model-len` (see [backends](backends.md)).
- If loads stay flat, `chat tts: first audio … synth Xs` gives the synthesis cost of the
  first chunk: with the cost model above, a ~30-char first chunk costs ~0.6 s + 0.42 × 1.7 s
  ≈ 1.3 s, while a 70-char one costs ~2.2 s. A first audio that arrives only once the LLM
  reply is complete means the first chunk's synthesis is longer than the whole reply.

## STT sessions

`/ws/stt` proxies to a single upstream realtime session per client socket
(`_open_stt_upstream` tolerates a slow handshake and retries once). The upstream service
serves one session at a time, so a client that leaves its socket open keeps the slot busy
and the next handshake times out (`TimeoutError: timed out during opening handshake`).

`GET /api/metrics` counters:

- `stt_ws_opened` / `stt_ws_closed` — a growing gap means a client is leaking sessions;
- `stt_ws_failed` — upstream handshake timeouts (STT service busy, restarting or
  unreachable; check `docker compose logs stt`).

## Roadmap

### Phase 2 — remaining
- Server-side VAD/endpointing (the current implementation is a client-side energy VAD).
- Speculative start: begin the LLM request on streaming STT partials.
- Optional **Opus** encoding for WAN clients.
- `torch.compile` / CUDA graphs for the Qwen families.

### Phase 3 — remaining
- Validate the vLLM-Omni path against a real deployment and expose it per family.
- Full-duplex AEC reference so barge-in is robust on speakers (current: browser AEC).

## Tips for integrators

- Prefer `stream=true` / `/ws/tts` / `/ws/chat` and reuse one connection.
- Pre-warm the voice (`POST /api/voices/load`) at session start; keep the model resident.
- Buffer ~150–300 ms of PCM before playback; align to `X-Sample-Rate`.
- Cancel in-flight work on user speech (`input.cancel`) instead of dropping audio client-side.

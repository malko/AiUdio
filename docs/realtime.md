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
  chopped, word-by-word speech. The **first chunk** of a turn is the exception: it is cut at
  the earliest natural break past ~30 chars (a comma when one comes first, otherwise the end
  of a word — never inside a word), because the engine synthesizes audio ~4-6× slower than the
  LLM writes text, so a whole first sentence would only be spoken once the reply is printed.
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
- **vLLM-Omni streaming backend** (experimental): with `vllm.enabled`, CustomVoice
  streaming is served by vLLM-Omni (native codec-window PCM) via `app/vllm_backend.py`,
  falling back to the local per-sentence engine if the service is unavailable. Enable it
  without editing the state file:

  ```bash
  docker compose --profile vllm up -d tts-vllm     # the service
  curl -s -X PUT localhost:8004/api/settings -H 'Content-Type: application/json' \
       -d '{"vllm":{"enabled":true,"base_url":"http://tts-vllm:8091"}}'
  ```

  A manager that cannot pass a profile (Web UI, Dockge, Portainer…) should **delete the
  `profiles: ["vllm"]` line** of the `tts-vllm` service instead: it then belongs to the
  default stack and can be started/stopped from the UI. Keep `shm_size: "8gb"` (vLLM needs
  a large `/dev/shm`; Docker's default is 64 MB) and remember that the service reserves its
  own VRAM (see [backends](backends.md)), so keep it stopped when unused.

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

  `counters.tts_vllm_sessions` confirms the route is used (0 ⇒ the local engine, e.g. the
  service is unreachable and the call fell back).
- **Observability**: `/api/metrics` (TTFA, RTF, counters, queue depth, model
  loads/evictions/hot-swap thrash) and `/ready`. The app logs its own timings at INFO
  (`server.log_level`, now actually applied): one line per synthesis call
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

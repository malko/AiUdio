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
Chunking | wait for a boundary | first chunk released at the first clause/word boundary past ~40 chars (shorter sentences are merged)
TTS | model load + per-chunk synth | **per-sentence**, not token-level (`qwen-tts` exposes no codec streaming)
Transport | PCM16 over HTTP/WS | uncompressed mono 24 kHz
Playback | scheduling | client `PCMPlayer` with a small prebuffer (~200 ms / 250 ms timeout)

Measurable targets: **time-to-first-audio (TTFA)**, real-time factor (RTF), scheduler queue
depth, and per-stage timings (`/api/metrics`).

## Implemented (Phase 1 + 2/3 quick wins)

- **Streaming bypasses micro-batching**: `_stream_pcm` calls
  `BatchScheduler.submit_streaming()`, which runs immediately instead of waiting for the
  batch window (`generation.batch_window_ms`). Batching still applies to non-streaming REST.
- **Sentence-aware chunking**: `_pop_chat_chunks` releases a chunk at the first sentence
  boundary, merging sentences shorter than ~40 chars with the next one (a tiny first chunk
  such as "Bonjour !" would drain the playback buffer and cause an audible pause). Long
  run-on openings are still cut early at the first clause or word boundary.
- **Preload + optional warm-up**: `startup.tts_family` preloads a family at boot;
  `startup.warmup: true` runs a tiny synthesis to absorb CUDA init. `residency: multi`
  keeps several families hot.
- **Adaptive client endpointing**: the VAD shortens the silence window when the streaming
  transcript already ends with sentence punctuation (faster turn-taking).
- **Browser echo cancellation**: the mic is opened with `echoCancellation`,
  `noiseSuppression`, `autoGainControl`, so hands-free barge-in works without headphones.
- **vLLM-Omni streaming backend** (experimental): with `vllm.enabled`, CustomVoice
  streaming is served by vLLM-Omni (native codec-window PCM) via `app/vllm_backend.py`,
  falling back to the local per-sentence engine if the service is unavailable.
- **Observability**: `/api/metrics` (TTFA, RTF, counters, queue depth, model
  loads/evictions/hot-swap thrash) and `/ready`.
- **Client jitter buffer**: `PCMPlayer` buffers the first ~200 ms before starting playback
  (forced flush after 250 ms for very short clips) to avoid underruns.

### Diagnosing a slow engine (Qwen3-TTS vs Chatterbox)

A large, constant per-request TTFA on Qwen (while Chatterbox feels instant) points at the
model family, not the pipeline. Check `GET /api/metrics`:

- `counters.model_loads` growing every turn ⇒ the family is being **reloaded**. The common
  cause is oscillating between two families (e.g. a preset `custom_voice` voice and a cloned
  `base` avatar) under `residency: single`. Fix: `residency: multi` with both in
  `models.resident` (the shipped default), or pin a single voice.
- `counters.model_hot_swap_thrash` > 0 confirms it (a model was evicted within 120 s of its
  last use). The engine logs `Hot-swap thrash: evicting …`.
- If loads stay flat but `ttfa_seconds.avg` is high, it is the engine itself: use the 0.6B
  family, cap response length, and/or route CustomVoice through the `vllm` profile.

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

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
Chunking | wait for a boundary | first chunk released at the first clause/word boundary past `chat.min_chars`
TTS | model load + per-chunk synth | **per-sentence**, not token-level (`qwen-tts` exposes no codec streaming)
Transport | PCM16 over HTTP/WS | uncompressed mono 24 kHz
Playback | scheduling | client `PCMPlayer` with a small prebuffer (~200 ms / 250 ms timeout)

Measurable targets: **time-to-first-audio (TTFA)**, real-time factor (RTF), scheduler queue
depth, and per-stage timings (`/api/metrics`).

## Implemented (Phase 1 — quick wins)

- **Streaming bypasses micro-batching**: `_stream_pcm` calls
  `BatchScheduler.submit_streaming()`, which runs immediately instead of waiting for the
  batch window (`generation.batch_window_ms`). Batching still applies to non-streaming REST.
- **Tunable chunking**: `chat.min_chars` / `chat.max_chars` (config) control the TTFA/size
  trade-off without code changes.
- **Preload + optional warm-up**: `startup.tts_family` preloads a family at boot;
  `startup.warmup: true` runs a tiny synthesis to absorb CUDA init. `residency: multi`
  keeps several families hot.
- **Observability**: `/api/metrics` (TTFA, RTF, request/session counters, queue depth) and
  `/ready`.
- **Client jitter buffer**: `PCMPlayer` buffers the first ~200 ms before starting playback
  (forced flush after 250 ms for very short clips) to avoid underruns.

## Roadmap

### Phase 2 — throughput and quality
- Priority streaming lane; synthesize the first chunk as soon as possible and cap
  `max_new_tokens` for conversation.
- Use the **0.6B** family for chat; tune dtype/attention (`bfloat16`, `sdpa`), try
  `torch.compile` / CUDA graphs.
- **Server-side VAD/endpointing** (e.g. NeMo) to replace the fixed client silence window and
  cut the ~900 ms endpoint delay.
- **Speculative start**: begin the LLM request on streaming STT partials.
- Optional **Opus** encoding for WAN clients.
- Chatterbox: half precision / Turbo for lower per-utterance latency.

### Phase 3 — true streaming
- Introduce a streaming engine interface (`synthesize_stream`) so a backend can emit
  **codec-window PCM** instead of whole sentences; wire the `vllm` profile
  (`vllm.enabled`) to it, falling back to per-sentence.
- Echo cancellation / echo reference so barge-in works without headphones.

## Tips for integrators

- Prefer `stream=true` / `/ws/tts` / `/ws/chat` and reuse one connection.
- Pre-warm the voice (`POST /api/voices/load`) at session start; keep the model resident.
- Buffer ~150–300 ms of PCM before playback; align to `X-Sample-Rate`.
- Cancel in-flight work on user speech (`input.cancel`) instead of dropping audio client-side.

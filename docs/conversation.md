# Conversation (STT → LLM → TTS cascade)

This project does **not** serve an LLM: it connects to an external OpenAI-compatible
endpoint (llama.cpp, Ollama, vLLM, Text Generation WebUI, OpenAI...).

## Configuration

`config.yaml`:

```yaml
llm:
  base_url: http://host.docker.internal:8080/v1
  api_key: ""
  model: ""
  system_prompt: >-
    You are a helpful voice assistant. Answer concisely, in complete sentences.
  temperature: 0.7
  max_tokens: -1
  history_turns: 12
```

Or via `.env`: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Editable in the **Settings** tab
of the UI (persisted).

> From a container, the host is `host.docker.internal` (already mapped via `extra_hosts`).
> For an LLM on another LAN machine: use its IP (`http://192.168.x.y:8080/v1`).

## llama.cpp example

```bash
llama-server -hf ggml-org/gemma-3-4b-it-GGUF --host 0.0.0.0 --port 8080
```

Then `LLM_BASE_URL=http://host.docker.internal:8080/v1`.

## UI flow (Conversation tab)

```
🎤 mic ─▶ WS /ws/stt (streaming STT) ─▶ text
                                         │
                                         ▼
                   WS /ws/chat ─▶ LLM (llm.delta)
                                         │
                       sentence splitting
                                         ▼
                        per-sentence TTS ─▶ audio.{start,done} + PCM
                                         ▼
                                  gapless playback (AudioWorklet)
```

- **Push-to-talk**: the 🎤 button in the Conversation tab (sends the dictation to the field
  and then to the LLM).
- **Engine per session**: built-in voices force their engine (Qwen3-TTS); cloned voice-bank
  voices expose a selector (Qwen3-TTS / Chatterbox) next to the voice list. It defaults to
  the admin **Clone engine (default)** setting and is sent per request (`clone_backend`).
- **VAD (voice detection)**: the *VAD* button (shown only when a streaming STT model is
  active + the service is reachable) streams the mic continuously and auto-sends the utterance
  after a pause. The mic is opened with browser **echo cancellation** and the silence window
  shortens when the transcript already looks like a complete sentence (adaptive endpointing).
  The mic is muted while the assistant speaks to avoid feeding the TTS back into the STT.
- **Barge-in**: in VAD mode, speaking over the assistant (louder threshold, sustained ~300 ms)
  interrupts it: playback stops and the turn is cancelled, then the new utterance is captured.
- **Replay**: every assistant bubble has a 🔊 button. The last 8 replies' PCM is kept in the
  browser; older ones are re-synthesized on demand via `POST /api/tts` with the current voice.
- **Time to first audio**: TTS does **not** wait for the full LLM reply. The server synthesizes
  sentence by sentence as tokens arrive; only **complete sentences** are released (never a
  clause or a word, which the engine would render as a chopped utterance with its own
  intonation). A short opening sentence (< ~40 chars) is merged with the next one so playback
  does not drain, and once the first chunk is out the following sentences are grouped up to
  ~200 chars (`_pop_chat_chunks`), so a reply costs a few synthesis calls rather than one per
  sentence.

## Endpoints

- `POST /api/chat`: text-only conversation (SSE), useful to test the LLM without audio.
- `WS /ws/chat`: full cascade (LLM + TTS), one turn per `input.text`.

## Future work

- Improved barge-in (echo cancellation so the assistant can be interrupted without headphones).
- Named multiple sessions and persistent context.
- Tools / function calling (NeMo-Speech STT and vLLM-Omni support it at the transport level).

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
- **Barge-in**: at this stage, ongoing playback is not automatically interrupted by user
  speech; a **Stop** button is provided. Anti-echo barge-in is an improvement item (see
  "Future work").

## Endpoints

- `POST /api/chat`: text-only conversation (SSE), useful to test the LLM without audio.
- `WS /ws/chat`: full cascade (LLM + TTS), one turn per `input.text`.

## Future work

- True barge-in (cancel synthesis/playback as soon as voice is detected, AEC).
- Named multiple sessions and persistent context.
- Tools / function calling (NeMo-Speech STT and vLLM-Omni support it at the transport level).

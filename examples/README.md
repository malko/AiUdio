# Examples

Runnable clients for the AiUdio API. All read `AIUDIO_BASE_URL` (default
`http://localhost:8004`) and `AIUDIO_API_KEY` (omit when auth is disabled).

| Example | What it shows |
|---|---|
| `python/tts.py` | One-shot synthesis to a WAV file. |
| `python/tts_stream.py` | Streaming PCM over HTTP (low latency). |
| `python/chat_ws.py` | `/ws/chat` cascade (LLM + TTS), saving PCM. |
| `node/tts.mjs` | One-shot TTS (Node ≥ 18, built-in fetch). |
| `node/tts_stream.mjs` | Streaming PCM over HTTP (Node ≥ 18). |
| `node/chat.mjs` | `/ws/chat` cascade (Node ≥ 22, global WebSocket). |
| `curl/README.md` | Copy-paste `curl` commands for every flow. |

```bash
export AIUDIO_BASE_URL=http://localhost:8004
export AIUDIO_API_KEY=aiud_...        # only if auth is enabled

python3 python/tts.py "Hello world"
python3 python/tts_stream.py "Sentence one. Sentence two." && \
  ffplay -f s16le -ar 24000 -ch_layout mono out.pcm
python3 python/chat_ws.py "Give me a fun fact."

node node/tts.mjs "Hello world"
node node/chat.mjs "Hello, how are you?"
```

The streaming examples write raw **mono signed-16-bit little-endian** PCM at the sample
rate announced in the `X-Sample-Rate` header (24 kHz by default). In a real app, feed those
bytes to a small **jitter buffer** (~150–300 ms) before playback — see
[`../docs/integration.md`](../docs/integration.md).

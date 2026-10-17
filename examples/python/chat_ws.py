#!/usr/bin/env python3
"""Conversation over /ws/chat: LLM reply streamed as text + PCM audio.

Saves the assistant audio to reply.pcm and prints the events.
Play back with: ffplay -f s16le -ar 24000 -ch_layout mono reply.pcm
"""

import asyncio
import json
import os
import sys

import websockets

BASE = os.environ.get("AIUDIO_BASE_URL", "http://localhost:8004")
KEY = os.environ.get("AIUDIO_API_KEY", "")


def ws_url() -> str:
    base = BASE.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    return f"{base}/ws/chat" + (f"?api_key={KEY}" if KEY else "")


async def main() -> None:
    text = " ".join(sys.argv[1:]) or "Hello, how are you?"
    async with websockets.connect(ws_url()) as ws:
        with open("reply.pcm", "wb") as pcm:
            await ws.send(json.dumps({
                "type": "session.config", "speak": True, "tts": {"speaker": "Vivian"},
            }))
            await ws.send(json.dumps({"type": "input.text", "text": text}))
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    pcm.write(message)  # PCM16, sample rate at audio.start
                    continue
                event = json.loads(message)
                kind = event.get("type")
                if kind == "llm.delta":
                    sys.stdout.write(event.get("text", ""))
                    sys.stdout.flush()
                elif kind in ("transcript", "llm.done", "turn.cancelled", "error"):
                    print(f"\n[{kind}]", event.get("text") or event.get("message") or "")
                if kind in ("llm.done", "turn.cancelled"):
                    break
    print("wrote reply.pcm")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Realtime speech-to-text over /ws/stt.

The gateway is a passthrough to NeMo-Speech.cpp's realtime endpoint, which expects
raw **PCM16 mono 16 kHz** frames. This example streams a WAV file and prints the
transcripts. A **streaming** STT model must be active (see GET /api/stt/models).

Usage: python3 stt_ws.py audio.wav
Env: AIUDIO_BASE_URL, AIUDIO_API_KEY, AIUDIO_STT_LANGUAGE (e.g. "English").

Resample anything else first, e.g.:
  ffmpeg -i input.wav -ac 1 -ar 16000 -sample_fmt s16 audio16k.wav
"""

import asyncio
import json
import os
import sys
import wave

import websockets

BASE = os.environ.get("AIUDIO_BASE_URL", "http://localhost:8004")
KEY = os.environ.get("AIUDIO_API_KEY", "")
FRAME_BYTES = 3200  # 100 ms of 16 kHz mono s16le


def ws_url() -> str:
    base = BASE.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    return f"{base}/ws/stt" + (f"?api_key={KEY}" if KEY else "")


def read_pcm16(path: str) -> bytes:
    """Read a WAV as mono s16le at 16 kHz (no resampling: resample upstream)."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise SystemExit("only 16-bit PCM WAV is supported")
        rate = wf.getframerate()
        if rate != 16000:
            raise SystemExit(
                f"expected 16 kHz, got {rate} Hz — resample first "
                "(ffmpeg -i in.wav -ac 1 -ar 16000 -sample_fmt s16 out.wav)"
            )
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    if channels == 2:  # keep the left channel of an interleaved stereo WAV
        raw = b"".join(raw[i:i + 2] for i in range(0, len(raw), 4))
    elif channels != 1:
        raise SystemExit(f"expected 1 or 2 channels, got {channels}")
    return raw


async def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
    pcm = read_pcm16(path)
    transcript = ""

    async with websockets.connect(ws_url(), max_size=None, open_timeout=30.0) as ws:
        session = {"sample_rate": 16000, "automatic_punctuation": True}
        if os.environ.get("AIUDIO_STT_LANGUAGE"):
            session["language"] = os.environ["AIUDIO_STT_LANGUAGE"]
        await ws.send(json.dumps({"type": "session.update", "session": session}))

        async def reader() -> None:
            nonlocal transcript
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    continue
                event = json.loads(message)
                kind = event.get("type", "")
                if kind.endswith("delta"):
                    sys.stdout.write(event.get("delta", ""))
                    sys.stdout.flush()
                elif kind.endswith("completed"):
                    text = event.get("transcript") or event.get("text") or ""
                    if text:
                        transcript = f"{transcript} {text}".strip()
                        print(f"\n[final] {text}")
                elif kind == "error":
                    print(f"\n[error] {event.get('message')}", file=sys.stderr)

        task = asyncio.create_task(reader())
        for i in range(0, len(pcm), FRAME_BYTES):
            await ws.send(pcm[i:i + FRAME_BYTES])
        # Tell the service the utterance is complete, then let the final event land.
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await asyncio.sleep(1.0)
        task.cancel()

    print(f"transcript: {transcript}")


if __name__ == "__main__":
    asyncio.run(main())

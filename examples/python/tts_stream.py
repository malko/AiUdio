#!/usr/bin/env python3
"""Streaming TTS: write mono s16le PCM (see X-Sample-Rate) to out.pcm.

Play back with: ffplay -f s16le -ar 24000 -ch_layout mono out.pcm
"""

import os
import sys

import httpx

BASE = os.environ.get("AIUDIO_BASE_URL", "http://localhost:8004")
KEY = os.environ.get("AIUDIO_API_KEY", "")
HEADERS = {"X-API-Key": KEY} if KEY else {}


def main() -> None:
    text = " ".join(sys.argv[1:]) or "Sentence one. Sentence two. Sentence three."
    total = 0
    with httpx.stream(
        "POST",
        f"{BASE}/api/tts",
        headers=HEADERS,
        timeout=120.0,
        json={"text": text, "stream": True, "format": "pcm"},
    ) as resp:
        resp.raise_for_status()
        sample_rate = resp.headers.get("x-sample-rate", "24000")
        with open("out.pcm", "wb") as fh:
            for chunk in resp.iter_bytes():
                # A real app would push `chunk` into a jitter buffer here.
                fh.write(chunk)
                total += len(chunk)
    print(f"wrote out.pcm ({total} bytes, {sample_rate} Hz)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-shot TTS to a WAV file. Usage: python3 tts.py "Hello world" """

import os
import sys

import httpx

BASE = os.environ.get("AIUDIO_BASE_URL", "http://localhost:8004")
KEY = os.environ.get("AIUDIO_API_KEY", "")
HEADERS = {"X-API-Key": KEY} if KEY else {}


def main() -> None:
    text = " ".join(sys.argv[1:]) or "Hello from AiUdio."
    r = httpx.post(
        f"{BASE}/api/tts",
        headers=HEADERS,
        timeout=120.0,
        json={
            "text": text,
            "family": "custom_voice",
            "speaker": "Vivian",
            "language": "English",
        },
    )
    r.raise_for_status()
    with open("out.wav", "wb") as fh:
        fh.write(r.content)
    print(f"wrote out.wav ({len(r.content)} bytes)")


if __name__ == "__main__":
    main()

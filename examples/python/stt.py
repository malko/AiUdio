#!/usr/bin/env python3
"""Speech-to-text (file): POST /v1/audio/transcriptions.

Usage: python3 stt.py audio.wav
Env: AIUDIO_BASE_URL (default http://localhost:8004), AIUDIO_API_KEY (if auth).
"""

import os
import sys

import httpx

BASE = os.environ.get("AIUDIO_BASE_URL", "http://localhost:8004")
KEY = os.environ.get("AIUDIO_API_KEY", "")
HEADERS = {"X-API-Key": KEY} if KEY else {}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
    with open(path, "rb") as fh:
        r = httpx.post(
            f"{BASE}/v1/audio/transcriptions",
            headers=HEADERS,
            files={"file": (os.path.basename(path), fh, "audio/wav")},
            data={"response_format": "json"},
            timeout=300.0,
        )
    r.raise_for_status()
    try:
        print(r.json().get("text", r.text))
    except ValueError:
        print(r.text)


if __name__ == "__main__":
    main()

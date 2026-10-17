"""Experimental vLLM-Omni streaming client (native PCM, not per-sentence).

When ``vllm.enabled`` is true, the gateway can route streaming synthesis of the
``custom_voice`` family to a vLLM-Omni service instead of synthesizing sentence
by sentence locally. The contract mirrors the OpenAI speech API with
``stream=true`` + ``response_format=pcm``: the service returns chunks of mono
PCM16 decoded on the fly.

If the service is unreachable, the caller falls back to the local engine.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

log = logging.getLogger(__name__)


def speech_payload(req, text: str, sample_rate: Optional[int] = None) -> Dict[str, Any]:
    """Map a ``TTSRequest`` to the vLLM-Omni OpenAI-style speech payload.

    ``stream_format: "audio"`` is what asks for **raw PCM16 bytes** over HTTP.
    With ``stream: true`` alone, vLLM-Omni answers the OpenAI ``speech.audio.*``
    SSE events (base64 deltas), which cannot be forwarded to our clients as PCM.
    """
    payload: Dict[str, Any] = {
        "input": text,
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
    }
    if sample_rate:
        payload["sample_rate"] = int(sample_rate)
    voice = req.resolve_speaker()
    if voice:
        # The served CustomVoice speakers are lowercase ('vivian', 'serena'…).
        payload["voice"] = voice.strip().lower()
    return payload


def speech_url(base_url: str, path: str = "/v1/audio/speech") -> str:
    """Absolute URL of the speech route (``path`` is configurable: vLLM-Omni also
    exposes ``/v1/audio/speech/stream``)."""
    return f"{base_url.rstrip('/')}/{(path or '/v1/audio/speech').lstrip('/')}"


async def stream_pcm(
    base_url: str,
    payload: Dict[str, Any],
    *,
    path: str = "/v1/audio/speech",
    timeout: float = 300.0,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[bytes]:
    """Yield raw PCM16 chunks from ``POST {base_url}{path}``."""
    url = speech_url(base_url, path)
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))
    try:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk
    finally:
        if owns_client:
            await client.aclose()

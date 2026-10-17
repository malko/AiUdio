"""Experimental vLLM-Omni streaming client (native PCM, not per-sentence).

When ``vllm.enabled`` is true, the gateway can route streaming synthesis of one
family to a vLLM-Omni service instead of synthesizing sentence by sentence
locally: ``custom_voice`` (preset speakers) or ``base`` (voice cloning from a
reference clip, sent inline as a base64 data URL). The contract mirrors the OpenAI speech API with
``stream=true`` + ``response_format=pcm``: the service returns chunks of mono
PCM16 decoded on the fly.

If the service is unreachable, the caller falls back to the local engine.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

log = logging.getLogger(__name__)


def speech_payload(
    req,
    text: str,
    sample_rate: Optional[int] = None,
    *,
    family: str = "custom_voice",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
) -> Dict[str, Any]:
    """Map a ``TTSRequest`` to the vLLM-Omni OpenAI-style speech payload.

    ``stream_format: "audio"`` is what asks for **raw PCM16 bytes** over HTTP.
    With ``stream: true`` alone, vLLM-Omni answers the OpenAI ``speech.audio.*``
    SSE events (base64 deltas), which cannot be forwarded to our clients as PCM.

    For ``base`` (voice cloning) the reference clip is sent as ``ref_audio`` with
    ``task_type="Base"`` — the service infers the task anyway, but being explicit
    turns a checkpoint mismatch into a clear 400 instead of empty audio. A
    transcript (``ref_text``) selects in-context cloning (better prosody); without
    one the service falls back to the speaker embedding (``x_vector_only_mode``).
    """
    payload: Dict[str, Any] = {
        "input": text,
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
    }
    if sample_rate:
        payload["sample_rate"] = int(sample_rate)
    if family == "base":
        payload["task_type"] = "Base"
        if ref_audio:
            payload["ref_audio"] = ref_audio
        if ref_text:
            payload["ref_text"] = ref_text
        payload["x_vector_only_mode"] = bool(x_vector_only_mode) or not ref_text
        return payload
    voice = req.resolve_speaker()
    if voice:
        # The served CustomVoice speakers are lowercase ('vivian', 'serena'…).
        payload["voice"] = voice.strip().lower()
    return payload


def audio_data_url(data: bytes, mime: str = "audio/wav") -> str:
    """Inline reference clip for ``ref_audio`` (base64 data URL).

    The service accepts an HTTP URL, a ``file://`` URI under
    ``--allowed-local-media-path`` or a data URL, and caches the *decoded* clip by
    the locator string — so an inline, stable locator avoids mounting our voice
    directory into the vLLM container.
    """
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def family_for_model(model_id: str) -> Optional[str]:
    """Best-effort family of a served checkpoint id (``…-Base`` → ``base``)."""
    lowered = (model_id or "").lower()
    if lowered.endswith("-base") or "-base-" in lowered:
        return "base"
    if "customvoice" in lowered:
        return "custom_voice"
    return None


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

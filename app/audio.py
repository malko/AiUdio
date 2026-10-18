"""Audio encoding (PCM16, WAV) with no GPU dependency."""

from __future__ import annotations

import io
import wave

import numpy as np


def to_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return np.clip(arr, -1.0, 1.0)


def to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float32 [-1, 1] signal to mono little-endian PCM16."""
    arr = to_float32(audio)
    return (arr * 32767.0).astype("<i2").tobytes()


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 signal as a mono 16-bit WAV."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(to_pcm16(audio))
    return buf.getvalue()

"""Audio encoding/decoding (PCM16, WAV, base64) with no GPU dependency."""

from __future__ import annotations

import base64
import io
import wave
from typing import Tuple

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


def pcm16_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 signal as a mono 16-bit WAV."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(to_pcm16(audio))
    return buf.getvalue()


def decode_audio(data: bytes, target_sr: int) -> Tuple[np.ndarray, int]:
    """Decode an audio file (WAV/FLAC/…) to mono float32 via soundfile."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    audio = to_float32(audio)
    if sr != target_sr:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(target_sr))
            sr = target_sr
        except Exception:  # pragma: no cover - librosa missing
            pass
    return audio, int(sr)


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def parse_audio_ref(ref: str) -> str:
    """Normalize an audio reference: path, URL or base64 data-URL."""
    return ref.strip() if ref else ""

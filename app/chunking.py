"""Text splitting into sentences/chunks and audio concatenation."""

from __future__ import annotations

import re
from typing import List

import numpy as np

# Common abbreviations (FR/EN) that do not end a sentence.
_ABBREVIATIONS = {
    "m", "mm", "mme", "mr", "mrs", "ms", "dr", "prof", "etc", "ex",
    "cf", "env", "vs", "av", "apr", "no", "nos", "art", "fig", "p",
    "e.g", "i.e", "al", "st", "ste",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")
_HARD_SPLIT = re.compile(r"(?<=[,;:–—-])\s+")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, avoiding false breaks (abbreviations)."""
    text = (text or "").strip()
    if not text:
        return []

    raw = _SENTENCE_SPLIT.split(text)
    sentences: List[str] = []
    buffer = ""
    for part in raw:
        candidate = f"{buffer} {part}".strip() if buffer else part.strip()
        last = candidate.rstrip(".").split()[-1].lower() if candidate else ""
        if last in _ABBREVIATIONS or re.fullmatch(r"[A-Za-z]\.", candidate.split()[-1] if candidate else ""):
            buffer = candidate
            continue
        if candidate:
            sentences.append(candidate)
        buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences


def chunk_text(text: str, max_chars: int = 240) -> List[str]:
    """Group sentences into chunks <= max_chars, splitting overly long sentences."""
    max_chars = max(40, int(max_chars))
    chunks: List[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sentence in split_sentences(text):
        if len(sentence) > max_chars:
            flush()
            pieces, piece = [], ""
            for token in _HARD_SPLIT.split(sentence):
                if piece and len(piece) + len(token) + 1 > max_chars:
                    pieces.append(piece.strip())
                    piece = token
                else:
                    piece = f"{piece} {token}".strip() if piece else token
            if piece.strip():
                pieces.append(piece.strip())
            # Last resort: hard split on spaces.
            for p in pieces:
                while len(p) > max_chars:
                    cut = p.rfind(" ", 0, max_chars) or max_chars
                    chunks.append(p[:cut].strip())
                    p = p[cut:].strip()
                if p:
                    chunks.append(p)
            continue

        if current and len(current) + len(sentence) + 1 > max_chars:
            flush()
        current = f"{current} {sentence}".strip() if current else sentence

    flush()
    return chunks


def concat_audio(chunks: List[np.ndarray], sample_rate: int, silence_ms: int = 120) -> np.ndarray:
    """Concatenate mono float32 waveforms, inserting a short silence."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(int(sample_rate * max(0, silence_ms) / 1000.0), dtype=np.float32)
    parts: List[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        if i:
            parts.append(silence)
        parts.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

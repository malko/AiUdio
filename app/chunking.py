"""Text splitting into sentences/chunks and audio concatenation."""

from __future__ import annotations

import logging
import re
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

# Chunk-size bounds. Under the floor a chunk is too short to sound natural; over
# the ceiling a single call has to synthesize more audio than an engine can
# produce in one generation (Chatterbox stops at `max_new_tokens=1000` ≈ 40 s at
# 25 Hz, i.e. roughly 600 characters of Latin text) — the excess would be dropped
# silently. A longer input is split into several chunks instead.
MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 600

# Common abbreviations (FR/EN) that do not end a sentence.
_ABBREVIATIONS = {
    "m", "mm", "mme", "mr", "mrs", "ms", "dr", "prof", "etc", "ex",
    "cf", "env", "vs", "av", "apr", "no", "nos", "art", "fig", "p",
    "e.g", "i.e", "al", "st", "ste",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…؟۔।])\s+|(?<=[。！？])\s*")
_HARD_SPLIT = re.compile(r"(?<=[,;:–—-])\s+")

# Boundary classes, strongest first. A chunk ends on a sentence, else on a
# clause, else on a line/paragraph break, else between words — never inside one.
_BOUNDARY_CLASSES = (
    ".!?…؟۔।。！？",       # end of sentence (CJK ones need no trailing space)
    ",;:—–、，；：",        # end of clause
    "\n",                   # end of line / paragraph
    " \t",                  # between words
)
_BOUNDARY_RANK = {
    ch: rank for rank, chars in enumerate(_BOUNDARY_CLASSES) for ch in chars
}

# How far back a cut may move to land on a strong boundary (and how much shorter
# than `max_chars` a chunk may get because of it).
_MIN_LOOKBACK = 24


def _cut_point(text: str, limit: int) -> int:
    """Best cut index at most ``limit``: sentence > clause > line > word."""
    window = max(_MIN_LOOKBACK, limit // 4)
    found: Dict[int, int] = {}
    for idx in range(limit, max(0, limit - window), -1):
        rank = _BOUNDARY_RANK.get(text[idx - 1])
        if rank is not None and rank not in found:
            found[rank] = idx
    if found:
        return found[min(found)]
    # Nothing strong nearby: fall back to the nearest word/line break, without
    # cutting the chunk below half the budget. A *single* token longer than the
    # budget is the only case where a cut inside it is unavoidable.
    floor = max(1, limit // 2)
    for idx in range(limit, floor - 1, -1):
        if text[idx - 1] in " \t\n":
            return idx
    return limit


# CJK sentences carry their own terminator and are not separated by spaces.
_NO_SPACE_AFTER = "。！？"


def _join_sentences(current: str, sentence: str) -> str:
    if not current:
        return sentence
    separator = "" if current[-1] in _NO_SPACE_AFTER else " "
    return f"{current}{separator}{sentence}"


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
    """Group sentences into chunks <= max_chars, splitting overly long sentences.

    ``max_chars`` is clamped to ``[MIN_CHUNK_CHARS, MAX_CHUNK_CHARS]``: above the
    ceiling one model call would have to generate more audio than it can, and the
    tail would be lost silently.
    """
    requested = int(max_chars)
    max_chars = min(MAX_CHUNK_CHARS, max(MIN_CHUNK_CHARS, requested))
    if max_chars != requested:
        log.warning(
            "chunking.max_chars=%s out of [%d, %d]: using %d",
            requested, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS, max_chars,
        )
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
            # Last resort: cut this piece, landing on a boundary when possible.
            for p in pieces:
                while len(p) > max_chars:
                    cut = _cut_point(p, max_chars)
                    chunks.append(p[:cut].strip())
                    p = p[cut:].strip()
                if p:
                    chunks.append(p)
            continue

        if current and len(current) + len(sentence) + 1 > max_chars:
            flush()
        current = _join_sentences(current, sentence)

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

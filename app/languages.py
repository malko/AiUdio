"""Language name <-> ISO code helpers.

The app speaks human-readable language names ("French", "Chinese", ...) while some
backends (Chatterbox) expect ISO-639-1 codes ("fr", "zh"). This module centralizes
the translation so the engine/adapters stay readable.
"""

from __future__ import annotations

from typing import List, Optional

# ISO-639-1 code -> English name. Covers the Qwen3-TTS set plus the broader set
# supported by Chatterbox Multilingual.
ISO_TO_NAME = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

# Normalized English name -> ISO code (built from the table, plus a couple of
# common aliases).
_NAME_TO_ISO = {name.lower(): code for code, name in ISO_TO_NAME.items()}
_NAME_TO_ISO.update(
    {
        "mandarin": "zh",
        "castilian": "es",
    }
)


def to_iso(name: Optional[str]) -> Optional[str]:
    """Map a language name (or an already-ISO code) to a lowercase ISO code."""
    if not name:
        return None
    value = str(name).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in ISO_TO_NAME:
        return lowered
    return _NAME_TO_ISO.get(lowered)


def to_name(code: Optional[str]) -> str:
    """Map an ISO code to the app's English display name (best effort)."""
    if not code:
        return ""
    lowered = str(code).strip().lower()
    return ISO_TO_NAME.get(lowered, str(code))


def names(codes: List[str]) -> List[str]:
    return [to_name(c) for c in codes]

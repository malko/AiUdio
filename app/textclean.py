"""Input-text cleanup for TTS.

Speech engines cannot pronounce everything a client sends: emojis, zero-width
characters, control codes, and the Markdown an LLM leaves in its answers. What
to do with them is a **policy** (globally, and per API key for the ``text``
block), so it lives here — in the gateway, applied *before* the text is chunked —
rather than inside each engine (which only sees a string).

Levels (``Policy.cleanup``):

- ``off`` (default): nothing, byte for byte what the client sent;
- ``light``: characters that carry no sound — control codes, zero-width
  joiners/BOM, soft hyphens — plus whitespace normalization and typographic
  punctuation folded to ASCII. Sentence punctuation is never removed (the
  prosody depends on it);
- ``llm``: ``light`` plus the Markdown noise of an LLM answer (``**bold**``,
  ``# headings``, ``> quotes``, bullets, ``[text](url)``, inline code). Fenced
  code blocks are dropped or announced (``Policy.code_blocks``) and never read
  aloud.

Emojis follow their own policy (``Policy.emoji``) at every level, independently
of ``cleanup``: ``keep``, ``strip`` (removed — as a space, so neighbouring words
do not get glued together), or ``name`` (the ~20 most frequent ones are spoken,
in French or English following the request language, and the rest is stripped).

The module is pure apart from ``CleanState``, the small per-conversation object
the streaming chat passes back in so an open fenced block is remembered across
chunks.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------- policy
CHOICES = {
    "cleanup": ("off", "light", "llm"),
    "emoji": ("keep", "strip", "name"),
    "code_blocks": ("drop", "announce"),
}
_DEFAULT = {"cleanup": "off", "emoji": "keep", "code_blocks": "drop"}


class Policy(BaseModel):
    """What to do with non-speech input (see the module docstring)."""

    cleanup: str = "off"        # off | light | llm
    emoji: str = "keep"         # keep | strip | name
    code_blocks: str = "drop"   # drop | announce (only used by `llm`)

    @field_validator("cleanup", "emoji", "code_blocks")
    @classmethod
    def _known(cls, value: str, info: ValidationInfo) -> str:
        field = info.field_name
        allowed = CHOICES[field]
        text = str(value or "").strip().lower()
        if text in allowed:
            return text
        log.warning(
            "Ignoring text.%s=%r (expected one of %s): using %s",
            field, value, ", ".join(allowed), _DEFAULT[field],
        )
        return _DEFAULT[field]


def resolve(base: Policy, override: Optional[Policy]) -> Policy:
    """Merge a per-client ``override`` onto the global ``base`` policy.

    Only the fields the override actually sets win, so a key can change the
    emoji handling and keep the global cleanup level.
    """
    if override is None:
        return base
    return base.model_copy(update=override.model_dump(exclude_unset=True))


@dataclass
class CleanState:
    """Per-conversation state: a code fence may span several chat chunks."""

    in_code: bool = False


# ------------------------------------------------------------------ characters
# Control codes (keep \t and \n), BOM, zero-width joiners/spaces, soft hyphen.
_INVISIBLE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    "\u00ad\u180e\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063"
    "\u20e3\ufeff\ufe00-\ufe0f]"
)
# Unicode spaces (not \n) collapse to a plain space.
_SPACES = re.compile("[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
_TYPO = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00ab": '"', "\u00bb": '"',
}


def _strip_invisible(text: str) -> str:
    text = _INVISIBLE.sub("", text)
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep the text composed (engines see one code point per accented letter).
    return unicodedata.normalize("NFC", text)


def _fold_punctuation(text: str) -> str:
    for src, dst in _TYPO.items():
        if src in text:
            text = text.replace(src, dst)
    return text


def _normalize_whitespace(text: str) -> str:
    text = _SPACES.sub(" ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text.strip()


# ----------------------------------------------------------------------- emoji
# Pictographic blocks (letters, arrows and math symbols are deliberately out of
# scope: `emoji: keep` must not silently drop them).
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),   # pictographs, emoticons, transport, extended-A
    (0x2600, 0x27BF),     # misc symbols, dingbats
    (0x2B00, 0x2BFF),     # misc symbols and arrows (⭐⭕⬛…)
    (0x203C, 0x203C), (0x2049, 0x2049), (0x2139, 0x2139),
    (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3297), (0x3299, 0x3299),
)
_ZWJ = 0x200D
_VARIATIONS = (0xFE0E, 0xFE0F, 0x20E3)     # text/emoji style, keycap
_KEYCAP = re.compile("([0-9#*])\ufe0f?\u20e3")
_SKIN_TONES = (0x1F3FB, 0x1F3FF)
_TAGS = (0xE0020, 0xE007F)                 # subdivision flags
_REGIONAL = (0x1F1E6, 0x1F1FF)             # flag pairs

# The most frequent emojis of LLM/chat text: spoken as a short word, FR and EN.
_EMOJI_NAMES: Dict[int, Tuple[str, str]] = {
    0x1F600: ("sourire", "smiley"),
    0x1F605: ("rires", "laughing"),
    0x1F602: ("rires", "laughing"),
    0x1F60A: ("sourire", "smiling"),
    0x1F609: ("clin d'œil", "wink"),
    0x1F60D: ("cœur", "heart eyes"),
    0x1F618: ("bisou", "kiss"),
    0x1F60E: ("cool", "cool"),
    0x1F914: ("réflexion", "thinking"),
    0x1F644: ("yeux au ciel", "eye roll"),
    0x1F622: ("tristesse", "sad"),
    0x1F62D: ("pleurs", "crying"),
    0x1F621: ("colère", "angry"),
    0x1F631: ("surprise", "shocked"),
    0x1F44D: ("bravo", "thumbs up"),
    0x1F44E: ("pouce en bas", "thumbs down"),
    0x1F44F: ("applaudissements", "applause"),
    0x1F64C: ("applaudissements", "cheering"),
    0x1F64F: ("merci", "thanks"),
    0x2764: ("cœur", "heart"),
    0x1F525: ("feu", "fire"),
    0x2728: ("étincelles", "sparkles"),
    0x1F389: ("fête", "party"),
    0x2705: ("validé", "check mark"),
    0x274C: ("erreur", "cross mark"),
    0x26A0: ("attention", "warning"),
    0x1F440: ("yeux", "eyes"),
    0x1F4A1: ("idée", "idea"),
    0x1F4AF: ("cent sur cent", "hundred"),
}


def _in(codepoint: int, ranges) -> bool:
    return any(lo <= codepoint <= hi for lo, hi in ranges)


def _is_emoji(ch: str) -> bool:
    return _in(ord(ch), _EMOJI_RANGES)


def _emoji_clusters(text: str) -> List[Tuple[int, int, int]]:
    """``(start, end, base codepoint)`` of each emoji, joiners included.

    A family (``👨‍👩‍👧‍👦``), a flag (regional indicator pair), a skin tone or a
    keycap sequence is one cluster, so nothing is counted or replaced twice.
    """
    clusters: List[Tuple[int, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if not _is_emoji(text[i]):
            i += 1
            continue
        start, base = i, ord(text[i])
        i += 1
        if _in(base, (_REGIONAL,)) and i < n and _in(ord(text[i]), (_REGIONAL,)):
            i += 1
        while i < n:
            cp = ord(text[i])
            if cp == _ZWJ:
                if i + 1 < n and _is_emoji(text[i + 1]):
                    i += 2
                    continue
                i += 1
                continue
            if cp in _VARIATIONS or _in(cp, (_SKIN_TONES, _TAGS)):
                i += 1
                continue
            break
        clusters.append((start, i, base))
    return clusters


def _is_french(language: Optional[str]) -> bool:
    return str(language or "").strip().lower().startswith(("fr", "french"))


def _emoji_replacement(base: int, policy: str, language: Optional[str]) -> str:
    if policy == "name":
        names = _EMOJI_NAMES.get(base)
        if names:
            return f" {names[0] if _is_french(language) else names[1]} "
    # A space rather than nothing: "fin😀de" must not become "finde".
    return " "


def _apply_emoji(text: str, policy: str, language: Optional[str]) -> str:
    # A keycap sequence ("1️⃣") is a digit plus combiners: keep the digit.
    text = _KEYCAP.sub(r"\1", text)
    clusters = _emoji_clusters(text)
    if not clusters:
        return text
    out, last = [], 0
    for start, end, base in clusters:
        out.append(text[last:start])
        out.append(_emoji_replacement(base, policy, language))
        last = end
    out.append(text[last:])
    return "".join(out)


# -------------------------------------------------------------------- markdown
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_RULE = re.compile(r"^[ \t]*([-*_][ \t]*){3,}$", re.M)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M)
_QUOTE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.M)
_BULLET = re.compile(r"^[ \t]*[-*+•][ \t]+", re.M)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_ITALIC_OPEN = re.compile(r"(?<![0-9A-Za-z_])_{1,2}(?=\S)")
_ITALIC_CLOSE = re.compile(r"(?<=\S)_{1,2}(?![0-9A-Za-z_])")


def _announce(language: Optional[str]) -> str:
    return " Bloc de code. " if _is_french(language) else " Code block. "


def _fenced_blocks(text: str, code_blocks: str, language: Optional[str],
                   state: CleanState) -> str:
    """Drop (or announce) fenced code, remembering an open fence in ``state``."""
    out: List[str] = []
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            if state.in_code:
                state.in_code = False
                if code_blocks == "announce":
                    out.append(_announce(language))
            else:
                state.in_code = True
            out.append("\n")
            continue
        out.append("" if state.in_code else line)
    return "".join(out)


def _markdown(text: str, code_blocks: str, language: Optional[str],
              state: CleanState) -> str:
    text = _fenced_blocks(text, code_blocks, language, state)
    if state.in_code:  # inside a block: nothing else to clean
        return text
    text = _RULE.sub("", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _ITALIC_OPEN.sub("", text)
    text = _ITALIC_CLOSE.sub("", text)
    return text.replace("*", "")


# ------------------------------------------------------------------------ api
def clean(text: str, policy: Policy, language: Optional[str] = None,
          state: Optional[CleanState] = None) -> Tuple[str, CleanState]:
    """Clean ``text``; returns ``(text, state)`` (pass ``state`` back on a stream).

    ``policy.emoji`` acts on its own: with ``cleanup: off`` and ``emoji: strip``
    only the emojis are removed.
    """
    state = state if state is not None else CleanState()
    if not text:
        return text, state
    cleanup = policy.cleanup if policy.cleanup in CHOICES["cleanup"] else "off"
    emoji = policy.emoji if policy.emoji in CHOICES["emoji"] else "keep"
    code_blocks = policy.code_blocks if policy.code_blocks in CHOICES["code_blocks"] else "drop"
    if cleanup == "off" and emoji == "keep":
        return text, state

    out = text
    touched = False
    if cleanup != "off":
        out = _strip_invisible(out)
        touched = True
    if emoji != "keep":
        out = _apply_emoji(out, emoji, language)
        touched = True
    if cleanup == "llm":
        out = _markdown(out, code_blocks, language, state)
        touched = True
    if cleanup != "off":
        out = _fold_punctuation(out)
    return (_normalize_whitespace(out) if touched else out), state

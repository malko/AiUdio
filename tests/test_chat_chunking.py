import asyncio

import numpy as np

from app.config import Settings
from app.main import (
    _CHAT_COALESCE_CHARS,
    _CHAT_MAX_CHARS,
    _CHAT_MIN_CHARS,
    _chat_turn,
    _pop_chat_chunks,
)


class _Tokens:
    """LLM stub streaming the given tokens."""

    def __init__(self, tokens):
        self.tokens = tokens

    def stream(self, messages):
        async def gen():
            for token in self.tokens:
                yield token

        return gen()


class _Scheduler:
    """Records the synthesized texts (instead of running a GPU model)."""

    def __init__(self):
        self.texts = []

    async def submit_streaming(self, family, item):
        self.texts.append(item["text"])
        return np.zeros(2400, dtype=np.float32), 24000


class _Socket:
    def __init__(self):
        self.events = []

    async def send_json(self, obj):
        self.events.append(obj)

    async def send_bytes(self, payload):
        pass


def _run_turn(tokens) -> _Scheduler:
    scheduler = _Scheduler()
    asyncio.run(
        _chat_turn(
            _Socket(), Settings(), scheduler, _Tokens(tokens), {},
            [{"role": "user", "content": "salut"}], True, None,
        )
    )
    return scheduler


def test_chat_turn_skips_punctuation_only_chunks():
    # The sentence splitter leaves the trailing "." in the buffer, and a sentence
    # released earlier empties the rest: a lone "." must not become a TTS call.
    long_second = "Deuxieme phrase tres longue qui depasse largement le seuil de regroupement, " * 3
    scheduler = _run_turn([
        "Premiere phrase assez longue pour etre liberee tout de suite. ",
        long_second.strip() + ". ",
        ".",
    ])
    texts = scheduler.texts
    assert texts
    assert all(any(ch.isalnum() for ch in t) for t in texts), texts
    assert texts[0] == "Premiere phrase assez longue pour etre liberee tout de suite."


def _stream(reply: str):
    """Feed the text one character at a time, like the LLM token stream."""
    buf = ""
    out = []
    for ch in reply:
        buf += ch
        chunks, buf = _pop_chat_chunks(buf)
        out += chunks
    if buf.strip():
        out.append(buf.strip())
    return out


def test_pop_chat_chunks_sentence_boundary():
    chunks, rest = _pop_chat_chunks(
        "This opening sentence is definitely long enough. How are you"
    )
    assert chunks == ["This opening sentence is definitely long enough."]
    assert rest == "How are you"


def test_pop_chat_chunks_merges_short_sentences():
    # A tiny first sentence ("Bonjour !") must not be released alone, otherwise
    # playback drains before the next chunk is ready (audible pause).
    chunks = _stream("Bonjour ! Je vais tres bien, merci de demander.")
    assert chunks == ["Bonjour ! Je vais tres bien, merci de demander."]


def test_pop_chat_chunks_keeps_incomplete_text():
    text = "x" * 200  # no boundary to cut on
    chunks, rest = _pop_chat_chunks(text)
    assert chunks == []
    assert rest == text


def test_pop_chat_chunks_splits_long_text_on_spaces():
    text = ("word " * 60).strip()
    chunks, rest = _pop_chat_chunks(text, max_chars=50)
    assert chunks
    assert all(len(c) <= 52 for c in chunks)
    assert rest and text.startswith(chunks[0])


def test_pop_chat_chunks_releases_complete_sentences_only():
    # A clause — let alone a single word — is never released on its own: the TTS
    # engine renders each fragment as a standalone utterance with its own
    # intonation, which is heard as chopped, word-by-word speech.
    text = "This is a considerably longer opening clause, followed by more text"
    chunks, rest = _pop_chat_chunks(text)
    assert chunks == []
    assert rest == text


def test_pop_chat_chunks_releases_a_finished_sentence():
    text = "This is a considerably longer opening clause, followed by more text. And more"
    chunks, rest = _pop_chat_chunks(text)
    assert chunks == ["This is a considerably longer opening clause, followed by more text."]
    assert rest == "And more"


def test_pop_chat_chunks_cuts_a_very_long_unfinished_sentence():
    # No punctuation at all: audio must not stall forever, so the buffer is cut at
    # the last clause/space boundary past _CHAT_MAX_CHARS (last resort).
    text = ("mot " * 200).strip()
    chunks, rest = _pop_chat_chunks(text)
    assert chunks
    assert len(chunks[0]) <= _CHAT_MAX_CHARS + 2
    assert rest and text.startswith(chunks[0])


def test_pop_chat_chunks_coalesces_after_the_first_chunk():
    # The caller raises `min_chars` once the first sentence is out: the rest of the
    # reply is grouped into fewer, longer utterances instead of one call per
    # sentence.
    text = "Une premiere phrase assez longue pour etre liberee. Deuxieme phrase. Troisieme."
    first, rest = _pop_chat_chunks(text, min_chars=_CHAT_MIN_CHARS)
    assert first == ["Une premiere phrase assez longue pour etre liberee."]
    grouped, rest2 = _pop_chat_chunks(rest, min_chars=_CHAT_COALESCE_CHARS)
    assert grouped == []  # below the target, kept until the next sentence arrives
    assert rest2 == rest


def test_pop_chat_chunks_first_clause_starts_the_audio_early():
    # A long opening sentence may be cut at a clause for the first chunk (natural
    # prosodic break) so the audio does not wait for the whole sentence.
    head = "This opening sentence really keeps going on with a first clause that is fairly long indeed"
    text = f"{head}, and then it continues for a while without ever ending"
    chunks, rest = _pop_chat_chunks(text, allow_first_clause=True)
    assert chunks == [f"{head},"]
    assert rest == "and then it continues for a while without ever ending"
    # Without the flag, no clause fragment is released on its own.
    assert _pop_chat_chunks(text)[0] == []


def test_pop_chat_chunks_keeps_short_sentence():
    chunks, rest = _pop_chat_chunks("Hi, ok")
    assert chunks == []
    assert rest == "Hi, ok"


def test_streamed_reply_is_never_cut_mid_sentence():
    reply = "Bonjour ! Je vais tres bien, merci de demander. Et toi, comment vas-tu ?"
    chunks = _stream(reply)
    assert chunks
    assert all(c[-1] in ".!?…" for c in chunks), chunks
    assert " ".join(chunks) == reply

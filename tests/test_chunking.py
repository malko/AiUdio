import numpy as np

from app.chunking import (
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    chunk_text,
    concat_audio,
    split_sentences,
)


def test_split_sentences_basic():
    assert split_sentences("Bonjour. Comment ça va ? Très bien!") == [
        "Bonjour.",
        "Comment ça va ?",
        "Très bien!",
    ]


def test_split_sentences_abbreviations():
    out = split_sentences("Bonjour M. Dupont. Au revoir.")
    assert out == ["Bonjour M. Dupont.", "Au revoir."]


def test_chunk_text_respects_max_chars():
    text = " ".join(f"Phrase numéro {i}." for i in range(20))
    chunks = chunk_text(text, max_chars=60)
    assert all(len(c) <= 60 for c in chunks)
    assert "".join(chunks).count("Phrase") == 20


def test_chunk_text_splits_long_sentence():
    long_sentence = "mots " * 100
    chunks = chunk_text(long_sentence.strip(), max_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_chunk_text_clamps_max_chars():
    """A chunk must stay inside what one generation can produce (40..600)."""
    text = " ".join(f"Phrase numéro {i}." for i in range(200))
    chunks = chunk_text(text, max_chars=100_000)
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert all(len(c) >= 1 for c in chunks)

    tiny = chunk_text(text, max_chars=1)
    assert all(len(c) <= MIN_CHUNK_CHARS for c in tiny)
    assert min(len(c) for c in tiny) > 0


def test_chunk_never_cuts_inside_a_word():
    """No sentence punctuation: the cut still lands between words."""
    text = " ".join(f"mot{i}" for i in range(400))
    chunks = chunk_text(text, max_chars=60)
    assert all(len(c) <= 60 for c in chunks)
    # Every word is intact and in order (no chunk starts/ends mid-word).
    assert " ".join(chunks).split() == text.split()


def test_chunk_cuts_cjk_on_punctuation():
    """Chinese has no spaces: sentence/clause punctuation drives the split."""
    text = "这是一个测试。" * 40 + "这是第二部分，还有很多内容。" * 20
    chunks = chunk_text(text, max_chars=60)
    assert all(len(c) <= 60 for c in chunks)
    assert "".join(chunks) == text
    # Every chunk but the last ends on punctuation, never mid-sentence.
    assert all(c[-1] in "。，" for c in chunks[:-1])


def test_chunk_cuts_on_line_breaks():
    """Lyrics/subtitles: line breaks are boundaries even without punctuation."""
    lines = [f"line {i} with a few words" for i in range(120)]
    text = "\n".join(lines)
    chunks = chunk_text(text, max_chars=60)
    assert all(len(c) <= 60 for c in chunks)
    # Whole lines only, in order, nothing lost.
    assert [line for c in chunks for line in c.split("\n")] == lines


def test_chunk_giant_token_is_the_only_mid_word_cut():
    """A single token longer than the budget has to be cut (documented)."""
    text = "x" * 200
    chunks = chunk_text(text, max_chars=60)
    assert "".join(chunks) == text
    assert all(len(c) <= 60 for c in chunks)


def test_concat_audio_inserts_silence():
    a = np.ones(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    out = concat_audio([a, b], sample_rate=1000, silence_ms=10)
    assert len(out) == 10 + 10 + 10

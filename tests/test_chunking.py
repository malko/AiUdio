import numpy as np

from app.chunking import chunk_text, concat_audio, split_sentences


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


def test_concat_audio_inserts_silence():
    a = np.ones(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    out = concat_audio([a, b], sample_rate=1000, silence_ms=10)
    assert len(out) == 10 + 10 + 10

from app.main import _pop_chat_chunks


def test_pop_chat_chunks_sentence_boundary():
    chunks, rest = _pop_chat_chunks("Hello there. How are you")
    assert chunks == ["Hello there."]
    assert rest == "How are you"


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

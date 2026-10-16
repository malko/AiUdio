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


def test_pop_chat_chunks_releases_clause_past_min():
    text = "This is a considerably longer opening clause, followed by more text."
    chunks, rest = _pop_chat_chunks(text)
    assert len(chunks) == 1
    assert chunks[0].endswith("clause,")
    assert rest == "followed by more text."


def test_pop_chat_chunks_keeps_short_sentence():
    chunks, rest = _pop_chat_chunks("Hi, ok")
    assert chunks == []
    assert rest == "Hi, ok"

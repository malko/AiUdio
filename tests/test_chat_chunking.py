from app.main import _pop_chat_chunks


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


def test_pop_chat_chunks_releases_clause_past_min():
    text = "This is a considerably longer opening clause, followed by more text."
    chunks, rest = _pop_chat_chunks(text)
    assert len(chunks) == 1
    assert chunks[0].endswith("clause,")
    assert rest == "followed by more text."


def test_pop_chat_chunks_releases_at_word_without_punctuation():
    # A run-on opening with no comma must still let TTS start early (word boundary).
    text = "This opening sentence has no punctuation at all and keeps going for a while"
    chunks = _stream(text)
    assert chunks
    assert len(chunks[0]) >= 40
    assert chunks[0].split()[-1] in text.split()


def test_pop_chat_chunks_keeps_short_sentence():
    chunks, rest = _pop_chat_chunks("Hi, ok")
    assert chunks == []
    assert rest == "Hi, ok"


def test_pop_chat_chunks_first_chunk_can_be_shorter():
    # A smaller first-chunk threshold starts the audio earlier.
    text = "Yes indeed, this is a longer opening sentence that keeps going"
    chunks, _ = _pop_chat_chunks(text, min_chars=40, first_min_chars=10)
    assert chunks
    assert len(chunks[0]) < 40


def test_pop_chat_chunks_first_min_falls_back_to_min():
    text = "Hi there. A second sentence that is long enough to be released."
    with_first, _ = _pop_chat_chunks(text, first_min_chars=0)
    without, _ = _pop_chat_chunks(text)
    assert with_first == without

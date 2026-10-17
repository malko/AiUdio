"""LLM streaming helper: stop at the end of the generation, not at the socket close."""

import asyncio

from app.llm import _content_deltas


def _collect(lines):
    async def run():
        return [content async for content in _content_deltas(lines())]

    return asyncio.run(run())


def test_content_deltas_yields_the_content():
    consumed = []

    async def lines():
        for line in (
            'data: {"choices":[{"delta":{"content":"Bon"},"finish_reason":null}]}',
            "",
            "data: not json",
            'data: {"choices":[]}',
            'data: {"choices":[{"delta":{"content":"jour"},"finish_reason":null}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"tail"}}]}',
        ):
            consumed.append(line)
            yield line

    assert _collect(lines) == ["Bon", "jour"]
    # The [DONE] sentinel ends the stream: the trailing chunk is never read.
    assert len(consumed) == 6


def test_content_deltas_stops_at_finish_reason():
    # llama.cpp can keep the connection open ~2 s after the last token: the
    # generation is over as soon as a finish_reason is reported.
    consumed = []

    async def lines():
        for line in (
            'data: {"choices":[{"delta":{"content":"Bonjour"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: {"choices":[{"delta":{"content":"late"}}]}',
        ):
            consumed.append(line)
            yield line
        consumed.append("socket closed")

    assert _collect(lines) == ["Bonjour"]
    assert "socket closed" not in consumed

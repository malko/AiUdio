"""Micro-batching: grouping must never mix two clone references.

The engine synthesizes a whole batch from ``items[0]``'s reference clip, so the
scheduler has to keep items with different references in separate calls.
"""

import asyncio

import numpy as np

from app.scheduler import BatchScheduler, _batch_key


class _Engine:
    def __init__(self):
        self.calls = []

    def synthesize_batch(self, family, items):
        self.calls.append((family, [dict(i) for i in items]))
        return [(np.zeros(8, dtype=np.float32), 24000) for _ in items]


def _item(ref, language="Auto", **extra):
    return {"text": "hi", "ref_audio": ref, "language": language, **extra}


def test_batch_key_separates_clone_references():
    key = _batch_key
    # Non-clone families still batch purely by family.
    assert key("custom_voice", {"text": "x"}) == key("custom_voice", {"text": "y"})
    assert key("voice_design", {"instruct": "a"}) == key("voice_design", {"instruct": "b"})
    # Clone families: different reference, language or x-vector mode => different batch.
    assert key("base", _item("a.wav")) != key("base", _item("b.wav"))
    assert key("base", _item("a.wav")) != key("base", _item("a.wav", "French"))
    assert key("chatterbox", _item("a.wav")) != key(
        "chatterbox", _item("a.wav", x_vector_only_mode=True)
    )


def test_different_references_are_not_batched():
    async def run():
        engine = _Engine()
        sched = BatchScheduler(engine, batch_size=4, window_ms=20)
        await sched.start()
        try:
            results = await asyncio.gather(
                sched.submit("base", _item("a.wav")),
                sched.submit("base", _item("b.wav")),
            )
        finally:
            await sched.stop()
        return engine, results

    engine, results = asyncio.run(run())
    assert len(results) == 2
    assert len(engine.calls) == 2
    assert all(len(items) == 1 for _family, items in engine.calls)
    assert {items[0]["ref_audio"] for _family, items in engine.calls} == {"a.wav", "b.wav"}


def test_same_reference_is_still_batched():
    async def run():
        engine = _Engine()
        sched = BatchScheduler(engine, batch_size=4, window_ms=20)
        await sched.start()
        try:
            await asyncio.gather(
                sched.submit("base", _item("a.wav")),
                sched.submit("base", _item("a.wav")),
            )
        finally:
            await sched.stop()
        return engine

    engine = asyncio.run(run())
    assert len(engine.calls) == 1
    assert len(engine.calls[0][1]) == 2

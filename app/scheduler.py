"""Asynchronous micro-batching: groups requests by family to maximize throughput.

A single task drains the queue, collects up to `batch_size` items within a
`batch_window_ms` window, groups them by model family, then runs one `generate_*`
call per group (in a worker thread). Each caller receives its result via a
`Future`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .metrics import metrics

log = logging.getLogger(__name__)


def note_cuda_oom(exc: BaseException, family: str, items: List[Dict[str, Any]]) -> None:
    """Count and explain a CUDA OOM raised by a synthesis call.

    On a shared GPU this is usually *not* our process growing: another workload
    (a vLLM service, a model left loaded elsewhere) holds the memory. A plain
    traceback is not enough to tell that apart, hence the explicit log.
    """
    if "out of memory" not in str(exc).lower():
        return
    metrics.incr("tts_cuda_oom")
    log.error(
        "CUDA out of memory while synthesizing '%s' (n=%d, %d chars): another process may "
        "be holding the GPU. Free VRAM, unload the local models (POST /api/models/unload) "
        "or lower the vLLM-Omni reservation.",
        family, len(items), sum(len(i.get("text") or "") for i in items),
    )


@dataclass
class _Queued:
    family: str
    item: Dict[str, Any]
    future: asyncio.Future


class BatchScheduler:
    def __init__(self, engine, batch_size: int = 4, window_ms: int = 25):
        self.engine = engine
        self.batch_size = max(1, batch_size)
        self.window = max(0, window_ms) / 1000.0
        self._queue: asyncio.Queue[_Queued] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, family: str, item: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(_Queued(family=family, item=item, future=future))
        return await future

    async def submit_streaming(self, family: str, item: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """Run immediately, bypassing the micro-batch window.

        Live sessions care about time-to-first-audio, not throughput: waiting up
        to ``batch_window_ms`` (and sharing a batch with unrelated callers) only
        adds jitter. The engine lock still serializes inference.
        """
        try:
            results = await asyncio.to_thread(self.engine.synthesize_batch, family, [item])
        except RuntimeError as exc:
            note_cuda_oom(exc, family, [item])
            raise
        return results[0]

    def depth(self) -> int:
        return self._queue.qsize()

    async def _worker(self) -> None:
        while True:
            first = await self._queue.get()
            batch: List[_Queued] = [first]
            deadline = self.window
            while len(batch) < self.batch_size:
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=deadline)
                except asyncio.TimeoutError:
                    break
                batch.append(nxt)
                deadline = 0.0  # once started, drain whatever is already queued
            await asyncio.to_thread(self._run_batch, batch)

    def _run_batch(self, batch: List[_Queued]) -> None:
        groups: Dict[str, List[_Queued]] = {}
        for entry in batch:
            groups.setdefault(entry.family, []).append(entry)

        for family, entries in groups.items():
            items = [e.item for e in entries]
            try:
                results = self.engine.synthesize_batch(family, items)
                for entry, result in zip(entries, results, strict=False):
                    if not entry.future.done():
                        entry.future.set_result(result)
            except Exception as exc:  # propagate the error to every caller
                note_cuda_oom(exc, family, items)
                log.exception("Generation failed (family=%s, n=%d)", family, len(entries))
                for entry in entries:
                    if not entry.future.done():
                        entry.future.set_exception(exc)

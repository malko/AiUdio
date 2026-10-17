"""Lightweight in-process metrics.

Enough to observe the real-time profile (time-to-first-audio, real-time factor,
request counts) without a metrics backend. Exposed by ``GET /api/metrics``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Optional


def _avg(total: float, count: int) -> Optional[float]:
    return round(total / count, 4) if count else None


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.time()
        self.counters: Dict[str, int] = defaultdict(int)
        self._ttfa_count = 0
        self._ttfa_sum = 0.0
        self._ttfa_min: Optional[float] = None
        self._ttfa_max = 0.0
        self._rtf_count = 0
        self._rtf_sum = 0.0

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe_ttfa(self, seconds: float) -> None:
        with self._lock:
            self._ttfa_count += 1
            self._ttfa_sum += seconds
            self._ttfa_min = seconds if self._ttfa_min is None else min(self._ttfa_min, seconds)
            self._ttfa_max = max(self._ttfa_max, seconds)

    def observe_rtf(self, ratio: float) -> None:
        with self._lock:
            self._rtf_count += 1
            self._rtf_sum += ratio

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self.started, 1),
                "counters": dict(self.counters),
                "ttfa_seconds": {
                    "count": self._ttfa_count,
                    "avg": _avg(self._ttfa_sum, self._ttfa_count),
                    "min": round(self._ttfa_min, 4) if self._ttfa_min is not None else None,
                    "max": round(self._ttfa_max, 4) if self._ttfa_count else None,
                },
                "rtf": {
                    "count": self._rtf_count,
                    "avg": _avg(self._rtf_sum, self._rtf_count),
                },
            }


metrics = Metrics()

"""Utility functions for measuring performance metrics."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - pynvml may not be installed
    pynvml = None  # type: ignore


@dataclass
class RunResult:
    latencies: List[float]
    tokens: int | None = None
    samples: int | None = None
    peak_vram_mb: float | None = None
    steady_vram_mb: float | None = None

    @property
    def throughput(self) -> float | None:
        if self.tokens is not None:
            total_tokens = self.tokens * len(self.latencies)
            total_time = sum(self.latencies)
            return total_tokens / total_time if total_time else None
        if self.samples is not None:
            total_samples = self.samples * len(self.latencies)
            total_time = sum(self.latencies)
            return total_samples / total_time if total_time else None
        return None

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies)

    @property
    def p95(self) -> float:
        return statistics.quantiles(self.latencies, n=20)[18]


class GPUMemoryTracker:
    """Simple GPU memory sampler using NVML.

    The tracker samples total memory usage for device 0 every ``interval``
    seconds.  If ``pynvml`` is not available the tracker silently does nothing
    and all reported metrics are ``None``.
    """

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self._samples: List[int] = []
        self._running = False

    def __enter__(self) -> "GPUMemoryTracker":
        if pynvml is None:
            return self
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._running = True
        self._samples.clear()
        return self

    def sample(self) -> None:
        if not self._running:
            return
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self._samples.append(info.used)

    def __exit__(self, exc_type, exc, tb) -> None:
        if pynvml is None:
            return
        if self._running:
            pynvml.nvmlShutdown()
        self._running = False

    @property
    def peak(self) -> float | None:
        if not self._samples:
            return None
        return max(self._samples) / (1024 ** 2)

    @property
    def steady(self) -> float | None:
        if len(self._samples) < 5:
            return self.peak
        return statistics.mean(self._samples[-5:]) / (1024 ** 2)


def timeit(fn: Callable[[], None], iters: int) -> List[float]:
    """Return list of latencies (seconds) for ``fn`` executed ``iters`` times."""
    times: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def summarize(latencies: Iterable[float], tokens: int | None = None, samples: int | None = None,
              tracker: GPUMemoryTracker | None = None) -> RunResult:
    result = RunResult(list(latencies), tokens=tokens, samples=samples)
    if tracker is not None:
        result.peak_vram_mb = tracker.peak
        result.steady_vram_mb = tracker.steady
    return result

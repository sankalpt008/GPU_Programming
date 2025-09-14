"""Backend helper classes for different inference engines."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .metrics import GPUMemoryTracker, RunResult, summarize, timeit

try:  # optional imports
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - dependencies not installed
    np = None  # type: ignore
    ort = None  # type: ignore

# vLLM and TensorRT are optional and may not be present; we only import when used.


@dataclass
class BenchmarkConfig:
    seq_len: int
    batch: int
    gen_tokens: int = 0  # for generation models
    warmup: int = 0
    iters: int = 1
    fp16: bool = False


class BaseEngine:
    backend: str

    def run(self, cfg: BenchmarkConfig) -> RunResult:  # pragma: no cover - interface
        raise NotImplementedError


class ORTEncoder(BaseEngine):
    """Very small wrapper around ONNX Runtime for encoder-style models."""

    backend = "ort"

    def __init__(self, onnx_path: str, provider: str = "CUDAExecutionProvider") -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is not available")
        self.session = ort.InferenceSession(onnx_path, providers=[provider])

    def run(self, cfg: BenchmarkConfig) -> RunResult:
        dummy_inputs = {
            "input_ids": [[0] * cfg.seq_len for _ in range(cfg.batch)],
            "attention_mask": [[1] * cfg.seq_len for _ in range(cfg.batch)],
        }

        def _infer() -> None:
            self.session.run(None, dummy_inputs)

        with GPUMemoryTracker() as tracker:
            latencies = timeit(_infer, cfg.iters)
            result = summarize(latencies, samples=cfg.batch, tracker=tracker)
        return result


class ORTDecoder(BaseEngine):
    """Simplified decoder running one forward pass per token."""

    backend = "ort"

    def __init__(self, onnx_path: str, provider: str = "CUDAExecutionProvider") -> None:
        if ort is None or np is None:
            raise RuntimeError("onnxruntime or numpy is not available")
        self.session = ort.InferenceSession(onnx_path, providers=[provider])

    def run(self, cfg: BenchmarkConfig) -> RunResult:
        input_ids = np.zeros((cfg.batch, cfg.seq_len), dtype=np.int64)

        def _loop() -> None:
            for _ in range(cfg.gen_tokens):
                self.session.run(None, {"input_ids": input_ids})

        with GPUMemoryTracker() as tracker:
            latencies = timeit(_loop, cfg.iters)
            tokens = cfg.batch * cfg.gen_tokens
            result = summarize(latencies, tokens=tokens, tracker=tracker)
        return result


class DummyEngine(BaseEngine):
    """Fallback engine used when real backends are unavailable.

    The engine simply sleeps for a tiny amount of time to emulate work.
    """

    backend = "dummy"

    def run(self, cfg: BenchmarkConfig) -> RunResult:
        def _noop() -> None:
            time.sleep(0.001)

        latencies = timeit(_noop, cfg.iters)
        return summarize(latencies, samples=cfg.batch)


# Factory utilities ---------------------------------------------------------

def create_gpt2_engine(backend: str, onnx_path: str | None = None) -> BaseEngine:
    backend = backend.lower()
    if backend.startswith("ort"):
        provider = "CUDAExecutionProvider" if backend == "ort_cuda" else "TensorrtExecutionProvider"
        if onnx_path is None:
            raise ValueError("onnx_path required for ONNX Runtime backends")
        return ORTDecoder(onnx_path, provider)
    try:
        if backend == "vllm":
            from vllm import LLM  # type: ignore

            class VLLMEngine(BaseEngine):
                backend = "vllm"

                def __init__(self) -> None:
                    self.llm = LLM("gpt2-medium")

                def run(self, cfg: BenchmarkConfig) -> RunResult:
                    prompts = ["hello"] * cfg.batch
                    outputs = self.llm.generate(prompts, max_tokens=cfg.gen_tokens)

                    def _infer() -> None:
                        self.llm.generate(prompts, max_tokens=cfg.gen_tokens)

                    with GPUMemoryTracker() as tracker:
                        latencies = timeit(_infer, cfg.iters)
                        tokens = cfg.batch * cfg.gen_tokens
                        return summarize(latencies, tokens=tokens, tracker=tracker)

            return VLLMEngine()
    except Exception:  # pragma: no cover - vLLM optional
        pass
    return DummyEngine()


def create_bert_engine(backend: str, onnx_path: str | None = None, trt_engine: str | None = None) -> BaseEngine:
    backend = backend.lower()
    if backend.startswith("ort"):
        provider = "CUDAExecutionProvider" if backend == "ort_cuda" else "TensorrtExecutionProvider"
        if onnx_path is None:
            raise ValueError("onnx_path required for ONNX Runtime backends")
        return ORTEncoder(onnx_path, provider)
    # TensorRT backend would be implemented here. For environments without TensorRT
    # we fall back to a dummy engine.
    return DummyEngine()

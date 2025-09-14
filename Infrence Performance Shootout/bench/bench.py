"""CLI entry point for running benchmarks.

This module intentionally keeps the actual benchmarking logic minimal so that
it can run even on machines without the heavy dependencies installed.  When a
backend is not available a lightweight dummy engine is used instead.  The goal
is to provide a reproducible harness that users can extend with the real
inference libraries.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .engines import BenchmarkConfig, create_bert_engine, create_gpt2_engine
from .metrics import RunResult


def run_and_average(run_fn: Callable[[], RunResult], repeats: int) -> RunResult:
    runs = [run_fn() for _ in range(repeats)]
    latencies = [lat for r in runs for lat in r.latencies]
    tokens = runs[0].tokens
    samples = runs[0].samples
    peak = max(r.peak_vram_mb or 0 for r in runs)
    steady = sum(r.steady_vram_mb or 0 for r in runs) / repeats
    return RunResult(latencies, tokens=tokens, samples=samples,
                     peak_vram_mb=peak, steady_vram_mb=steady)


def save_results(result: RunResult, args: argparse.Namespace, model: str) -> None:
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "backend",
            "model",
            "batch",
            "seq_len",
            "throughput",
            "p50",
            "p95",
            "VRAM_peak_MB",
            "VRAM_steady_MB",
            "notes",
        ])
        writer.writerow([
            args.backend,
            model,
            args.batch,
            args.seq_len,
            result.throughput,
            result.p50,
            result.p95,
            result.peak_vram_mb,
            result.steady_vram_mb,
            "fp16" if args.fp16 else "fp32",
        ])
    # JSON sidecar
    json_path = path.with_suffix(".json")
    with json_path.open("w") as jf:
        json.dump(asdict(result), jf, indent=2)


def benchmark_gpt2(args: argparse.Namespace) -> RunResult:
    engine = create_gpt2_engine(args.backend, args.onnx)
    cfg = BenchmarkConfig(
        seq_len=args.seq_len,
        batch=args.batch,
        gen_tokens=args.gen_tokens,
        warmup=args.warmup,
        iters=args.iters,
        fp16=args.fp16,
    )
    return run_and_average(lambda: engine.run(cfg), args.repeat)


def benchmark_bert(args: argparse.Namespace) -> RunResult:
    engine = create_bert_engine(args.backend, args.onnx, args.trt_engine)
    cfg = BenchmarkConfig(
        seq_len=args.seq_len,
        batch=args.batch,
        warmup=args.warmup,
        iters=args.iters,
        fp16=args.fp16,
    )
    return run_and_average(lambda: engine.run(cfg), args.repeat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark harness")
    sub = parser.add_subparsers(dest="model", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", required=True)
    common.add_argument("--seq-len", type=int, required=True)
    common.add_argument("--batch", type=int, required=True)
    common.add_argument("--warmup", type=int, default=0)
    common.add_argument("--iters", type=int, default=1)
    common.add_argument("--repeat", type=int, default=3, help="Number of runs to average")
    common.add_argument("--fp16", action="store_true")
    common.add_argument("--csv", required=True, help="Where to store CSV results")
    common.add_argument("--onnx", help="Path to ONNX model if required")
    common.add_argument("--trt-engine", help="Path to TensorRT engine if required")

    gpt2 = sub.add_parser("gpt2", parents=[common])
    gpt2.add_argument("--gen-tokens", type=int, required=True)

    sub.add_parser("bert", parents=[common])

    args = parser.parse_args()
    if args.model == "gpt2":
        result = benchmark_gpt2(args)
    else:
        result = benchmark_bert(args)

    save_results(result, args, args.model)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

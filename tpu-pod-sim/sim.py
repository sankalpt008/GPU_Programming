"""Simulate a 64-chip TPU v5e pod and run calibration and batch sweeps.

Usage:
  python sim.py run --arrival_rps 30 --batch_size 256 --seq_len 32 --sim_time 60
  python sim.py sweep --arrival_rps 30 --seq_len 32 --batches 128,192,256,320
  python sim.py calibrate --counters data.csv --out params.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt

# Base throughput assumptions (tokens/s) for one scaler=1.0.
BASE_COMPUTE_TPS = 1_000_000.0
BASE_HBM_TPS = 2_000_000.0
BASE_FABRIC_TPS = 1_000_000.0
CHIPS = 64


@dataclass
class Scalers:
    """Service time multipliers for different subsystems."""

    compute: float = 1.0
    hbm: float = 1.0
    fabric: float = 1.0

    @classmethod
    def from_json(cls, path: Path | None) -> "Scalers":
        if path and path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return cls()

    def to_dict(self) -> Dict[str, float]:
        return {"compute": self.compute, "hbm": self.hbm, "fabric": self.fabric}


class PodModel:
    """SimPy model of a TPU v5e pod."""

    def __init__(self, env: simpy.Environment, scalers: Scalers) -> None:
        self.env = env
        self.scalers = scalers
        self.compute = simpy.Resource(env, capacity=CHIPS)
        self.hbm = simpy.Resource(env, capacity=CHIPS)
        self.fabric = simpy.Resource(env, capacity=CHIPS)
        self.latencies: List[float] = []
        self.completed = 0
        self.completed_tokens = 0

    def generator(
        self,
        arrival_rps: float,
        batch_size: int,
        seq_len: int,
        burstiness: float,
    ):
        i = 0
        while True:
            wait = random.expovariate(arrival_rps) * burstiness
            yield self.env.timeout(wait)
            self.env.process(self.request(i, batch_size, seq_len))
            i += 1

    def request(self, req_id: int, batch_size: int, seq_len: int):
        start = self.env.now
        tokens = batch_size * seq_len
        with self.compute.request() as r:
            yield r
            t = tokens / (BASE_COMPUTE_TPS * CHIPS)
            yield self.env.timeout(t * self.scalers.compute)
        with self.hbm.request() as r:
            yield r
            t = tokens / (BASE_HBM_TPS * CHIPS)
            yield self.env.timeout(t * self.scalers.hbm)
        with self.fabric.request() as r:
            yield r
            t = tokens / (BASE_FABRIC_TPS * CHIPS)
            yield self.env.timeout(t * self.scalers.fabric)
        latency = self.env.now - start
        self.latencies.append(latency)
        self.completed += 1
        self.completed_tokens += tokens


def simulate_workload(
    arrival_rps: float,
    batch_size: int,
    seq_len: int,
    sim_time: float,
    burstiness: float,
    scalers: Scalers,
) -> Dict[str, float]:
    env = simpy.Environment()
    model = PodModel(env, scalers)
    env.process(model.generator(arrival_rps, batch_size, seq_len, burstiness))
    env.run(until=sim_time)
    throughput = model.completed_tokens / sim_time
    p95 = float(np.percentile(model.latencies, 95)) if model.latencies else 0.0
    return {
        "throughput_tokens_per_s": throughput,
        "p95_s": p95,
        "completed": model.completed,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_error(df: pd.DataFrame, scalers: Scalers, sim_time: float) -> float:
    err = 0.0
    for _, row in df.iterrows():
        pred = simulate_workload(
            row["arrival_rps"],
            int(row["batch_size"]),
            int(row["seq_len"]),
            sim_time,
            row.get("burstiness", 1.0),
            scalers,
        )
        err += (pred["throughput_tokens_per_s"] - row["measured_tokens_per_s"]) ** 2
        err += (pred["p95_s"] - row["measured_p95_s"]) ** 2
    return err


def random_search_calibration(df: pd.DataFrame, sim_time: float, n: int = 200) -> Scalers:
    best = (float("inf"), Scalers())
    for _ in range(n):
        cand = Scalers(
            compute=random.uniform(0.1, 3.0),
            hbm=random.uniform(0.1, 3.0),
            fabric=random.uniform(0.1, 3.0),
        )
        e = calibration_error(df, cand, sim_time)
        if e < best[0]:
            best = (e, cand)
    return best[1]


def sweep_worker(args: Tuple[float, int, float, float, Dict[str, float], int]) -> Tuple[int, Dict[str, float]]:
    """Helper for multiprocessing pool in batch sweep."""
    arrival_rps, seq_len, sim_time, burstiness, scalers_dict, bs = args
    scalers = Scalers(**scalers_dict)
    res = simulate_workload(arrival_rps, bs, seq_len, sim_time, burstiness, scalers)
    return bs, res


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    scalers = Scalers.from_json(Path(args.params) if args.params else None)
    res = simulate_workload(
        args.arrival_rps,
        args.batch_size,
        args.seq_len,
        args.sim_time,
        args.burstiness,
        scalers,
    )
    print(res)


def cmd_sweep(args: argparse.Namespace) -> None:
    scalers = Scalers.from_json(Path(args.params) if args.params else None)
    batches = [int(b) for b in args.batches.split(",")]

    scalers_dict = scalers.to_dict()
    inputs = [
        (args.arrival_rps, args.seq_len, args.sim_time, args.burstiness, scalers_dict, bs)
        for bs in batches
    ]

    with mp.Pool(processes=min(len(inputs), mp.cpu_count())) as pool:
        outputs = pool.map(sweep_worker, inputs)

    records = [
        {
            "batch_size": bs,
            "throughput_tokens_per_s": r["throughput_tokens_per_s"],
            "p95_s": r["p95_s"],
            "completed": r["completed"],
        }
        for bs, r in outputs
    ]
    df = pd.DataFrame.from_records(records)
    df.to_csv("results.csv", index=False)

    plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(df["batch_size"], df["throughput_tokens_per_s"], "-o", label="tokens/s")
    ax2.plot(df["batch_size"], df["p95_s"], "-s", color="orange", label="p95 latency")
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("tokens/s")
    ax2.set_ylabel("p95 latency (s)")
    ax1.grid(True)
    plt.title("Batch sweep")
    plt.savefig("sweep.png", bbox_inches="tight")

    baseline = df.iloc[0]
    eligible = df[df["p95_s"] <= baseline["p95_s"] + 1e-9]
    best = eligible.iloc[eligible["throughput_tokens_per_s"].idxmax()]
    if best["batch_size"] == baseline["batch_size"]:
        print(f"Recommended batch size: {int(baseline['batch_size'])} (no change)")
    else:
        delta_tps = best["throughput_tokens_per_s"] - baseline["throughput_tokens_per_s"]
        delta_p95 = best["p95_s"] - baseline["p95_s"]
        print(
            f"Recommended batch size: {int(baseline['batch_size'])}→{int(best['batch_size'])}"
            f" (Δtokens/s {delta_tps:.1f}, Δp95 {delta_p95:+.3f}s)"
        )


def cmd_calibrate(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.counters)
    scalers = random_search_calibration(df, sim_time=args.sim_time)
    Path(args.out).write_text(json.dumps(scalers.to_dict(), indent=2))
    print(json.dumps(scalers.to_dict()))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--arrival_rps", type=float, required=True)
    common.add_argument("--seq_len", type=int, required=True)
    common.add_argument("--sim_time", type=float, default=60)
    common.add_argument("--burstiness", type=float, default=1.0)
    common.add_argument("--params", type=str, help="JSON file of scalers")

    p_run = sub.add_parser("run", parents=[common])
    p_run.add_argument("--batch_size", type=int, required=True)
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", parents=[common])
    p_sweep.add_argument("--batches", type=str, required=True, help="comma list")
    p_sweep.set_defaults(func=cmd_sweep)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--counters", type=str, required=True)
    p_cal.add_argument("--out", type=str, default="params.json")
    p_cal.add_argument("--sim_time", type=float, default=30)
    p_cal.set_defaults(func=cmd_calibrate)

    return p


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

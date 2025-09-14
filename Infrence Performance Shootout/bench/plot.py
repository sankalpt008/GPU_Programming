"""Plotting utilities for benchmark results."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def plot_throughput(csv_files: List[Path], out: Path) -> None:
    plt.figure()
    for csvf in csv_files:
        df = pd.read_csv(csvf)
        label = csvf.stem
        plt.plot(df["seq_len"], df["throughput"], marker="o", label=label)
    plt.xlabel("Sequence length")
    plt.ylabel("Throughput")
    plt.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark CSVs")
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/plot.png"))
    args = parser.parse_args()
    plot_throughput(args.csvs, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()

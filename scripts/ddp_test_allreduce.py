#!/usr/bin/env python3
"""Minimal DDP all_reduce test."""
import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP all_reduce sanity test")
    parser.add_argument("--backend", default="nccl", help="distributed backend")
    parser.add_argument("--timeout", type=int, default=180, help="init timeout in seconds")
    args = parser.parse_args()

    dist.init_process_group(
        backend=args.backend, timeout=timedelta(seconds=args.timeout)
    )
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    tensor = torch.ones(1, device="cuda")
    dist.all_reduce(tensor)
    print(f"Rank {rank} summed value: {tensor.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

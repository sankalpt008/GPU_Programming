#!/usr/bin/env python3
"""Print CUDA/NCCL environment information."""
import os
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - best effort for missing torch
    print(f"Failed to import torch: {exc}")
    sys.exit(1)

CHECK = "\u2705"  # check mark

print(f"{CHECK} torch: {torch.__version__}")
print(f"{CHECK} torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CUDA not available. Please verify drivers and GPU access.")
    sys.exit(1)

print(f"{CHECK} CUDA version: {torch.version.cuda}")
print(f"{CHECK} NCCL version: {torch.cuda.nccl.version()}")
print(f"{CHECK} GPU: {torch.cuda.get_device_name(0)}")
print(f"{CHECK} Device count: {torch.cuda.device_count()}")

for var in ["NCCL_P2P_DISABLE", "NCCL_DEBUG", "CUDA_VISIBLE_DEVICES"]:
    print(f"{var}={os.environ.get(var)}")

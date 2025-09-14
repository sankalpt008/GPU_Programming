#!/bin/bash
set -euo pipefail

OUT_DIR="onnx/gpt2"
mkdir -p "$OUT_DIR"

python -m bench.export --model gpt2-medium --out "$OUT_DIR/gpt2.onnx" --fp16 "$@"
# Build TensorRT engine if trtexec is available
if command -v trtexec >/dev/null 2>&1; then
  trtexec --onnx="$OUT_DIR/gpt2.onnx" --saveEngine="$OUT_DIR/gpt2.engine" --fp16
else
  echo "trtexec not found; skipping TensorRT engine build" >&2
fi

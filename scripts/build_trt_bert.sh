#!/bin/bash
set -euo pipefail

OUT_DIR="onnx/bert"
mkdir -p "$OUT_DIR"

python -m bench.export --model bert-base-uncased --out "$OUT_DIR/bert.onnx" --fp16 "$@"
if command -v trtexec >/dev/null 2>&1; then
  trtexec --onnx="$OUT_DIR/bert.onnx" --saveEngine="$OUT_DIR/bert.engine" --fp16
else
  echo "trtexec not found; skipping TensorRT engine build" >&2
fi

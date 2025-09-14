#!/bin/bash
set -euo pipefail

mkdir -p results

# GPT-2 example runs
python -m bench.bench gpt2 --backend vllm --seq-len 1024 --gen-tokens 16 --batch 1 --iters 1 --csv results/gpt2_vllm.csv || true
python -m bench.bench gpt2 --backend ort_cuda --seq-len 1024 --gen-tokens 16 --batch 1 --iters 1 --onnx onnx/gpt2/gpt2.onnx --csv results/gpt2_ort.csv || true

# BERT example runs
python -m bench.bench bert --backend ort_cuda --seq-len 128 --batch 8 --iters 1 --onnx onnx/bert/bert.onnx --csv results/bert_ort.csv || true

# Generate a simple plot
python -m bench.plot results/*.csv --out results/throughput.png || true

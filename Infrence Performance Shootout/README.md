# GPU Backend Benchmark Shootout

This project provides a small, reproducible harness for comparing
[vLLM](https://github.com/vllm-project/vllm),
[TensorRT](https://developer.nvidia.com/tensorrt) and
[ONNX Runtime](https://onnxruntime.ai/) on a single NVIDIA GPU.
It targets two reference models:

* **GPT‑2 Medium** – text generation benchmark.
* **BERT Base** – sequence classification benchmark (SST‑2).

The harness measures throughput, latency and GPU memory consumption and writes
results as CSV/JSON files for later analysis.

## Installation

1. Install a PyTorch wheel with CUDA support for your system.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Optional: install TensorRT and ensure the `trtexec` utility is on the
   `PATH`.  The project can still run without TensorRT – the scripts simply skip
   those benchmarks.

## Quickstart

Export models (creates ONNX files and, if TensorRT is available, builds
TensorRT engines):

```bash
bash scripts/build_trt_gpt2.sh
bash scripts/build_trt_bert.sh
```

Run a few example benchmarks:

```bash
# GPT‑2 8k context, FP16
python -m bench.bench gpt2 --backend vllm --seq-len 8192 --gen-tokens 128 --batch 2 --warmup 20 --iters 100 --fp16 --csv results/gpt2_vllm_8k.csv
python -m bench.bench gpt2 --backend ort_trt --seq-len 8192 --gen-tokens 128 --batch 2 --warmup 20 --iters 100 --fp16 --onnx onnx/gpt2/gpt2.onnx --csv results/gpt2_orttrt_8k.csv
python -m bench.bench gpt2 --backend trt --seq-len 8192 --gen-tokens 128 --batch 2 --warmup 20 --iters 100 --fp16 --csv results/gpt2_trt_8k.csv

# BERT SST-2, FP16
python -m bench.bench bert --backend ort_cuda --seq-len 256 --batch 32 --warmup 50 --iters 500 --fp16 --onnx onnx/bert/bert.onnx --csv results/bert_ortcuda.csv
python -m bench.bench bert --backend ort_trt  --seq-len 256 --batch 32 --warmup 50 --iters 500 --fp16 --onnx onnx/bert/bert.onnx --csv results/bert_orttrt.csv
python -m bench.bench bert --backend trt      --seq-len 256 --batch 32 --warmup 50 --iters 500 --fp16 --csv results/bert_trt.csv
```

Generate plots:

```bash
python -m bench.plot results/*.csv --out results/throughput.png
```

To run every example in one shot:

```bash
bash scripts/run_all.sh
```

## Notes

* vLLM targets autoregressive LLM generation workloads and is only used for
  GPT‑2 in this project.
* ONNX Runtime provides both CUDA and TensorRT execution providers and works
  for both GPT‑2 and BERT.
* TensorRT is generally fastest but requires an engine build step and a
  matching GPU/driver.
* If TensorRT is not available the harness still functions – TensorRT specific
  runs will simply be skipped.

The repository is intentionally light‑weight; the benchmarking logic uses
small placeholder implementations when the heavy dependencies are missing.  The
structure is ready for extending with real measurement code on an RTX 4070 or
similar GPU.

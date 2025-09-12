# TPU v5e Pod Simulator

This project provides a lightweight SimPy model of a 64‑chip TPU v5e pod. It can
simulate inference workloads, fit service‑time scalers from performance counter
samples, and sweep batch sizes to recommend an optimal configuration.

## Quickstart

```bash
python -m pip install -r requirements.txt
```

### Run a single simulation

```bash
python sim.py run --arrival_rps 30 --batch_size 256 --seq_len 32 --sim_time 60
```

### Sweep batch sizes

```bash
python sim.py sweep --arrival_rps 30 --seq_len 32 --batches 128,192,256,320,384,448,512
```

This creates `results.csv` and `sweep.png` and prints the recommended batch
size that improves throughput without increasing p95 latency.

### Calibrate scalers from counters

```bash
python sim.py calibrate --counters data/counters.csv --out params.json
```

`params.json` can then be passed to `run` or `sweep` via `--params`.

## Assumptions

The model is deliberately simple and runs on a single laptop CPU. It models
compute, memory (HBM), and interconnect as sequential resources. Service times
are proportional to `batch_size * seq_len` and scaled by calibration factors.
Modify `sim.py` to refine these assumptions as needed.

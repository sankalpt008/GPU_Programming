# GPU Programming Experiments

## Multi-node over two laptops

1. **Find your LAN IP**
   - Linux: `ip addr`
   - Windows: `ipconfig`
   - WSL2: run `ip addr` inside the WSL terminal
2. **Open the firewall for port 29500**
   - Linux with UFW: `sudo ufw allow 29500`
   - Windows: allow inbound TCP for 29500 in Windows Defender Firewall
3. **Use identical code and Python environment on both machines**

### Environment check
Run on each laptop:

```bash
python scripts/env_check.py
```

### Training launcher examples
Master (Laptop A):

```bash
MASTER_ADDR=192.168.1.50 MASTER_PORT=29500 WORLD_SIZE=2 \
NODE_RANK=0 ./scripts/launch_master.sh --cmd "python train.py --mode ddp --devices 1 --config configs/tiny.yaml --mixed_precision fp16 --gradient_checkpointing --max_steps 200"
```

Worker (Laptop B):

```bash
MASTER_ADDR=192.168.1.50 MASTER_PORT=29500 WORLD_SIZE=2 \
NODE_RANK=1 ./scripts/launch_worker.sh --cmd "python train.py --mode ddp --devices 1 --config configs/tiny.yaml --mixed_precision fp16 --gradient_checkpointing --max_steps 200"
```

### DDP sanity test examples
Master:

```bash
MASTER_ADDR=192.168.1.50 MASTER_PORT=29500 WORLD_SIZE=2 NODE_RANK=0 \
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  scripts/ddp_test_allreduce.py --backend nccl
```

Worker:

```bash
MASTER_ADDR=192.168.1.50 MASTER_PORT=29500 WORLD_SIZE=2 NODE_RANK=1 \
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  scripts/ddp_test_allreduce.py --backend nccl
```

### Troubleshooting
- NCCL timeouts? try wired Ethernet, ensure both nodes can ping each other, open firewall for 29500, set `NCCL_DEBUG=INFO`.
- If still flaky on Wi-Fi/WSL2, allow fallback envs:
  ```bash
  export NCCL_IB_DISABLE=1
  export NCCL_P2P_DISABLE=1
  export GLOO_SOCKET_IFNAME=Ethernet
  export NCCL_SOCKET_IFNAME=Ethernet
  ```
- WSL2 users: enable NVIDIA CUDA on WSL and match PyTorch CUDA wheels on both nodes.


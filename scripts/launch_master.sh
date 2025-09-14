#!/usr/bin/env bash
# Master launch script for multi-node training.
# Example:
# MASTER_ADDR=192.168.1.50 MASTER_PORT=29500 WORLD_SIZE=2 NODE_RANK=0 \
#   ./scripts/launch_master.sh --cmd "python train.py ..."

set -e

MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR not set}
MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=${WORLD_SIZE:-2}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}

CMD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cmd)
      shift
      CMD="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$CMD" ]]; then
  echo "--cmd argument required"
  exit 1
fi

torchrun --nnodes="$NNODES" --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" $CMD

#!/bin/bash
# Run `python -m <module> <args>` on every GPU of the job with ONE torchrun per node, so ranks on the same node
# share an IPC namespace (NCCL's shared-memory transport hangs across separate srun tasks).
#   per_node_torchrun.sh <port> <module> [args...]
set -euo pipefail
PORT=$1; shift
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
srun --nodes=$NODES --ntasks=$NODES --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  NP=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
  echo "[launch] $(hostname) nproc=$NP CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exec '"$PY"' -m torch.distributed.run --nnodes='"$NODES"' --nproc-per-node=$NP --rdzv-backend=c10d \
    --rdzv-endpoint='"$MASTER_ADDR:$PORT"' --rdzv-id='"$SLURM_JOB_ID-$PORT"' -m "$@"' _ "$@"

#!/bin/bash
# 3B dense code model pretraining on the v0 mix: owner-sharded ZeRO inside each node (train/zero.py), data parallel across nodes.
#   sbatch --nodes=N bench/slurm/dense3b_train.sbatch            (resumes from $CKPT automatically)
# DATA is a comma-separated list of shard globs with optional :R repeat suffix (see train/data.py).
#SBATCH --job-name=dense3b-train
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --time=72:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -euo pipefail
export PYTHON=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export RESULTS=/data/run01/scxi253/code3b-train/results/${SLURM_JOB_ID}
export LOCAL_SCRATCH=/tmp/$USER/dense3b
source bench/slurm/env.sh
export NCCL_DEBUG=WARN
export PYTHONPATH=$PWD:$PWD/.deps
CKPT=${CKPT:-/data/run01/scxi253/code3b-train/ckpt}
V0=/data/run01/scxi253/code3b-data-20261003/v0/shards
DATA=${DATA:?set DATA to the v0 mix, e.g. "$V0/swallow-code/*.u16:2,$V0/stackv2-edu/*.u16"}
# 256 x 4096 = 1.05M tokens per optimizer step; STEPS=95000 -> 100B tokens
STEPS=${STEPS:-95000}
mkdir -p "$CKPT"
RESUME=$( [ -n "$(ls -A "$CKPT" 2>/dev/null)" ] && echo --resume || true )
# Measured on 4090 (2026-10-06): BF16 + compile + doc mask = 21.9 ms/layer (83% layer MFU); 0.49 GB activations per
# stored block, 0.025 GB per recomputed block -> 21 of 28 blocks recomputed fit 23 GB (planner: ~64% node MFU).
run_torchrun train.train_dense --preset dense3b --data "'$DATA'" --val-data "'${VAL:-}'" --seq-len 4096 \
  --micro-batch 1 --global-batch 256 --steps "$STEPS" --lr ${LR:-7e-4} --warmup 2000 --cooldown-frac 0.2 \
  --ckpt-layers ${CKPT_LAYERS:-21} --compile --ckpt-dir "$CKPT" --ckpt-every 500 --val-every 500 \
  --metrics "$RESULTS/metrics.jsonl" $RESUME

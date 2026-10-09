#!/bin/bash
# FP8 vs BF16 loss A/B on one GPU: dense3b width (3072, GQA 24/8, SwiGLU 8192), 8 layers (~1.06B params),
# identical seed / token stream / optimizer / schedule; the only difference is --fp8 (set ARM=fp8 or bf16).
#   sbatch --export=ALL,ARM=bf16 bench/slurm/dense_fp8_ab.sbatch ; sbatch --export=ALL,ARM=fp8 ...
#SBATCH --job-name=dense3b-ab
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export PYTHONPATH=$PWD:$PWD/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
D=/data/run01/scxi253/code3b-train/abdata
until [ -f $D/PACK.json ]; do sleep 60; done
OUT=/data/run01/scxi253/code3b-train/results/ab-${ARM}-${SLURM_JOB_ID}; mkdir -p $OUT
$PY -m train.train_dense --preset dense3b --override '{"n_layers": 8, "n_dense_layers": 8}' \
  --data "$D/train_*.u16" --val-data "$D/val_*.u16" --val-every 100 --val-windows 128 \
  --seq-len 4096 --global-batch 64 --micro-batch 1 --steps 1500 --lr 7e-4 --warmup 150 --cooldown-frac 0.2 \
  --compile --seed 0 --metrics $OUT/metrics.jsonl --log-every 20 $( [ "$ARM" = fp8 ] && echo --fp8 || true ) \
  2>&1 | tee $OUT/train.log

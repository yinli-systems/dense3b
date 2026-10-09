#!/bin/bash
# One GPU: per-layer time / activation memory sweep of dense3b blocks + memory-fit planner (bench/dense_sweep.py).
#SBATCH --job-name=dense3b-sweep
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export PYTHONPATH=$PWD:$PWD/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
OUT=/data/run01/scxi253/code3b-train/results/sweep-${SLURM_JOB_ID}
SYN=/tmp/$USER/dense3b-synth; mkdir -p "$OUT" "$SYN"
$PY -m bench.synth_data "$SYN" --shards 2 --tokens-per-shard 4000000 > /dev/null
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv > "$OUT/gpu.csv"
$PY -m pytest -q tests/dense/test_fp8_gpu.py 2>&1 | tee "$OUT/gpu_tests.txt" || true
$PY -m bench.dense_sweep --data "$SYN/synth_*.u16" ${CONFIGS:+--configs "$CONFIGS"} --out "$OUT/sweep.json" 2>&1 | tee "$OUT/sweep.log"

#!/bin/bash
# One 4090: kernel correctness, expert-path timing, and fwd+bwd throughput/memory of the 1B-active MoE candidates.
#SBATCH --job-name=moe1b-sweep
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/data/run01/scxi253/moe1b/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/moe1b-src
export PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/data/run01/scxi253/moe1b/triton-cache TORCHINDUCTOR_CACHE_DIR=/data/run01/scxi253/moe1b/inductor-cache
OUT=/data/run01/scxi253/moe1b/results/sweep-${SLURM_JOB_ID}; mkdir -p $OUT
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv > $OUT/gpu.csv
$PY -m bench.moe1b_sweep --out $OUT/sweep.json ${SWEEP_ARGS:-} 2>&1 | tee $OUT/sweep.log

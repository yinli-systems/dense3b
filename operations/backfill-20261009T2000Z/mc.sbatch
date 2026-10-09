#!/bin/bash
#SBATCH --job-name=ckpt-backfill-mc
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z/logs/mc-%A_%a.out
set -euo pipefail
C=/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z
cd "$C"
export PYTHONPATH="$C/source:/data/run01/scxi253/code3b-dense3b-src/.deps"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/scxi253-backfill-${SLURM_JOB_ID}-triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/scxi253-backfill-${SLURM_JOB_ID}-inductor"
exec /ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python -u "$C/run_mc.py" --index "$SLURM_ARRAY_TASK_ID"

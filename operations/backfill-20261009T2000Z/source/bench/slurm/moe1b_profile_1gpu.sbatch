#!/bin/bash
#SBATCH --job-name=moe1b-prof
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:40:00
#SBATCH --output=/data/run01/scxi253/moe1b/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/moe1b-src
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/data/run01/scxi253/moe1b/triton-cache TORCHINDUCTOR_CACHE_DIR=/data/run01/scxi253/moe1b/inductor-cache
OUT=/data/run01/scxi253/moe1b/results/prof-${SLURM_JOB_ID}; mkdir -p $OUT
$PY -m bench.moe1b_profile --preset ${PRESET:-moe1b} --micro ${MICRO:-2} --ckpt ${CKPT:-3} --out $OUT/prof.json ${EXTRA:-}

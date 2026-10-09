#!/bin/bash
# One GPU (5090 by default): kernel check + throughput/memory of the r3b_s preset at 4k and 16k context.
#SBATCH --job-name=r3b-bench
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/data/run01/scxi253/reason3b-data/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/reason3b-src
export PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/data/run01/scxi253/reason3b-data/triton-cache-$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -dc A-Za-z0-9 | head -c 20)
export TORCHINDUCTOR_CACHE_DIR=$TRITON_CACHE_DIR-inductor
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader); echo "GPU $GPU"
case "$GPU" in *5090*) export PEAK_TFLOPS=209.5;; *) export PEAK_TFLOPS=165.2;; esac
OUT=/data/run01/scxi253/reason3b-data/bench-${SLURM_JOB_ID}; mkdir -p $OUT
$PY -m bench.moe1b_sweep --out $OUT/kernels.json --check-only 2>&1 | tail -3
for cfg in "4096 R3S,1,0" "4096 R3S,1,6" "4096 R3S,1,12" "4096 R3S,2,12" "4096 C,1,6" "16384 R3S,1,24" "16384 R3S,1,12"; do
  set -- $cfg
  SEQ=$1 $PY -m bench.moe1b_sweep --out /dev/null --one $2 2>&1 | grep RESULT | tee -a $OUT/model.jsonl
done
echo DONE

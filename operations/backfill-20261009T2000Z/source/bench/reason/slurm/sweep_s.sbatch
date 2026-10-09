#!/bin/bash
# 5090: r3b_s throughput sweep: micro x recompute, default vs exhaustive GMM autotune, BF16 vs FP8 block Linears.
#SBATCH --job-name=r3b-sweep
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/data/run01/scxi253/reason3b-data/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/reason3b-src
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PEAK_TFLOPS=209.5
export TORCHINDUCTOR_CACHE_DIR=/data/run01/scxi253/reason3b-data/inductor-5090-sweep
run() { tag=$1; shift; for cfg in "$@"; do env $EXTRA $PY -m bench.moe1b_sweep --out /dev/null --one $cfg 2>&1 | grep RESULT | sed "s/^/$tag /"; done; }
export TRITON_CACHE_DIR=/data/run01/scxi253/reason3b-data/triton-cache-NVIDIAGeForceRTX5090
EXTRA="" run base R3S,1,0 R3S,2,0 R3S,2,3 R3S,2,6 R3S,3,6 R3S,4,12
export TRITON_CACHE_DIR=/data/run01/scxi253/reason3b-data/triton-cache-5090-wide
EXTRA="MOE_GMM_WIDE=1" run wide R3S,1,0 R3S,2,3
EXTRA="FP8=1" run fp8 R3S,1,0 R3S,2,3
echo DONE

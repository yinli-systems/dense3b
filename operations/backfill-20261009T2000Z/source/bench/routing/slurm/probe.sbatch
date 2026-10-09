#!/bin/bash
# Routing-study probe on one checkpoint (1 GPU). Usage: sbatch probe.sbatch CKPT OUT.json [--permute] [--k 32]
#SBATCH --job-name=rs-probe
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/data/run01/scxi253/routing/logs/%x-%j.out
cd /data/run01/scxi253/routing-src
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/data/run01/scxi253/reason3b-data/triton-cache-NVIDIAGeForceRTX4090
CK=$1; OUT=$2; shift 2
/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python -u -m bench.routing.probe "$CK" --out "$OUT" "$@"

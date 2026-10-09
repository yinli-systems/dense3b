#!/bin/bash
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
SYN=/tmp/$USER/dense3b-synth; mkdir -p $SYN
$PY -m bench.synth_data $SYN --shards 2 --tokens-per-shard 4000000 > /dev/null
$PY -m bench.budget_sweep "$SYN/synth_*.u16"

#!/bin/bash
# Build SFT (16k, masked, packed) and P2 mid-train shards from norm/: 16 tasks, partitioned by prompt hash.
#SBATCH --job-name=r3b-sft
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/data/run01/scxi253/reason3b-data/logs/%x-%A_%a.out
set -euo pipefail
cd /data/run01/scxi253/reason3b-src
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export RAYON_NUM_THREADS=3
I=$SLURM_ARRAY_TASK_ID
$PY -u -m bench.reason.build_sft --mode sft --seq 16384 --task $I --ntasks 16 --out /data/run01/scxi253/reason3b-data/sft-v1 &
$PY -u -m bench.reason.build_sft --mode mid --seq 16384 --task $I --ntasks 16 --out /data/run01/scxi253/reason3b-data/mid-v1 &
wait

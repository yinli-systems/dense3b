#!/bin/bash
# web-v1 pack: 6 workers per task claim selected files, apply the benchmark 13-gram filter, write u16 shards.
#SBATCH --job-name=code3b-webv1
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%A_%a.out
set -uo pipefail
cd /data/run01/scxi253/code3b-data-20261003/v0/src
export PYTHONPATH=/data/run01/scxi253/moe100b-parallel-quality-20261002T1701Z/deps OMP_NUM_THREADS=1
for i in 1 2 3 4 5 6; do
  /ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python -u build_web_v1.py pack --out /data/run01/scxi253/code3b-train/web-v1 &
done
wait
echo "DONE $(date +%FT%T)"

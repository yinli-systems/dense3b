#!/bin/bash
# Normalize + licence/quality filter + decontaminate the reasoning corpora: 24 array tasks x 6 processes.
#SBATCH --job-name=r3b-norm
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --array=0-23
#SBATCH --output=/data/run01/scxi253/reason3b-data/logs/%x-%A_%a.out
set -uo pipefail
cd /data/run01/scxi253/reason3b-src
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
I=$SLURM_ARRAY_TASK_ID; NT=$((24 * 6))
for k in 0 1 2 3 4 5; do
  $PY -u -m bench.reason.build_reason_data normalize --task $((I * 6 + k)) --ntasks $NT &
done
wait
echo "DONE $(date +%FT%T)"

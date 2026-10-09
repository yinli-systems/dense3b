#!/bin/bash
# One array task = 6 worker processes claiming v0 inputs until none are left. Stage = map | pack.
#   sbatch --array=0-15 --export=ALL,STAGE=map code3b/v0_stage.sbatch
# CPU-only work; the GPU is requested because the partition requires one per job.
#SBATCH --job-name=code3b-v0
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --output=/data/run01/scxi253/code3b-data-20261003/v0/logs/%x-%A_%a.out
set -uo pipefail
V0=/data/run01/scxi253/code3b-data-20261003/v0
export PYTHONPATH=/data/run01/scxi253/moe100b-parallel-quality-20261002T1701Z/deps
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
cd "$V0/src"
$PY -u build_v0.py release --config v0.json   # free claims left by tasks that timed out
for i in 1 2 3 4 5 6; do
  $PY -u build_v0.py "$STAGE" --config v0.json ${ONLY:+--only "$ONLY"} &
done
wait
echo "STAGE_DONE $STAGE $(date +%FT%T)"

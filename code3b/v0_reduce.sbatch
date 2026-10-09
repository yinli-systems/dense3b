#!/bin/bash
# Global exact + MinHash-LSH dedup over all v0 map outputs (single process; refuses if any input is unmapped).
#SBATCH --job-name=code3b-v0-reduce
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --output=/data/run01/scxi253/code3b-data-20261003/v0/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/code3b-data-20261003/v0/src
export PYTHONPATH=/data/run01/scxi253/moe100b-parallel-quality-20261002T1701Z/deps OMP_NUM_THREADS=6
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
$PY -u build_v0.py release --config v0.json
/usr/bin/time -v $PY -u build_v0.py reduce --config v0.json 2>&1 | grep -vE "^\s+(Page|Volun|Invol|Swaps|File sys|Socket|Signals|Average|Exit|Minor|Major)"

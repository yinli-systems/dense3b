#!/bin/bash
# FineMath-3+ -> 64k-tokenizer u16 shards, decontaminated. 16 tasks x 6 cores.
#SBATCH --job-name=r3b-fm3
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
export RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=true
/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python -u -m bench.reason.pack_text \
  --glob 'raw/HuggingFaceTB__finemath/finemath-3plus/*.parquet' --name finemath-3plus --task $SLURM_ARRAY_TASK_ID --ntasks 16

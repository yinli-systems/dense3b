#!/bin/bash
# Re-tokenize moe1b mix-m1 (train + val shards) into the 64k reasoning tokenizer. 24 tasks x 6 cores.
#SBATCH --job-name=r3b-retok
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --array=0-23
#SBATCH --output=/data/run01/scxi253/reason3b-data/logs/%x-%A_%a.out
set -euo pipefail
cd /data/run01/scxi253/reason3b-src
export RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=true
/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python -u -m bench.reason.retokenize \
  --list /data/run01/scxi253/moe1b/mix-m1/data.txt --list /data/run01/scxi253/moe1b/mix-m1/val.txt \
  --task $SLURM_ARRAY_TASK_ID --ntasks 24

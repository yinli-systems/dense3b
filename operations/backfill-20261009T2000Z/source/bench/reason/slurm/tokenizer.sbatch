#!/bin/bash
# Sample text, train the 64k digit-split BPE, report bytes/token vs NeoX-50k and Qwen-151k.
#SBATCH --job-name=r3b-tok
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=/data/run01/scxi253/reason3b-data/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/reason3b-src
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
OUT=/data/run01/scxi253/reason3b-data/tokenizer-v1
export RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=true
[ -e $OUT/sample/stats.json ] || $PY -u -m bench.reason.train_tokenizer sample --out $OUT
$PY -u -m bench.reason.train_tokenizer train --out $OUT
echo "DONE $(date +%FT%T)"

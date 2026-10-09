#!/bin/bash
# One-GPU evaluation of a dense3b checkpoint (default: newest complete one). Writes results/eval-<step>.json.
#   sbatch bench/slurm/eval_ckpt.sbatch [CKPT_DIR]
#SBATCH --job-name=dense3b-eval
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export PYTHONPATH=$PWD:$PWD/.deps OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
CK=${1:-$(dirname "$(ls /data/run01/scxi253/code3b-train/ckpt-v0/step_*/COMPLETE | sort | tail -1)")}
OUT=/data/run01/scxi253/code3b-train/results/eval-$(basename $CK).json
echo "evaluating $CK -> $OUT"
$PY -u -m bench.eval_ckpt "$CK" --out "$OUT"

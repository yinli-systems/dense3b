#!/bin/bash
# One-GPU base-model capability suite (bench/eval_suite.py) on one checkpoint.
#   sbatch bench/slurm/eval_suite.sbatch CKPT_DIR [extra args, e.g. --limit 6 --tasks humaneval]
#SBATCH --job-name=dense3b-eval
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export PYTHONPATH=$PWD:$PWD/.deps OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
CK=$1; shift
TAG=$(basename $CK)$( [ $# -gt 0 ] && echo "-${SLURM_JOB_ID}" || true )
[ -n "${TAG_OVERRIDE:-}" ] && TAG=$TAG_OVERRIDE
OUT=/data/run01/scxi253/code3b-train/results/suite-$TAG.json
echo "evaluating $CK -> $OUT ($*)"
$PY -u -m bench.eval_suite "$CK" --out "$OUT" "$@"

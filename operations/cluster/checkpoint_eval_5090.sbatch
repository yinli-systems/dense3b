#!/bin/bash
#SBATCH --job-name=ckpt-eval
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/data/run01/scxi253/eval-logs/%x-%j.out
set -euo pipefail
NAME=$1; CK=$2; OUTDIR=$3; TASKS=${4:-humaneval,mbpp,cruxeval,gsm8k}
mkdir -p "$OUTDIR" /data/run01/scxi253/eval-logs
cd /data/run01/scxi253/routing-src
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/scxi253-eval-${SLURM_JOB_ID}-triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/scxi253-eval-${SLURM_JOB_ID}-inductor
TAG=$(basename "$CK")
echo "EVAL_START name=$NAME ckpt=$CK gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) tasks=$TASKS"
$PY -u -m bench.eval_suite "$CK" --out "$OUTDIR/suite-$TAG.json" --tasks "$TASKS"
$PY -u -m bench.mc_eval "$CK" --out "$OUTDIR/mc-$TAG.json"
echo "EVAL_DONE name=$NAME ckpt=$CK"

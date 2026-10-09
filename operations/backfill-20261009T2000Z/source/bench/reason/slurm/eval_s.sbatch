#!/bin/bash
# r3b_s base checkpoint: code/math suite + general MC suite on one GPU (same protocol as 3B dense / 1B MoE).
# Usage: sbatch eval_s.sbatch CKPT
#SBATCH --job-name=r3b-eval
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/data/run01/scxi253/reason3b/logs/%x-%j.out
cd /data/run01/scxi253/reason3b-src
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/scxi253-r3b-eval-triton
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
CK=$1; N=$(basename $CK); R=/data/run01/scxi253/reason3b/results
$PY -u -m bench.eval_suite "$CK" --out $R/suite-$N.json --tasks humaneval,mbpp,cruxeval,gsm8k
$PY -u -m bench.mc_eval "$CK" --out $R/mc-$N.json
echo ALL_DONE

#!/bin/bash
#SBATCH --job-name=dense3b-fp8dbg
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export PYTHONPATH=$PWD:$PWD/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
D=/data/run01/scxi253/code3b-train/abdata
echo "== gpu tests"; $PY -m pytest -q -x tests/dense/test_fp8_gpu.py 2>&1 | tail -15
for extra in "" "--compile"; do
  echo "== fp8 train 2 layers $extra"
  $PY -m train.train_dense --preset dense3b --override '{"n_layers": 2, "n_dense_layers": 2}' --data "$D/train_*.u16" \
    --seq-len 4096 --global-batch 2 --micro-batch 1 --steps 4 --warmup 1 --fp8 --log-every 1 $extra 2>&1 | grep -v Warn | tail -5
done
echo "== fp8 debug probe"; $PY -m bench.fp8_probe 2>&1 | tail -30

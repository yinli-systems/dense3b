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
run() { echo "== $*"; $PY -m train.train_dense --preset dense3b --data "$D/train_*.u16" --seq-len 4096 --micro-batch 1 \
  --steps 2 --warmup 150 --fp8 --debug-nan --log-every 1 "$@" 2>&1 | grep -v Warn | grep -E "nonfinite|step|Error" | tail -4; }
run --override '{"n_layers": 8, "n_dense_layers": 8}' --global-batch 64 --compile
echo "== gpu tests"; $PY -m pytest -q -x tests/dense/test_fp8_gpu.py 2>&1 | tail -4
SYN=/tmp/$USER/dense3b-synth; mkdir -p $SYN; $PY -m bench.synth_data $SYN --shards 2 --tokens-per-shard 4000000 >/dev/null
$PY -m bench.dense_sweep --data "$SYN/synth_*.u16" --configs s4096_compile_doc_fp8 --out /data/run01/scxi253/code3b-train/results/sweep-fp8op-$SLURM_JOB_ID.json 2>&1 | grep per_layer | cut -c1-700

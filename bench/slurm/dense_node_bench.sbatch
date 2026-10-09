#!/bin/bash
# One 8-GPU node, full dense3b with the owner-sharded ZeRO path: step time / MFU / memory for several
# activation-checkpointing levels and sequence layouts. Global batch 64 (8 micro-batches per GPU) keeps it short;
# --timing splits compute from the once-per-step communication so 256-sequence steps can be extrapolated.
#SBATCH --job-name=dense3b-node-bench
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export PYTHON=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export RESULTS=/data/run01/scxi253/code3b-train/results/node-${SLURM_JOB_ID} LOCAL_SCRATCH=/tmp/$USER/dense3b
source bench/slurm/env.sh
export NCCL_DEBUG=WARN PYTHONPATH=$PWD:$PWD/.deps
diagnostics
$PYTHON -m bench.synth_data "$LOCAL_SCRATCH/synth" --shards 4 --tokens-per-shard 20000000 > /dev/null
DATA="$LOCAL_SCRATCH/synth/synth_*.u16"
for cfg in "4096 1 21" "4096 1 24" "4096 1 18" "4096 1 28" "2048 2 14" "2048 2 0"; do
  set -- $cfg; seq=$1; mb=$2; ck=$3; shift 3
  tag="s${seq}_mb${mb}_ck${ck}$( [ $# -gt 0 ] && echo _fp8 || true )"
  run_torchrun train.train_dense --preset dense3b --data "'$DATA'" --seq-len $seq --micro-batch $mb \
    --global-batch $((8 * mb * 8)) --steps 8 --warmup 2 --ckpt-layers $ck --compile --timing \
    --metrics "$RESULTS/$tag.jsonl" --log-every 1 "$@" || echo "$tag failed (likely OOM)" | tee -a "$RESULTS/failures.txt"
done
echo "done: $RESULTS"

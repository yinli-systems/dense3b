#!/bin/bash
# Day-1, one node: environment, grouped-GEMM autotune, intra-node all-to-all, memory + step-time sweep.
#SBATCH --job-name=moe7b-day1-node
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=results/%x-%j.out
set -euo pipefail
source bench/slurm/env.sh
diagnostics

# 1. grouped GEMM: correctness + autotune (one GPU), writes $RESULTS/gmm.config.json
CUDA_VISIBLE_DEVICES=0 $PYTHON -m bench.gemm --out "$RESULTS/gmm.json" || echo "gemm bench failed" | tee -a "$RESULTS/failures.txt"
CUDA_VISIBLE_DEVICES=0 $PYTHON -m pytest -q tests/moe -k gpu 2>&1 | tee "$RESULTS/pytest_gpu.txt" || true
[ -f "$RESULTS/gmm.config.json" ] || echo "no tuned tiles; using defaults" | tee -a "$RESULTS/failures.txt"
[ -f "$RESULTS/gmm.config.json" ] && export MOE_GMM_CONFIG="$RESULTS/gmm.config.json"

# 2. intra-node all-to-all (EP=8; on 5090 also EP=4, socket-local if CUDA devices 0-3 share a socket)
run_torchrun bench.comm --ep 8 --out "$RESULTS/comm_node_ep8.json"
run_torchrun bench.comm --ep 4 --out "$RESULTS/comm_node_ep4.json"

# 3. full model on one node (mesh 1x8): peak memory and phase times, synthetic tokens
$PYTHON -m bench.synth_data "$LOCAL_SCRATCH/synth" --shards 4 --tokens-per-shard 5000000 > /dev/null
DATA="$LOCAL_SCRATCH/synth/synth_*.u16"
for cfg in "1 --recompute-experts" "1" "2 --recompute-experts"; do
  set -- $cfg; mb=$1; shift
  tag="mb${mb}$( [ $# -gt 0 ] && echo _recompute || true )"
  run_torchrun train.train --preset moe7b --data "'$DATA'" --ep 8 --micro-batch $mb \
    --global-batch $((8 * mb * 4)) --steps 8 --warmup 2 --timing --metrics "$RESULTS/step_$tag.jsonl" "$@" \
    || echo "config $tag failed (likely OOM)" | tee -a "$RESULTS/failures.txt"
done
# 4. profiler trace of the default 4090 config (steps 3-5)
run_torchrun train.train --preset moe7b --data "'$DATA'" --ep 8 --micro-batch 1 --recompute-experts \
  --global-batch 32 --steps 6 --warmup 2 --profile-dir "$RESULTS/trace" --metrics "$RESULTS/step_profile.jsonl"
echo "done: $RESULTS"

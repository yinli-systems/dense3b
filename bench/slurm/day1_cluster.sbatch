#!/bin/bash
# Day-1, all 6 nodes: inter-node all-reduce (+ concurrent with all-to-all), then a real-batch throughput run.
#SBATCH --job-name=moe7b-day1-cluster
#SBATCH --partition=gpu_4090
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=results/%x-%j.out
set -euo pipefail
source bench/slurm/env.sh
diagnostics
[ -n "${MOE_GMM_CONFIG:-}" ] || echo "warning: MOE_GMM_CONFIG unset, using default tiles (run day1_node first)"

# 1. communication at training sizes: expert-gradient buckets up to 1 GB per GPU
run_torchrun bench.comm --ep 8 --ar-mb 64,256,1024 --out "$RESULTS/comm_cluster.json"
$PYTHON tools/budget.py --measured "$RESULTS/comm_cluster.json" --gpu "${GPU:-4090}" > "$RESULTS/budget_measured.md"

# 2. throughput at the real global batch (960 x 4096 = 3.93M tokens/step), synthetic tokens, 25 steps
srun --ntasks-per-node=1 $PYTHON -m bench.synth_data "$LOCAL_SCRATCH/synth" --shards 8 --tokens-per-shard 20000000 > /dev/null
DATA="$LOCAL_SCRATCH/synth/synth_*.u16"
MB=${MICRO_BATCH:-1}
run_torchrun train.train --preset moe7b --data "'$DATA'" --ep 8 --micro-batch $MB --recompute-experts \
  --global-batch 960 --steps 25 --warmup 5 --timing --profile-steps 20,21 --profile-dir "$RESULTS/trace_cluster" \
  --metrics "$RESULTS/throughput.jsonl"
echo "done: $RESULTS"

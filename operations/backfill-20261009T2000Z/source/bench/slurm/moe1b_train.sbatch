#!/bin/bash
# 1B-active MoE pretraining on GPUs scattered over the cluster: 8 nodes x 1 GPU, every rank holds the full model
# (no expert parallelism: experts run through the sync-free local path in model/moe_local.py), optimizer state
# owner-sharded over all ranks, one reduce-scatter + all-gather per optimizer step. Resumes from $CKPT.
#SBATCH --job-name=moe1b-train
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=8-32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --output=/data/run01/scxi253/moe1b/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/moe1b-src
MIXD=${MIXD:-/data/run01/scxi253/moe1b/mix-m0}
CKPT=${CKPT:-/data/run01/scxi253/moe1b/ckpt}
RESULTS=/data/run01/scxi253/moe1b/results/train-${SLURM_JOB_ID}; mkdir -p "$RESULTS" "$CKPT"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export MOE_DATA_CACHE=${MOE_DATA_CACHE:-/tmp/scxi253-moe1b-data}   # node-local copies of the shards (background)
export TRITON_CACHE_DIR=/tmp/scxi253-moe1b-triton TORCHINDUCTOR_CACHE_DIR=/tmp/scxi253-moe1b-inductor
export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
export NCCL_P2P_DISABLE=1 NCCL_NET_GDR_LEVEL=LOC NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# measured 2026-10-07 (2 nodes, 256 MB-1 GB): 4 QPs with data split across them spreads traffic over both bond legs,
# reduce-scatter/all-gather 4.3-4.9 -> 5.1-5.4 GB/s
export NCCL_IB_QPS_PER_CONNECTION=4 NCCL_IB_SPLIT_DATA_ON_QPS=1
srun bash -c 'n=$(nvidia-smi -L | wc -l); [ "$n" = 1 ] || { echo "LAYOUT_BAD $(hostname) gpus=$n"; exit 4; }' || exit 4
LAUNCH="srun --kill-on-bad-exit=1 bash -c"
# ~1,100 memory-mapped shards (each mmap holds a file descriptor): lift the soft fd limit to the hard limit
RANKENV='ulimit -n $(ulimit -Hn); RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0 LOCAL_WORLD_SIZE=1'
$LAUNCH "$RANKENV XNODE_MB=64 MASTER_PORT=$((MASTER_PORT + 1)) timeout 300 $PY -m bench.xnode_check" | tee "$RESULTS/preflight.json"
grep -q busbw "$RESULTS/preflight.json" || { echo "PREFLIGHT_FAILED on $SLURM_JOB_NODELIST"; exit 3; }
# global batch: ~${TARGET_SEQS:-512} sequences of 4096, rounded up so every rank runs the same number of micro-batches
NN=$SLURM_NNODES; MB=${MICRO:-2}; TS=${TARGET_SEQS:-$(( NN * 64 > 512 ? NN * 64 : 512 ))}   # >= 64 seqs per GPU keeps comm < ~30%
GBS=${GBS:-$(( NN * MB * ( (TS + NN * MB - 1) / (NN * MB) ) ))}
echo "NODES=$NN GBS=$GBS ($(( GBS * 4096 )) tokens/step)"
RESUME=$( ls "$CKPT"/step_*/COMPLETE >/dev/null 2>&1 && echo --resume || true )
cp $MIXD/data.txt $RESULTS/data.txt; cp $MIXD/val.txt $RESULTS/val.txt; cp $MIXD/mix.json $RESULTS/mix.json
$LAUNCH "$RANKENV $PY -m train.train_dense --preset ${PRESET:-moe1b} --data @$RESULTS/data.txt \
  --val-data \"\$(cat $RESULTS/val.txt)\" --seq-len 4096 --micro-batch ${MICRO:-2} --global-batch $GBS \
  --steps ${STEPS:-50000} --lr ${LR:-6e-4} --warmup ${WARMUP:-1000} --cooldown-frac 0.2 --ckpt-layers ${CKPT_LAYERS:-0} \
  --compile --shard-scope world --log-every 10 --ckpt-dir $CKPT --ckpt-every ${CKPT_EVERY:-250} --val-every 500 \
  --val-windows 256 --milestone-every 2500 --metrics $RESULTS/metrics.jsonl --timing $RESUME ${EXTRA:-}"

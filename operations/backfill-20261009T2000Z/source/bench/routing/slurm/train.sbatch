#!/bin/bash
# Routing-causality study (docs/ROUTING_STUDY.md): one of the three twins (PRESET=moe1b_r | dense1b_l | dense1b_s),
# all on the moe1b mix-m1 data with the same seed and a CONSTANT global batch of 512 x 4096, so every run sees the
# identical token stream step by step. Constant LR after warmup (no cooldown: interventions/forks stay comparable).
# Checkpoints every 120 steps (~252M tokens), all kept as BF16 snapshots; full states at the fork points are kept.
#SBATCH --job-name=rs-train
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=8-32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --output=/data/run01/scxi253/routing/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/routing-src
MIXD=${MIXD:-/data/run01/scxi253/moe1b/mix-m1}
PRESET=${PRESET:?set PRESET}
CKPT=${CKPT:-/data/run01/scxi253/routing/ckpt-$PRESET}
RESULTS=/data/run01/scxi253/routing/results/$PRESET-train-${SLURM_JOB_ID}; mkdir -p "$RESULTS" "$CKPT"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
CACHE_TAG=$(basename "$MIXD")-$(sha256sum "$MIXD/data.txt" | cut -c1-12)
export MOE_DATA_CACHE=${MOE_DATA_CACHE:-/var/tmp/scxi253-pretrain-cache/$CACHE_TAG}  # persistent per-node cache shared across jobs
# compiler caches in node-local RAM: a full /tmp on one node killed rs-moe1b_r (1675409, ENOSPC in Triton)
export TRITON_CACHE_DIR=/dev/shm/scxi253-rs-triton TORCHINDUCTOR_CACHE_DIR=/dev/shm/scxi253-rs-inductor
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
# a slow link drags the whole ring (measured 2026-10-08: 1.0 GB/s allocation ran at 97k tok/s vs 171k at 3.9 GB/s)
BW=$(grep -o '"all_gather_64MB_per_rank": {[^}]*' "$RESULTS/preflight.json" | grep -o 'busbw_gbs": [0-9.]*' | awk '{print $2}')
awk -v b="$BW" -v m="${MIN_BUSBW:-2.5}" 'BEGIN{exit !(b<m)}' && { echo "PREFLIGHT_SLOW busbw=$BW on $SLURM_JOB_NODELIST"; /data/run01/scxi253/slurm_retry.sh /data/run01/scxi253/routing-src/bench/routing/slurm/train.sbatch PRESET MICRO CKPT_LAYERS MIN_BUSBW EXTRA CKPT MIXD STEPS LR WARMUP || true; exit 5; }
# global batch: ~${TARGET_SEQS:-512} sequences of 4096, rounded up so every rank runs the same number of micro-batches
NN=$SLURM_NNODES; MB=${MICRO:-1}; TS=512                     # fixed: identical batches across the twins
GBS=${GBS:-$(( NN * MB * ( (TS + NN * MB - 1) / (NN * MB) ) ))}
echo "NODES=$NN GBS=$GBS ($(( GBS * 4096 )) tokens/step)"
HOT_PREFETCH_PID=
if [ "${HOT_PREFETCH:-1}" = 1 ]; then
  mkdir -p /data/run01/scxi253/prefetch-state
  touch /data/run01/scxi253/prefetch-state/${SLURM_JOB_ID}.auto
  PREFETCH_LOG="$RESULTS/hot-prefetch.log"
  PREFETCH_CMD="CUDA_VISIBLE_DEVICES= $PY -u /data/run01/scxi253/deterministic_hot_prefetch.py --manifest $MIXD/data.txt --log $RESULTS/metrics.jsonl --checkpoint-dir $CKPT --rank \$SLURM_PROCID --world $SLURM_NTASKS --gbs $GBS --horizon ${PREFETCH_HORIZON:-768} --refill ${PREFETCH_REFILL:-192} --cache-root $MOE_DATA_CACHE"
  srun --overlap -N$NN -n$NN --ntasks-per-node=1 --cpus-per-task=1 --gres=none bash -lc "$PREFETCH_CMD" >>"$PREFETCH_LOG" 2>&1 &
  HOT_PREFETCH_PID=$!
  trap 'test -n "${HOT_PREFETCH_PID:-}" && kill "$HOT_PREFETCH_PID" 2>/dev/null || true' EXIT
fi
RESUME=$( ls "$CKPT"/step_*/COMPLETE >/dev/null 2>&1 && echo --resume || true )
cp $MIXD/data.txt $RESULTS/data.txt; cp $MIXD/val.txt $RESULTS/val.txt; cp $MIXD/mix.json $RESULTS/mix.json
$LAUNCH "$RANKENV $PY -m train.train_dense --preset $PRESET --data @$RESULTS/data.txt \
  --val-data \"\$(cat $RESULTS/val.txt)\" --seq-len 4096 --micro-batch ${MICRO:-1} --global-batch $GBS \
  --steps ${STEPS:-9600} --lr ${LR:-6e-4} --warmup ${WARMUP:-1000} --cooldown-frac 0.0 --ckpt-layers ${CKPT_LAYERS:-0} --fused-ce --zero-buckets 0 \
  --compile --shard-scope world --log-every 10 --ckpt-dir $CKPT --ckpt-every 120 --val-every 120 \
  --val-windows 256 --milestone-every 120 --keep-last 2 --keep-full-steps 959,1919,2879,3839 --metrics $RESULTS/metrics.jsonl --timing $RESUME ${EXTRA:-}" || { rc=$?; /data/run01/scxi253/slurm_badnode_retry.sh /data/run01/scxi253/routing-src/bench/routing/slurm/train.sbatch /data/run01/scxi253/routing/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out PRESET MICRO CKPT_LAYERS MIN_BUSBW EXTRA CKPT MIXD STEPS LR WARMUP; exit $rc; }

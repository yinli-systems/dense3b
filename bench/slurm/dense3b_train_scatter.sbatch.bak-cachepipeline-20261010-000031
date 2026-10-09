#!/bin/bash
# 3B dense pretraining on GPUs scattered over the cluster: 8 nodes x 1 GPU (the layout verified to rendezvous and
# run NCCL cross-node: env:// static init, no intra-node shared memory, one NIC per GPU). Optimizer state is
# owner-sharded over all ranks (--shard-scope world), so each GPU holds ~17 GB of state like a full 8-GPU node;
# the once-per-step reduce-scatter / all-gather crosses the RoCE network. Resumes from $CKPT automatically.
#SBATCH --job-name=dense3b-train
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -euo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
MIXD=${MIXD:-/data/run01/scxi253/code3b-train/mix-v0}
CKPT=${CKPT:-/data/run01/scxi253/code3b-train/ckpt-v0}
RESULTS=/data/run01/scxi253/code3b-train/results/train-${SLURM_JOB_ID}; mkdir -p "$RESULTS" "$CKPT"
until [ -f "$MIXD/data.txt" ]; do sleep 60; done          # mix is written once the web shards are packed
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:$PWD/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
# Measured 2026-10-06: GPUDirect RDMA off (GeForce) doubles cross-node collective bandwidth (4.2-4.8 vs 2.5 GB/s)
# Every compute node has a USB/BMC NIC enxb03af2b6059f with the same link-local [LINK_LOCAL_ADDRESS]: NCCL bootstrap picked it
# and peers hung connecting to it. Pin bootstrap/sockets to the data network.
export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
export NCCL_P2P_DISABLE=1 NCCL_NET_GDR_LEVEL=LOC NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
# preflight: a 2-minute cross-node collective check; fail fast instead of hanging on a bad node
# every node must hold exactly one of our GPUs (static env:// rendezvous, one rank per node)
srun bash -c 'n=$(nvidia-smi -L | wc -l); [ "$n" = 1 ] || { echo "LAYOUT_BAD $(hostname) gpus=$n"; exit 4; }' || exit 4
LAUNCH="srun --kill-on-bad-exit=1 bash -c"
RANKENV='RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0 LOCAL_WORLD_SIZE=1'
$LAUNCH "$RANKENV XNODE_MB=64 MASTER_PORT=$((MASTER_PORT + 1)) timeout 300 $PY -m bench.xnode_check" | tee "$RESULTS/preflight.json"
grep -q busbw "$RESULTS/preflight.json" || { echo "PREFLIGHT_FAILED on $SLURM_JOB_NODELIST"; exit 3; }
RESUME=$( ls "$CKPT"/step_*/COMPLETE >/dev/null 2>&1 && echo --resume || true )
STEPS=${STEPS:-95000}                     # 95,000 x 256 x 4096 = 100B tokens
echo "$(cat $MIXD/data.txt)" > "$RESULTS/data.txt"; echo "$(cat $MIXD/val.txt)" > "$RESULTS/val.txt"
$LAUNCH "$RANKENV $PY -m train.train_dense --preset dense3b --data \"\$(cat $RESULTS/data.txt)\" \
  --val-data \"\$(cat $RESULTS/val.txt)\" --seq-len 4096 --micro-batch 1 --global-batch 256 --steps $STEPS \
  --lr ${LR:-7e-4} --warmup 2000 --cooldown-frac 0.2 --ckpt-layers ${CKPT_LAYERS:-21} --compile --shard-scope world \
  --log-every 10 --ckpt-dir $CKPT --ckpt-every 250 --val-every 500 --val-windows 256 --metrics $RESULTS/metrics.jsonl $RESUME"

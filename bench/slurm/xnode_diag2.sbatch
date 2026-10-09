#!/bin/bash
# Multi-node (>2 nodes, mixed GPUs per node) NCCL diagnosis with one torchrun per node.
#SBATCH --job-name=dense3b-xnode
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=3
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:45:00
#SBATCH --exclude=wqd10nba07g7,wqd10nba07g8
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:$PWD/.deps OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 XNODE_MB=64
LOG=/data/run01/scxi253/code3b-train/results/xnode2-$SLURM_JOB_ID; mkdir -p $LOG
i=0
for env in "NCCL_NET_GDR_LEVEL=LOC" "NCCL_NET_GDR_LEVEL=LOC NCCL_IB_DISABLE=1" "NCCL_NET_GDR_LEVEL=LOC NCCL_SHM_DISABLE=1" \
           "NCCL_NET_GDR_LEVEL=LOC NCCL_GIN_ENABLE=0 NCCL_NET_PLUGIN=none" ; do
  i=$((i+1)); echo "== variant $i: $env"
  ( export NCCL_P2P_DISABLE=1 $env NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$LOG/v$i.%h.%p.log
    timeout 150 bench/slurm/per_node_torchrun.sh $((22000 + i)) bench.xnode_check 2>&1 | grep -E "busbw|\[launch\]" )
  echo "   exit=$?  last NCCL lines: $(tail -qn1 $LOG/v$i.* 2>/dev/null | cut -c1-160 | sort | uniq -c | head -3 | tr '\n' ' ')"
done

#!/bin/bash
# 8 nodes x 1 GPU NCCL diagnosis (the training layout), full NCCL INFO logs per variant.
#SBATCH --job-name=dense3b-xnode
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python PYTHONPATH=$PWD:$PWD/.deps XNODE_MB=64
LOG=/data/run01/scxi253/code3b-train/results/xnode8-$SLURM_JOB_ID; mkdir -p $LOG
echo "nodes: $SLURM_JOB_NODELIST"
srun bash -c 'echo "$(hostname) $(ibv_devinfo 2>/dev/null | grep -E "hca_id|link_layer|state" | tr -s " \t" " " | tr "\n" " ")"' | sort
i=0
for env in "NCCL_NET_GDR_LEVEL=LOC" "NCCL_NET_GDR_LEVEL=LOC NCCL_IB_GID_INDEX=3" "NCCL_IB_DISABLE=1"; do
  i=$((i+1)); echo "== variant $i: $env"
  srun --kill-on-bad-exit=1 bash -c "export NCCL_P2P_DISABLE=1 $env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,BOOTSTRAP \
    NCCL_DEBUG_FILE=$LOG/v$i.\$(hostname).log MASTER_PORT=$((23000 + i)) RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=0; \
    timeout 150 $PY -m bench.xnode_check" 2>&1 | grep -E "busbw" | cut -c1-400
  echo "   per-node last NCCL line:"; for f in $LOG/v$i.*.log; do echo "     $(basename $f): $(grep -v '^\s*$' $f | tail -1 | cut -c1-170)"; done
done

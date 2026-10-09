#!/bin/bash
# Cross-node NCCL diagnosis: same 2-rank bandwidth check under several network settings, each with a timeout.
#SBATCH --job-name=dense3b-xnode
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=6
#SBATCH --time=00:40:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export PYTHONPATH=$PWD:$PWD/.deps NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
LOG=/data/run01/scxi253/code3b-train/results/xnode-$SLURM_JOB_ID; mkdir -p $LOG
i=0
for env in "" "NCCL_NET_GDR_LEVEL=LOC" "NCCL_NET_GDR_LEVEL=LOC NCCL_IB_GID_INDEX=3" "NCCL_IB_DISABLE=1"; do
  i=$((i+1)); echo "== variant $i: ${env:-defaults}"
  srun --kill-on-bad-exit=1 bash -c "export $env MASTER_PORT=$((21000 + i)) RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS \
    LOCAL_RANK=\$SLURM_LOCALID NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$LOG/v$i.%h.%p.log; timeout 150 $PY -m bench.xnode_check" 2>&1 \
    | grep -E "busbw|Error|error" | head -3 || true
  echo "   exit=$? transport: $(grep -hoE 'via NET/[A-Za-z0-9_]+(/[0-9]+)?(/GDRDMA)?' $LOG/v$i.* 2>/dev/null | sort | uniq -c | head -3 | tr '\n' ' ')"
done

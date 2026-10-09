#!/bin/bash
#SBATCH --job-name=dense3b-xnode
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export PYTHONPATH=$PWD:$PWD/.deps NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET OMP_NUM_THREADS=4
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
L="RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID LOCAL_WORLD_SIZE=1"
srun bash -c "$L $PY -m bench.xnode_check" 2>&1 | grep -E "busbw|NET/|via NET|error|Error" | sort | uniq | head -12
srun bash -c "$L $PY -m train.train_dense --preset dense3b --override '{\"n_layers\": 2, \"n_dense_layers\": 2}' \
  --data '/data/run01/scxi253/code3b-train/abdata/train_*.u16' --seq-len 4096 --global-batch 4 --steps 4 --warmup 1 \
  --shard-scope world --compile --log-every 1" 2>&1 | grep -E '"step"|Error|error' | grep -v Warn | tail -5

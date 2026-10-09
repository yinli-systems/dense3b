#!/bin/bash
#SBATCH --job-name=dense3b-xnode
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=3
#SBATCH --gpus=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --exclude=wqd10nba07g7,wqd10nba07g8
#SBATCH --output=/data/run01/scxi253/code3b-train/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/code3b-dense3b-src
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) PORT=29877
export PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
echo "MASTER_ADDR=$MASTER_ADDR nodes=$SLURM_JOB_NODELIST"
srun --ntasks-per-node=1 bash -c 'echo "$(hostname): resolves $MASTER_ADDR -> $(getent hosts $MASTER_ADDR | head -1); my ips: $(hostname -I | cut -c1-80)"'
cat > /tmp/scxi_rdzv_$SLURM_JOB_ID.py <<PYEOF
import os, socket, torch.distributed as dist
dist.init_process_group("gloo"); print(socket.gethostname(), "RANK", dist.get_rank(), "WORLD", dist.get_world_size(), flush=True)
PYEOF
cp /tmp/scxi_rdzv_$SLURM_JOB_ID.py /data/run01/scxi253/code3b-train/results/rdzv_probe.py
srun --ntasks-per-node=1 bash -c 'timeout 120 $PY -m torch.distributed.run --nnodes=3 --nproc-per-node=1 --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$PORT --rdzv-id=d$SLURM_JOB_ID /data/run01/scxi253/code3b-train/results/rdzv_probe.py 2>&1 | grep -vE "^\s*$" | tail -6 | sed "s/^/$(hostname): /"'
echo "== static rendezvous (MASTER_ADDR/PORT env, no c10d)"
srun --ntasks-per-node=1 bash -c 'timeout 120 $PY -m torch.distributed.run --nnodes=3 --nproc-per-node=1 --node-rank=$SLURM_NODEID \
  --master-addr=$MASTER_ADDR --master-port=$((PORT+1)) /data/run01/scxi253/code3b-train/results/rdzv_probe.py 2>&1 | grep -vE "^\s*$" | tail -4 | sed "s/^/$(hostname): /"'

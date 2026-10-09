#!/bin/bash
# Bucketed (overlapped) ZeRO vs plain OwnerZeRO on 2 GPUs: identical seeds/data, per-step loss and grad norm must match.
#SBATCH --job-name=moe1b-zeq
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=/data/run01/scxi253/moe1b/logs/%x-%j.out
cd /data/run01/scxi253/moe1b-src-dev
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/tmp/scxi253-zeq-triton NCCL_P2P_DISABLE=1
ulimit -n $(ulimit -Hn)
OUT=/data/run01/scxi253/moe1b/results/zeq-$SLURM_JOB_ID; mkdir -p $OUT
for K in 1 2 4; do
  $PY -m torch.distributed.run --standalone --nproc-per-node 2 -m train.train_dense --preset moe1b --override '{"n_layers": 8}' \
    --data @/data/run01/scxi253/moe1b/mix-m1/data.txt --seq-len 4096 --micro-batch 1 --global-batch 8 --steps 6 \
    --lr 6e-4 --warmup 2 --compile --shard-scope world --log-every 1 --timing --fused-ce --zero-buckets $K \
    --metrics $OUT/k$K.jsonl > $OUT/k$K.log 2>&1
  echo "K=$K rc=$?"
done
$PY - <<PYEOF
import json
r = {K: [json.loads(l) for l in open(f'$OUT/k{K}.jsonl')] for K in (1, 2, 4)}
for K in (2, 4):
    same = all(a['loss'] == b['loss'] and a['grad_norm'] == b['grad_norm'] for a, b in zip(r[1], r[K])) and len(r[1]) == len(r[K])
    print('K', K, 'IDENTICAL' if same else 'DIFFERENT', [(round(a['loss'], 5), round(b['loss'], 5)) for a, b in zip(r[1], r[K])])
    print('   step_s', [round(x['step_s'], 2) for x in r[K]], 'vs', [round(x['step_s'], 2) for x in r[1]])
PYEOF
grep -h "zero_padding" $OUT/k*.log | cut -c1-200

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
# wrapper (2026-10-08): runs the original script unchanged; on a bad-data-node failure resubmits itself without that node
set -uo pipefail
bash /data/run01/scxi253/code3b-dense3b-src/bench/slurm/dense3b_train_scatter.sbatch; rc=$?
[ $rc -eq 0 ] || /data/run01/scxi253/slurm_badnode_retry.sh /data/run01/scxi253/wrappers/dense3b.sbatch /data/run01/scxi253/code3b-train/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out MIXD CKPT STEPS
exit $rc

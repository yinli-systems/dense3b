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
# wrapper (2026-10-08): runs the original script unchanged; on a bad-data-node failure resubmits itself without that node
set -uo pipefail
bash /data/run01/scxi253/moe1b-src/bench/slurm/moe1b_train.sbatch; rc=$?
[ $rc -eq 0 ] || /data/run01/scxi253/slurm_badnode_retry.sh /data/run01/scxi253/wrappers/moe1b.sbatch /data/run01/scxi253/moe1b/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out MIXD CKPT CKPT_LAYERS CKPT_EVERY STEPS LR EXTRA TARGET_SEQS
exit $rc

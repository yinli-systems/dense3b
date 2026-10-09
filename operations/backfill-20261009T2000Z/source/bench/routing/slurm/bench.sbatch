#!/bin/bash
# One 4090: throughput / memory of the three routing-study twins (micro x recompute) to tune the training jobs.
#SBATCH --job-name=rs-bench
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/data/run01/scxi253/routing/logs/%x-%j.out
cd /data/run01/scxi253/routing-src
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/data/run01/scxi253/reason3b-data/triton-cache-NVIDIAGeForceRTX4090
for cfg in MR,2,6 MR,2,0 MR,4,6 DL,1,6 DL,1,0 DL,2,6 DS,2,0 DS,4,0; do
  $PY -m bench.moe1b_sweep --out /dev/null --one $cfg 2>&1 | grep RESULT
done
echo DONE

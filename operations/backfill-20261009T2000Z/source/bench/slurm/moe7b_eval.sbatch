#!/bin/bash
# RunPod 7B MoE on one 4090: convert the OLMo-core export, verify against reference logits from the original code,
# then the same code/math and general-text suites as the 3B and 1B MoE.
#SBATCH --job-name=moe7b-eval
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --output=/data/run01/scxi253/moe7b-eval/logs/%x-%j.out
set -uo pipefail
cd /data/run01/scxi253/moe1b-src-dev
PY=/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python
export PYTHONPATH=$PWD:/data/run01/scxi253/code3b-dense3b-src/.deps OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/scxi253-moe7b-eval-triton
E=/data/run01/scxi253/moe7b-eval; STEP=${STEP:-69000}; CK=$E/ckpt/step_000$STEP
[ -f $CK/COMPLETE ] || $PY -u -m bench.convert_olmo_moe7b $E/weights/weights/step$STEP/model.safetensors $CK --step $STEP \
    --tokenizer $E/meta/tokenizer/tokenizer.json || exit 2
$PY -u -m bench.verify_port $CK $E/heldout || exit 3
$PY -u -m bench.mc_eval $CK --out $E/results/mc-step$STEP.json
$PY -u -m bench.eval_suite $CK --out $E/results/suite-step$STEP.json --tasks humaneval,mbpp,cruxeval,gsm8k,lcb
echo ALL_DONE

#!/bin/bash
# keep_and_eval.sh STEP: when ckpt-v0/step_STEP is COMPLETE, hardlink its BF16 model into milestones/ and submit the eval suite.
S=$(printf "%08d" $1); C=/data/run01/scxi253/code3b-train/ckpt-v0/step_$S; M=/data/run01/scxi253/code3b-train/milestones/step_$S
until [ -f $C/COMPLETE ]; do sleep 120; done
mkdir -p $M && ln -f $C/model.pt $M/model.pt && cp $C/meta.json $M/ && echo "$(date +%FT%T) kept $M"
cd /data/run01/scxi253/code3b-dense3b-src && sbatch bench/slurm/eval_suite.sbatch $M

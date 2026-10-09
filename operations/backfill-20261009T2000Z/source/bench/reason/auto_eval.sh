#!/bin/bash
# Submit eval_s.sbatch for every new r3b_s milestone checkpoint (runs on the login node; one line per submission).
CK=/data/run01/scxi253/reason3b/ckpt-s; SEEN=/data/run01/scxi253/reason3b/results/.auto_eval_seen
touch $SEEN
while true; do
  for d in $(ls -d $CK/milestones/step_* $CK/step_* 2>/dev/null); do
    s=$(basename $d)
    n=$((10#${s#step_}))
    [ $(( (n + 1) % 2500 )) = 0 ] || continue
    grep -qx "$s" $SEEN && continue
    [ -e $d/model.pt ] && [ -e $d/meta.json ] && { [ -e $d/COMPLETE ] || [ -e $d/MILESTONE ]; } || continue
    echo "$s" >> $SEEN
    cd /data/run01/scxi253/reason3b-src && echo "$(date +%FT%T) $s $(sbatch --parsable bench/reason/slurm/eval_s.sbatch $d)"
  done
  sleep 600
done

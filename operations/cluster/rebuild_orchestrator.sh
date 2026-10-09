#!/bin/bash
# Rebuild orchestrator (2026-10-08): the training data deleted at 18:49 CST is being re-packed from the kept-document
# lists. Every 5 min: a mix is "ready" when every shard in data.txt + val.txt exists and is non-empty; then its
# training jobs are submitted once (through the auto-retry wrappers / hooks). Log: rebuild_orchestrator.log
S=/data/run01/scxi253; LOG=$S/rebuild_orchestrator.log; ST=$S/rebuild_orchestrator.state; touch $ST
log() { echo "$(date +%FT%T) $*" >> $LOG; }
ready() {  # mix dir -> 0 if every listed shard exists with size > 0
  local m=$1 n=0 bad=0
  for p in $( { cat $m/data.txt; echo; cat $m/val.txt; } | tr "," "\n" | sed -E "s#:[0-9.]+\$##" | grep -v "^\$"); do n=$((n+1)); [ -s "$p" ] || bad=$((bad+1)); done
  echo "$m shards=$n missing=$bad" >> $LOG.progress; [ $n -gt 0 ] && [ $bad -eq 0 ]
}
done_() { grep -qx "$1" $ST; }
mark() { echo "$1" >> $ST; }
log "orchestrator started pid $$"
while true; do
  : > $LOG.progress
  if ! done_ moe1b && ready $S/vaccines/vaccine-m1; then
    cd $S/moe1b-src; a=$(sbatch --parsable --job-name=IgG-moe1b --export=ALL,MIXD=$S/vaccines/vaccine-m1,CKPT=$S/moe1b/ckpt,CKPT_LAYERS=6,CKPT_EVERY=250,STEPS=100000,LR=6e-4,EXTRA="--fused-ce --zero-buckets 0 --prev-global-batch 512" $S/wrappers/moe1b.sbatch)
    cd $S/routing-src
    b=$(sbatch --parsable --job-name=epitope-dense1b_l --export=ALL,MIXD=$S/vaccines/vaccine-m1,PRESET=dense1b_l,MICRO=1,CKPT_LAYERS=6,MIN_BUSBW=2.0 bench/routing/slurm/train.sbatch)
    c=$(sbatch --parsable --job-name=epitope-moe1b_r --export=ALL,MIXD=$S/vaccines/vaccine-m1,PRESET=moe1b_r,MICRO=2,CKPT_LAYERS=0,MIN_BUSBW=2.0 bench/routing/slurm/train.sbatch)
    d=$(sbatch --parsable --job-name=epitope-dense1b_s --export=ALL,MIXD=$S/vaccines/vaccine-m1,PRESET=dense1b_s,MICRO=2,CKPT_LAYERS=0,MIN_BUSBW=2.0 bench/routing/slurm/train.sbatch)
    mark moe1b; log "mix-m1 READY -> submitted moe1b $a rs-dense1b_l $b rs-moe1b_r $c rs-dense1b_s $d"
  fi
  if ! done_ dense3b && ready $S/vaccines/vaccine-v0; then
    a=$(sbatch --parsable --job-name=IgM-dense3b --export=ALL,MIXD=$S/vaccines/vaccine-v0 $S/wrappers/dense3b.sbatch); mark dense3b; log "mix-v0 READY -> submitted dense3b $a"
  fi
  if ! done_ r3b && ready $S/vaccines/vaccine-p1; then
    cd $S/reason3b-src; a=$(sbatch --parsable --job-name=IgE-r3b --exclude=wqd10naf02g1,wqd10naf02g2,wqd10nah10g1,wqd10nah10g5,wqd10nah11g1 --export=ALL,MIXD=$S/vaccines/vaccine-p1,MICRO=2,EXTRA="--keep-last 2" bench/reason/slurm/train_s.sbatch)
    mark r3b; log "mix-p1 READY -> submitted r3b-s $a"
  fi
  done_ moe1b && done_ dense3b && done_ r3b && { log "all submitted, exiting"; exit 0; }
  sleep 300
done

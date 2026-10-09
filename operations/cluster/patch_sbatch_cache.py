from pathlib import Path
import shutil, sys, time
STAMP=time.strftime('%Y%m%d-%H%M%S')
CACHE='''CACHE_TAG=$(basename "$MIXD")-$(sha256sum "$MIXD/data.txt" | cut -c1-12)
export MOE_DATA_CACHE=${MOE_DATA_CACHE:-/ssd/scxi253/pretrain-cache/$CACHE_TAG}  # persistent per-node cache shared across jobs
'''
HELPER='''HOT_PREFETCH_PID=
if [ "${HOT_PREFETCH:-1}" = 1 ]; then
  PREFETCH_LOG="$RESULTS/hot-prefetch.log"
  PREFETCH_CMD="CUDA_VISIBLE_DEVICES= $PY -u /data/run01/scxi253/deterministic_hot_prefetch.py --manifest $MIXD/data.txt --log $RESULTS/metrics.jsonl --checkpoint-dir $CKPT --rank \\$SLURM_PROCID --world $SLURM_NTASKS --gbs $GBS --horizon ${PREFETCH_HORIZON:-768} --refill ${PREFETCH_REFILL:-192} --cache-root $MOE_DATA_CACHE"
  srun --overlap -N$NN -n$NN --ntasks-per-node=1 --cpus-per-task=1 --gres=none bash -lc "$PREFETCH_CMD" >>"$PREFETCH_LOG" 2>&1 &
  HOT_PREFETCH_PID=$!
  trap 'test -n "${HOT_PREFETCH_PID:-}" && kill "$HOT_PREFETCH_PID" 2>/dev/null || true' EXIT
fi
'''

def backup(p):
    b=p.with_name(p.name+'.bak-cachepipeline-'+STAMP); shutil.copy2(p,b); return b

def patch_common(path, old_cache):
    p=Path(path).resolve(); b=backup(p); s=p.read_text()
    if 'persistent per-node cache shared across jobs' not in s:
        if old_cache not in s: raise SystemExit(f'cache line missing: {p}')
        s=s.replace(old_cache,CACHE.rstrip(),1)
    marker='echo "NODES=$NN GBS=$GBS ($(( GBS * 4096 )) tokens/step)"'
    if 'deterministic_hot_prefetch.py' not in s:
        if marker not in s: raise SystemExit(f'GBS marker missing: {p}')
        s=s.replace(marker,marker+'\n'+HELPER.rstrip(),1)
    p.write_text(s); print('patched',p,'backup',b)

patch_common('/data/run01/scxi253/moe1b-src/bench/slurm/moe1b_train.sbatch',
             'export MOE_DATA_CACHE=${MOE_DATA_CACHE:-/tmp/scxi253-moe1b-data}   # node-local copies of the shards (background)')
patch_common('/data/run01/scxi253/routing-src/bench/routing/slurm/train.sbatch',
             'export MOE_DATA_CACHE=${MOE_DATA_CACHE:-/tmp/scxi253-moe1b-data}   # node-local copies of the shards (background)')
patch_common('/data/run01/scxi253/reason3b-src/bench/reason/slurm/train_s.sbatch',
             'export MOE_DATA_CACHE=${MOE_DATA_CACHE:-/tmp/scxi253-r3b-data}   # node-local copies of the shards (background)')

p=Path('/data/run01/scxi253/code3b-dense3b-src/bench/slurm/dense3b_train_scatter.sbatch').resolve(); b=backup(p); s=p.read_text()
if 'persistent per-node cache shared across jobs' not in s:
    marker='export PYTHONPATH=$PWD:$PWD/.deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4'
    if marker not in s: raise SystemExit('dense env marker missing')
    s=s.replace(marker,marker+'\n'+CACHE.rstrip(),1)
if 'deterministic_hot_prefetch.py' not in s:
    marker='STEPS=${STEPS:-95000}                     # 95,000 x 256 x 4096 = 100B tokens'
    if marker not in s: raise SystemExit('dense steps marker missing')
    s=s.replace(marker,marker+'\nNN=$SLURM_NNODES; GBS=256\n'+HELPER.rstrip(),1)
p.write_text(s); print('patched',p,'backup',b)

#!/usr/bin/env python3
from __future__ import annotations
import hashlib, os, re, shlex, subprocess, time
from pathlib import Path

S=Path('/data/run01/scxi253')
STATE=S/'prefetch-state'; STATE.mkdir(exist_ok=True)
LOG=S/'prefetch-supervisor.log'
FIRST={}

def log(msg):
    line=f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'
    print(line,flush=True)
    with LOG.open('a') as f: f.write(line+'\n')

def config(name,jid):
    if name=='IgG-moe1b':
        return S/'vaccines/vaccine-m1/data.txt', S/f'moe1b/logs/{name}-{jid}.out', S/'moe1b/ckpt', 512, '/tmp/scxi253-moe1b-data'
    if name.startswith('epitope-'):
        preset=name.removeprefix('epitope-')
        return S/'vaccines/vaccine-m1/data.txt', S/f'routing/logs/{name}-{jid}.out', S/f'routing/ckpt-{preset}', 512, '/tmp/scxi253-moe1b-data'
    if name=='IgM-dense3b':
        return S/'vaccines/vaccine-v0/data.txt', S/f'code3b-train/logs/{name}-{jid}.out', S/'code3b-train/ckpt-v0', 256, ''
    if name=='IgE-r3b':
        return S/'vaccines/vaccine-p1/data.txt', S/f'reason3b/logs/{name}-{jid}.out', S/'reason3b/ckpt-s', 512, '/tmp/scxi253-r3b-data'
    return None

def job_nodes(jid):
    s=subprocess.check_output(['scontrol','show','job',jid],text=True,timeout=10)
    m=re.search(r'NumNodes=(\d+)',s)
    return int(m.group(1)) if m else 8

def numeric_steps(jid):
    try:
        s=subprocess.check_output(['squeue','--steps','-j',jid,'-h','-o','%i'],text=True,timeout=10)
        return [x for x in s.split() if re.fullmatch(re.escape(jid)+r'\.\d+',x)]
    except Exception:
        return []

def persistent_root(manifest):
    h=hashlib.sha256(Path(manifest).read_bytes()).hexdigest()[:12]
    return f'/var/tmp/scxi253-pretrain-cache/{Path(manifest).parent.name}-{h}'

def launch(jid,name,cfg,cache_root):
    manifest,joblog,ckpt,gbs,_=cfg; nodes=job_nodes(jid)
    out=S/f'{name}-hotprefetch-{jid}.log'
    inner=(
        'CUDA_VISIBLE_DEVICES= python3 -u /data/run01/scxi253/deterministic_hot_prefetch.py '
        f'--manifest {shlex.quote(str(manifest))} --log {shlex.quote(str(joblog))} '
        f'--checkpoint-dir {shlex.quote(str(ckpt))} --rank "$SLURM_PROCID" '
        f'--world {nodes} --gbs {gbs} --horizon 768 --refill 192 '
        + (f'--cache-root {shlex.quote(cache_root)}' if cache_root else '')
    )
    cmd=['srun',f'--jobid={jid}','--overlap',f'-N{nodes}',f'-n{nodes}','--ntasks-per-node=1','--cpus-per-task=1','--gres=none','bash','-lc',inner]
    f=open(out,'ab',buffering=0)
    p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (STATE/f'{jid}.supervisor').write_text(f'{p.pid} {name}\n')
    log(f'launched job={jid} name={name} pid={p.pid} nodes={nodes} gbs={gbs} cache={cache_root or "page-cache-only"}')

log(f'supervisor started pid={os.getpid()}')
while True:
    try:
        raw=subprocess.check_output(['squeue','-u',os.environ.get('USER','scxi253'),'-h','-o','%i|%j|%t'],text=True,timeout=15)
        now=time.time()
        live=set()
        for line in raw.splitlines():
            try: jid,name,state=line.split('|',2)
            except ValueError: continue
            if state!='R' or not config(name,jid): continue
            live.add(jid); FIRST.setdefault(jid,now)
            if (STATE/f'{jid}.skip').exists(): continue
            steps=numeric_steps(jid)
            markers=list(STATE.glob(f'{jid}.*'))
            if len(steps)>=2:
                continue
            age=now-FIRST[jid]
            auto=(STATE/f'{jid}.auto').exists()
            # Old queued jobs have no auto marker: attach the sidecar promptly. Patched jobs get
            # a longer grace period so their in-job helper can create its own overlapping step.
            if (not auto and age < 20) or (auto and age < 150):
                continue
            if any(p.suffix in ('.manual','.supervisor') for p in markers):
                # Existing manual helper may be between refills; do not duplicate.
                continue
            cfg=config(name,jid)
            cache=persistent_root(cfg[0]) if auto else cfg[4]
            launch(jid,name,cfg,cache)
        for jid in list(FIRST):
            if jid not in live: FIRST.pop(jid,None)
    except Exception as e:
        log(f'loop error={e!r}')
    time.sleep(20)

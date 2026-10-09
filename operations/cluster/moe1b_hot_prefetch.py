#!/usr/bin/env python3
import argparse, collections, glob, hashlib, os, re, time
from pathlib import Path
import numpy as np

def parse_specs(manifest):
    txt=Path(manifest).read_text().strip()
    out=[]
    for spec in txt.split(','):
        spec=spec.strip()
        if not spec: continue
        last=spec.rsplit(':',1)
        if len(last)==2 and last[1].isdigit(): pat,rep=last[0],int(last[1])
        else: pat,rep=spec,1
        for p in sorted(glob.glob(pat)):
            for _ in range(rep): out.append(p)
    return out

def latest_step(log):
    try:
        with open(log,'rb') as f:
            f.seek(0,2); n=f.tell(); f.seek(max(0,n-262144))
            s=f.read().decode('utf-8','ignore')
        xs=re.findall(r'"step"\s*:\s*(\d+)',s)
        return int(xs[-1]) if xs else None
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True); ap.add_argument('--log',required=True)
    ap.add_argument('--rank',type=int,required=True); ap.add_argument('--world',type=int,default=8)
    ap.add_argument('--seq',type=int,default=4096); ap.add_argument('--gbs',type=int,default=512)
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--horizon',type=int,default=512)
    ap.add_argument('--refill',type=int,default=128); ap.add_argument('--poll',type=int,default=20)
    ap.add_argument('--cache-root',default='/tmp/scxi253-moe1b-data')
    a=ap.parse_args()
    paths=parse_specs(a.manifest)
    counts=np.fromiter(((os.path.getsize(p)//2 - 1)//a.seq for p in paths), dtype=np.int64)
    cum=np.concatenate((np.array([0],dtype=np.int64),np.cumsum(counts,dtype=np.int64)))
    n=int(cum[-1]); per_rank=a.gbs//a.world
    assert a.gbs % a.world == 0
    perms={}
    def perm(epoch):
        if epoch not in perms:
            perms.clear()
            perms[epoch]=np.random.Generator(np.random.PCG64([a.seed,epoch])).permutation(n).astype(np.int64)
        return perms[epoch]
    def local_ok(p):
        lp=os.path.join(a.cache_root, hashlib.sha1(p.encode()).hexdigest()[:20]+'.u16')
        try: return os.path.getsize(lp)==os.path.getsize(p)
        except OSError: return False
    def warm(lo,hi):
        # rank-local sample positions for [lo, hi)
        blocks=[np.arange(st*a.gbs+a.rank*per_rank, st*a.gbs+(a.rank+1)*per_rank, dtype=np.int64) for st in range(lo,hi)]
        pos=np.concatenate(blocks) if blocks else np.empty(0,dtype=np.int64)
        bypath=collections.defaultdict(list)
        for epoch in np.unique(pos//n):
            mask=(pos//n)==epoch; ii=pos[mask]%n; jj=perm(int(epoch))[ii]
            ss=np.searchsorted(cum,jj,side='right')-1
            offs=(jj-cum[ss])*a.seq*2
            for s,o in zip(ss.tolist(),offs.tolist()): bypath[paths[s]].append(o)
        warmed=skipped=bytes_read=0; t=time.time()
        for p,offs in bypath.items():
            if local_ok(p):
                skipped += len(offs); continue
            try:
                fd=os.open(p,os.O_RDONLY)
                try:
                    for o in sorted(offs):
                        b=os.pread(fd,(a.seq+1)*2,o)
                        bytes_read += len(b); warmed += 1
                finally: os.close(fd)
            except OSError as e:
                print(f'PREFETCH_READ_ERROR rank={a.rank} path={p} err={e}',flush=True)
        dt=time.time()-t
        print(f'PREFETCH rank={a.rank} host={os.uname().nodename} steps={lo}:{hi} windows={len(pos)} warmed={warmed} local_skip={skipped} bytes={bytes_read} sec={dt:.2f}',flush=True)
    pref_end=-1
    print(f'PREFETCH_START rank={a.rank} host={os.uname().nodename} n_windows={n} paths={len(paths)}',flush=True)
    while True:
        st=latest_step(a.log)
        if st is None:
            time.sleep(a.poll); continue
        target=st+a.horizon
        if pref_end < 0:
            lo=st+8
        elif st >= pref_end-a.refill:
            lo=pref_end
        else:
            time.sleep(a.poll); continue
        hi=max(lo,target)
        if hi>lo:
            warm(lo,hi); pref_end=hi
        time.sleep(a.poll)
if __name__=='__main__': main()

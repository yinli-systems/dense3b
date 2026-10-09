#!/usr/bin/env python3
"""Warm the exact future rank-local token windows without touching GPUs or training state.

The training sampler is deterministic. This helper reproduces only the index mapping and issues
small pread() calls for future windows that are not already present in LocalCache. It therefore
warms the node's page cache while preserving sample order, checkpoint semantics, and model state.
"""
from __future__ import annotations
import argparse, collections, glob, hashlib, json, os, re, time
from pathlib import Path
import numpy as np


def parse_specs(manifest: str) -> list[str]:
    txt = Path(manifest).read_text().strip()
    out: list[str] = []
    for spec in txt.split(','):
        spec = spec.strip()
        if not spec:
            continue
        last = spec.rsplit(':', 1)
        if len(last) == 2 and last[1].isdigit():
            pat, rep = last[0], int(last[1])
        else:
            pat, rep = spec, 1
        matches = sorted(glob.glob(pat))
        if not matches:
            raise FileNotFoundError(f'no shard matches {pat!r}')
        for p in matches:
            out.extend([p] * rep)
    return out


def latest_logged_step(log: str) -> int | None:
    try:
        with open(log, 'rb') as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 524288))
            s = f.read().decode('utf-8', 'ignore')
        xs = re.findall(r'"step"\s*:\s*(\d+)', s)
        return int(xs[-1]) if xs else None
    except OSError:
        return None


def latest_checkpoint_step(root: str | None) -> int | None:
    if not root:
        return None
    best = None
    for meta in Path(root).glob('step_*/meta.json'):
        try:
            x = json.loads(meta.read_text()).get('step')
            if isinstance(x, int) and (best is None or x > best):
                best = x
        except Exception:
            pass
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--checkpoint-dir')
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--world', type=int, default=8)
    ap.add_argument('--seq', type=int, default=4096)
    ap.add_argument('--gbs', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--horizon', type=int, default=512)
    ap.add_argument('--refill', type=int, default=128)
    ap.add_argument('--poll', type=int, default=15)
    ap.add_argument('--lead', type=int, default=8)
    ap.add_argument('--cache-root', default='')
    ap.add_argument('--stop-file', default='')
    a = ap.parse_args()
    if a.gbs % a.world:
        raise ValueError(f'gbs={a.gbs} not divisible by world={a.world}')

    paths = parse_specs(a.manifest)
    counts = np.fromiter((((os.path.getsize(p) // 2) - 1) // a.seq for p in paths), dtype=np.int64)
    cum = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    n = int(cum[-1])
    if n <= 0:
        raise ValueError('manifest has no full windows')
    per_rank = a.gbs // a.world
    perms: dict[int, np.ndarray] = {}

    def perm(epoch: int) -> np.ndarray:
        if epoch not in perms:
            perms.clear()
            perms[epoch] = np.random.Generator(np.random.PCG64([a.seed, epoch])).permutation(n).astype(np.int64)
        return perms[epoch]

    def local_ok(p: str) -> bool:
        if not a.cache_root:
            return False
        lp = os.path.join(a.cache_root, hashlib.sha1(p.encode()).hexdigest()[:20] + '.u16')
        try:
            return os.path.getsize(lp) == os.path.getsize(p)
        except OSError:
            return False

    def warm(lo: int, hi: int) -> None:
        positions = np.concatenate([
            np.arange(st * a.gbs + a.rank * per_rank, st * a.gbs + (a.rank + 1) * per_rank, dtype=np.int64)
            for st in range(lo, hi)
        ]) if hi > lo else np.empty(0, dtype=np.int64)
        bypath: dict[str, list[int]] = collections.defaultdict(list)
        for epoch in np.unique(positions // n):
            mask = (positions // n) == epoch
            ii = positions[mask] % n
            jj = perm(int(epoch))[ii]
            ss = np.searchsorted(cum, jj, side='right') - 1
            offs = (jj - cum[ss]) * a.seq * 2
            for s, o in zip(ss.tolist(), offs.tolist()):
                bypath[paths[s]].append(o)

        warmed = skipped = bytes_read = errors = 0
        t0 = time.time()
        for p, offsets in bypath.items():
            if local_ok(p):
                skipped += len(offsets)
                continue
            try:
                fd = os.open(p, os.O_RDONLY)
                try:
                    for o in sorted(set(offsets)):
                        b = os.pread(fd, (a.seq + 1) * 2, o)
                        bytes_read += len(b)
                        warmed += 1
                finally:
                    os.close(fd)
            except OSError as e:
                errors += 1
                print(f'PREFETCH_READ_ERROR rank={a.rank} host={os.uname().nodename} path={p} err={e}', flush=True)
        dt = time.time() - t0
        print(
            f'PREFETCH rank={a.rank} host={os.uname().nodename} steps={lo}:{hi} '
            f'windows={len(positions)} warmed={warmed} local_skip={skipped} bytes={bytes_read} '
            f'errors={errors} sec={dt:.2f}', flush=True
        )

    ckpt = latest_checkpoint_step(a.checkpoint_dir)
    print(
        f'PREFETCH_START rank={a.rank} host={os.uname().nodename} n_windows={n} '
        f'paths={len(paths)} checkpoint_step={ckpt}', flush=True
    )
    pref_end = -1
    while True:
        if a.stop_file and os.path.exists(a.stop_file):
            print(f'PREFETCH_STOP rank={a.rank} stop_file={a.stop_file}', flush=True)
            return
        st = latest_logged_step(a.log)
        if st is None:
            st = ckpt
        if st is None:
            time.sleep(a.poll)
            continue
        target = st + a.horizon
        if pref_end < 0:
            lo = st + a.lead
        elif st >= pref_end - a.refill:
            lo = pref_end
        else:
            time.sleep(a.poll)
            continue
        hi = max(lo, target)
        if hi > lo:
            warm(lo, hi)
            pref_end = hi
        time.sleep(a.poll)


if __name__ == '__main__':
    main()

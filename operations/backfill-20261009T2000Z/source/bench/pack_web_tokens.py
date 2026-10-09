"""Concatenate pre-tokenised documents (uint16 ids ending in EOS) from strict-selected web outputs into
train/data.py shards: a training set and a held-out validation shard drawn from disjoint files.

  python -m bench.pack_web_tokens --root /data/run01/scxi253/web-strict-20261003/out --out DIR \
      --train-tokens 1000000000 --val-tokens 20000000
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

EOS = 50279


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-tokens', type=int, default=1_000_000_000)
    ap.add_argument('--val-tokens', type=int, default=20_000_000)
    ap.add_argument('--shard-tokens', type=int, default=250_000_000)
    a = ap.parse_args(argv)
    files = sorted(f for f in glob.glob(os.path.join(a.root, '**', '*.parquet'), recursive=True)
                   if os.path.exists(f + '.json'))
    os.makedirs(a.out, exist_ok=True)
    plan = [('val', a.val_tokens)] + [('train', a.train_tokens)]
    fi, stats = 0, {}
    for split, budget in plan:
        written, shard, buf, nbuf = 0, 0, [], 0

        def flush():
            nonlocal shard, buf, nbuf
            if buf:
                np.concatenate(buf).astype('<u2').tofile(os.path.join(a.out, f'{split}_{shard:04d}.u16'))
                shard += 1; buf, nbuf = [], 0
        while written < budget and fi < len(files):
            t = pq.read_table(files[fi], columns=['tokens'])['tokens'].to_pylist(); fi += 1
            for b in t:
                ids = np.frombuffer(b, dtype='<u2')
                assert ids[-1] == EOS
                buf.append(ids); nbuf += ids.size; written += ids.size
                if nbuf >= a.shard_tokens:
                    flush()
                if written >= budget:
                    break
        flush()
        stats[split] = dict(tokens=written, shards=shard, last_file_index=fi)
    json.dump(dict(stats, files_used=fi, root=a.root), open(os.path.join(a.out, 'PACK.json'), 'w'), indent=1)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()

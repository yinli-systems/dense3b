"""Synthetic packed shards for throughput / memory runs (random tokens, real vocab, EOS-separated documents).

  python -m bench.synth_data /local/synth --shards 8 --tokens-per-shard 20000000
"""
import argparse
from pathlib import Path

import numpy as np

from train.data import EOS, write_shard


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--shards', type=int, default=8)
    ap.add_argument('--tokens-per-shard', type=int, default=20_000_000)
    ap.add_argument('--vocab', type=int, default=50_280)
    ap.add_argument('--eos', type=int, default=EOS)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for i in range(a.shards):
        docs, n = [], 0
        while n < a.tokens_per_shard:
            L = int(rng.integers(64, 4096))
            docs.append(rng.integers(2, min(a.vocab, a.eos + 1), L, dtype=np.uint16))
            n += L + 1
        write_shard(out / f'synth_{i:04d}.u16', [d[d != a.eos] for d in docs], eos=a.eos)
    return sorted(map(str, out.glob('synth_*.u16')))


if __name__ == '__main__':
    print('\n'.join(main()))

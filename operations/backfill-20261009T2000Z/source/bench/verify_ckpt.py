"""Verify a train_dense checkpoint without a GPU: every model parameter is owned by exactly one writer rank,
masters are finite and match the BF16 model.pt, Muon/AdamW state is present and finite.

  python -m bench.verify_ckpt /data/run01/scxi253/code3b-train/ckpt-v0/step_00001499
"""
import glob
import json
import os
import sys

import torch

from model.config import get_config
from model.model import Transformer
from train.optim import param_groups


def main(path):
    meta = json.load(open(os.path.join(path, 'meta.json')))
    cfg = get_config(meta['args']['preset'], **json.loads(meta['args']['override']))
    with torch.device('meta'):
        m = Transformer(cfg)
    names = dict(m.named_parameters())                       # tied weights listed once
    kinds = {}
    for g in param_groups(m, meta['args']['optimizer'] == 'muon'):
        for p in g['params']:
            kinds[id(p)] = g['kind']
    kind = {n: kinds[id(p)] for n, p in names.items()}
    files = sorted(glob.glob(os.path.join(path, 'rank*.pt')))
    model = torch.load(os.path.join(path, 'model.pt'), map_location='cpu', mmap=True, weights_only=False)
    seen, problems, n_el = {}, [], 0
    for f in files:
        sd = torch.load(f, map_location='cpu', mmap=True, weights_only=False)
        for n, ent in sd.items():
            if n in seen:
                problems.append(f'{n} owned by {seen[n]} and {os.path.basename(f)}')
            seen[n] = os.path.basename(f)
            w = ent['master']
            if tuple(w.shape) != tuple(names[n].shape):
                problems.append(f'{n} shape {tuple(w.shape)} != {tuple(names[n].shape)}')
                continue
            if not torch.isfinite(w).all():
                problems.append(f'{n} master non-finite')
            diff = (w.bfloat16().float() - model[n].float()).abs().max().item()
            if diff > 0:
                problems.append(f'{n} master != model.pt (max abs diff {diff:.3g})')
            need = ['momentum'] if kind[n] == 'muon' else ['exp_avg', 'exp_avg_sq', 'step']
            for k in need:
                v = ent['state'].get(k)
                if v is None:
                    problems.append(f'{n} missing optimizer state {k}')
                elif torch.is_tensor(v) and not torch.isfinite(v).all():
                    problems.append(f'{n} optimizer state {k} non-finite')
            n_el += w.numel()
    missing = sorted(set(names) - set(seen))
    if missing:
        problems.append(f'{len(missing)} parameters owned by no rank, e.g. {missing[:3]}')
    out = dict(path=path, step=meta['step'], writers=meta.get('writers'), rank_files=len(files), params=len(names),
               owned=len(seen), elements=n_el, ok=not problems, problems=problems[:20])
    print(json.dumps(out, indent=1))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))

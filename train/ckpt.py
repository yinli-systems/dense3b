"""Per-rank checkpoints on node-local disk, written in a background thread, with an optional upload hook.

Layout: <dir>/step_XXXXXXXX/rank_RRRRR.pt (+ meta.json from rank 0). Every rank stores only its local
shards (FSDP shard of dense params, its EP experts, optimizer state, router biases), so a save costs no
network traffic; restarts must use the same (world, EP) layout. A step directory is complete when it
holds `world` rank files; files are written to *.tmp and renamed, so a crash never leaves a torn file.
`upload(path)` runs after the local write (e.g. copy to JuiceFS / object storage at a throttled rate so
it does not compete with gradient traffic on the shared RoCE link).
"""
import json
import os
import re
import threading
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def _cpu(obj):
    if isinstance(obj, DTensor):
        obj = obj.to_local()
    if isinstance(obj, torch.Tensor):
        return obj.detach().to('cpu', copy=True)
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


def save(root, step, model, opt, rank=0, world=1, extra=None, async_save=False, upload=None):
    d = Path(root) / f'step_{step:08d}'
    d.mkdir(parents=True, exist_ok=True)
    state = dict(model=_cpu(model.state_dict()), optim=_cpu(opt.state_dict()), step=step, extra=extra or {})
    if rank == 0:
        (d / 'meta.json').write_text(json.dumps(dict(step=step, world=world, extra=extra or {}), sort_keys=True))

    def write():
        path = d / f'rank_{rank:05d}.pt'
        tmp = path.with_suffix('.tmp')
        torch.save(state, tmp)
        with open(tmp, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if upload is not None:
            upload(path)

    if not async_save:
        write()
        return None
    t = threading.Thread(target=write, daemon=False)
    t.start()
    return t


def latest(root, world=1):
    steps = []
    for d in Path(root).glob('step_*'):
        m = re.fullmatch(r'step_(\d+)', d.name)
        if m and len(list(d.glob('rank_*.pt'))) == world:
            steps.append((int(m.group(1)), d))
    return max(steps)[1] if steps else None


def load(path, model, opt, rank=0):
    state = torch.load(Path(path) / f'rank_{rank:05d}.pt', map_location='cpu', weights_only=False)
    with torch.no_grad():
        for k, v in model.state_dict().items():
            t = v.to_local() if isinstance(v, DTensor) else v
            t.copy_(state['model'][k])
    opt.load_state_dict(state['optim'])
    return state['step'], state['extra']

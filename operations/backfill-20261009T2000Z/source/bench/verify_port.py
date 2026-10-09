"""Verify a converted checkpoint end to end: recompute the held-out CE on the exact token windows the original
training run evaluated (OLMo-core LMEvaluator, uint32 rows of 4096) and compare with the values it logged.

  python -m bench.verify_port CKPT HELDOUT_DIR   (HELDOUT_DIR: <domain>.bin + ref-heldout-stepN.json)
Exit 1 if any domain differs by more than 0.03 nats (bf16 tolerance)."""
import glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from bench.eval_suite import load

ckpt, hd = sys.argv[1], sys.argv[2]
ref = json.load(open(glob.glob(os.path.join(hd, 'ref-heldout-step*.json'))[0]))['ce']
model, meta = load(ckpt)
res = {}
with torch.no_grad():
    for dom, want in ref.items():
        rows = np.fromfile(os.path.join(hd, f'{dom}.bin'), dtype=np.uint32).reshape(-1, 4096).astype(np.int64)
        tot = n = 0.0
        for r in rows:
            ids = torch.from_numpy(r)[None].cuda()
            x = model.embed(ids)
            for blk in model.layers:
                x, _ = blk(x, model.cos, model.sin, None)
            h = model.head.norm(x)[0, :-1]
            logits = F.linear(h, model.head.weight.to(h.dtype)).float()
            tot += F.cross_entropy(logits, ids[0, 1:], reduction='sum').item(); n += ids.shape[1] - 1
        res[dom] = dict(port=round(tot / n, 4), reference=round(want, 4), diff=round(tot / n - want, 4))
        print(dom, res[dom], flush=True)
worst = max(abs(v['diff']) for v in res.values())
print('VERIFY', json.dumps(res), 'max_abs_diff', worst, flush=True)
print('VERIFY_OK' if worst < 0.03 else 'VERIFY_FAILED', flush=True)
sys.exit(0 if worst < 0.03 else 1)

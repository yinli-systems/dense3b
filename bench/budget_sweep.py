"""Per-layer step time / activation memory of dense3b blocks under compile's activation_memory_budget."""
import json, sys
from bench.dense_sweep import run
data = sys.argv[1]; out = {}
for budget in (1.0, 0.6, 0.4, 0.3, 0.2):
    r = {d: run(data, 4096, 1, True, True, d, False, budget=budget) for d in (2, 6)}
    if any('error' in v for v in r.values()):
        out[budget] = r; print(json.dumps({budget: r})[:600], flush=True); continue
    per = dict(t_layer_ms=round((r[6]['fwd_bwd_s'] - r[2]['fwd_bwd_s']) / 4 / 2 * 1000, 2),
               act_gb_layer=round((r[6]['mem_gb'] - r[2]['mem_gb']) / 4 - 3.24e9 / 28 * 12 / 1e9, 3), raw=r)
    out[budget] = per; print(json.dumps({budget: {k: v for k, v in per.items() if k != 'raw'}}), flush=True)
ck = {d: run(data, 4096, 1, True, True, d, True) for d in (2, 6)}
print(json.dumps({'full_block_ckpt': dict(t_layer_ms=round((ck[6]['fwd_bwd_s'] - ck[2]['fwd_bwd_s']) / 8 * 1000, 2),
                                         act_gb_layer=round((ck[6]['mem_gb'] - ck[2]['mem_gb']) / 4 - 3.24e9 / 28 * 12 / 1e9, 3))}), flush=True)

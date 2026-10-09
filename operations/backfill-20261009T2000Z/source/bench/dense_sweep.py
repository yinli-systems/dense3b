"""One-GPU throughput/memory sweep of full-width dense3b blocks, and a planner for the 28-layer model.

Each config runs train.train_dense at two depths (2 and 6 layers, same width) in its own process; the
difference gives the per-layer step time and activation memory without the embedding/LM-head constant.
The planner then picks, for a per-GPU memory budget, how many of the 28 blocks to recompute (fewest that
fit) and predicts node step time and MFU, including the measured once-per-step communication.

  python -m bench.dense_sweep --data '<synthetic shards glob>' --out results/sweep.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from model.config import get_config, param_counts
from train.train_dense import PEAK_TFLOPS, flops_per_token

CONFIGS = [  # name, seq, micro, compile, doc_mask[, fp8]
    ('s4096_compile_doc', 4096, 1, True, True),
    ('s4096_compile_doc_fp8', 4096, 1, True, True, True),
    ('s4096_eager_doc', 4096, 1, False, True),
    ('s4096_compile_sdpa', 4096, 1, True, False),
    ('s2048x2_compile_doc', 2048, 2, True, True),
]
DEPTHS = (2, 6)
BYTES_PER_PARAM_ZERO = 4          # resident BF16 params + BF16 grads on every rank
BYTES_PER_PARAM_OWNED = 8         # fp32 master + Muon momentum, owned share only


def run(data, seq, micro, compile_, doc, depth, ckpt_all, steps=6, fp8=False, budget=1.0):
    ov = dict(n_layers=depth, n_dense_layers=depth, doc_mask=doc)
    with tempfile.TemporaryDirectory() as d:
        m = Path(d) / 'm.jsonl'
        cmd = [sys.executable, '-m', 'train.train_dense', '--preset', 'dense3b', '--override', json.dumps(ov),
               '--data', data, '--seq-len', str(seq), '--micro-batch', str(micro), '--global-batch', str(2 * micro),
               '--steps', str(steps), '--warmup', '1', '--metrics', str(m), '--timing', '--log-every', '1000',
               '--ckpt-layers', str(depth if ckpt_all else 0)] + (['--compile'] if compile_ else []) + (['--fp8'] if fp8 else []) + ['--act-budget', str(budget)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return dict(error=r.stderr[-1500:])
        recs = [json.loads(l) for l in m.read_text().splitlines()][3:]
    return dict(step_s=statistics.median(x['step_s'] for x in recs),
                fwd_bwd_s=statistics.median(x['t_fwd_bwd'] for x in recs),
                mem_gb=max(x['mem_fwd_bwd_gb'] for x in recs), mem_step_gb=max(x['max_mem_gb'] for x in recs))


def plan(per_layer, budget_gb, n_layers=28, gpus=8, comm_gbs=9.0, global_batch=256):
    """Fewest recomputed blocks that fit `budget_gb`; predicted step time and MFU on one node."""
    c = get_config('dense3b')
    pc = param_counts(c)
    total = pc['total']
    owned = total / gpus + pc['embed']               # worst rank: also owns the embedding (+Adam second moment)
    fixed = total * BYTES_PER_PARAM_ZERO / 1e9 + owned * BYTES_PER_PARAM_OWNED / 1e9 + per_layer['const_mem_gb']
    best = None
    for k in range(n_layers + 1):                    # k blocks recomputed
        mem = fixed + (n_layers - k) * per_layer['act_stored_gb'] + k * per_layer['act_ckpt_gb']
        if mem <= budget_gb:
            t_micro = per_layer['const_s'] + (n_layers - k) * per_layer['t_stored_s'] + k * per_layer['t_ckpt_s']
            n_micro = global_batch // (gpus * per_layer['micro'])
            comm = 2 * total * 2 / 1e9 * (gpus - 1) / gpus / comm_gbs
            step = n_micro * t_micro + comm
            tps = global_batch * per_layer['seq'] / step
            fpt = flops_per_token(c, per_layer['seq'])
            mfu = tps / gpus * fpt / (per_layer['peak_tflops'] * 1e12)
            best = dict(ckpt_layers=k, mem_gb=round(mem, 2), step_s=round(step, 2), tok_per_s_node=round(tps),
                        tokens_per_day_node_B=round(tps * 86400 / 1e9, 2), mfu=round(mfu, 3))
            break
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budgets', default='23.0,31.0', help='usable GB per GPU (4090, 5090)')
    ap.add_argument('--configs', default='', help='comma-separated config names (default: all)')
    a = ap.parse_args(argv)
    import torch
    gpu = torch.cuda.get_device_name(0)
    peak = PEAK_TFLOPS.get(gpu, 165.2)
    results = dict(gpu=gpu, configs={})
    for name, seq, micro, comp, doc, *fp8 in [c for c in CONFIGS if not a.configs or c[0] in a.configs.split(',')]:
        fp8 = bool(fp8 and fp8[0])
        r = {}
        for depth in DEPTHS:
            for ck in (False, True):
                r[f'd{depth}_{"ckpt" if ck else "stored"}'] = run(a.data, seq, micro, comp, doc, depth, ck, fp8=fp8)
                print(json.dumps(dict(config=name, depth=depth, ckpt=ck, **r[f'd{depth}_{"ckpt" if ck else "stored"}'])),
                      flush=True)
        try:
            lo, hi = DEPTHS
            dl = hi - lo
            s_lo, s_hi, c_lo, c_hi = (r[f'd{lo}_stored'], r[f'd{hi}_stored'], r[f'd{lo}_ckpt'], r[f'd{hi}_ckpt'])
            # one optimizer step = 2 micro-batches here; per-micro numbers
            per = dict(seq=seq, micro=micro, peak_tflops=peak,
                       t_stored_s=(s_hi['fwd_bwd_s'] - s_lo['fwd_bwd_s']) / dl / 2,
                       t_ckpt_s=(c_hi['fwd_bwd_s'] - c_lo['fwd_bwd_s']) / dl / 2)
            per['const_s'] = s_lo['fwd_bwd_s'] / 2 - lo * per['t_stored_s']
            layer_params = (param_counts(get_config('dense3b'))['total'] - param_counts(get_config('dense3b'))['embed']) / 28
            pmem = layer_params * (BYTES_PER_PARAM_ZERO + BYTES_PER_PARAM_OWNED) / 1e9   # single GPU owns all
            per['act_stored_gb'] = (s_hi['mem_gb'] - s_lo['mem_gb']) / dl - pmem
            per['act_ckpt_gb'] = (c_hi['mem_gb'] - c_lo['mem_gb']) / dl - pmem
            per['const_mem_gb'] = 1.5                                                    # context + allocator slack
            per['fp8'] = fp8
            layer_flops = 6 * layer_params * seq * micro + 3 * 2 * 2 * 3072 * seq / 2 * seq * micro
            per['layer_mfu'] = layer_flops / per['t_stored_s'] / (peak * 1e12)
            per['plans'] = {f'{b}GB': plan(per, float(b)) for b in a.budgets.split(',')}
            r['per_layer'] = per
            print(json.dumps(dict(config=name, per_layer=per)), flush=True)
        except (KeyError, TypeError) as e:
            r['per_layer'] = dict(error=repr(e))
        results['configs'][name] = r
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()

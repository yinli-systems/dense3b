"""Day-1 grouped-GEMM benchmark and autotuner (one GPU).

Shapes are what one EP rank sees per micro-batch in the moe7b preset: 16 local experts, ~2,048 rows each
(4,096 tokens x 8 slots x 8 ranks / 128 experts), balanced and skewed (lognormal) row counts.
For each op it checks correctness against the reference, times every tile config, compares with the
per-expert cuBLAS loop and torch._grouped_mm, and writes the best configs to --out (use it via
MOE_GMM_CONFIG=<file>).

  python -m bench.gemm --out results/gmm_4090.json
"""
import argparse
import itertools
import json
import time

import torch

from model import grouped_gemm as G

FWD_SPACE = [dict(BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=s)
             for bm, bn, bk, w, s in itertools.product((64, 128), (64, 128), (32, 64), (4, 8), (2, 3, 4))]


def counts_for(E, rows, skew, gen):
    if not skew:
        return torch.full((E,), rows, dtype=torch.int64)
    w = torch.empty(E).log_normal_(0, 0.5, generator=gen)
    c = (w / w.sum() * rows * E).round().long()
    c[0] += rows * E - c.sum()
    return c.clamp(min=0)


def cuda_time(fn, iters):
    if not torch.cuda.is_available():               # CPU / Triton-interpreter dry run (tests)
        fn()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t) / iters
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters / 1e3


def ops(E, rows, d, f, skew, dev, dtype, gen):
    """(name, M-grouped?, inputs) for the six expert GEMMs of a forward+backward."""
    c = counts_for(E, rows, skew, gen)
    offs = c.cumsum(0).to(torch.int32).to(dev)
    M = int(c.sum())
    r = lambda *s: torch.randn(*s, device=dev, dtype=dtype, generator=None)
    x, h = r(M, d), r(M, f)
    w13, w2 = r(E, d, 2 * f) / d ** 0.5, r(E, f, d) / f ** 0.5
    gy, gh = r(M, d), r(M, 2 * f)
    return offs, M, [
        ('fwd_w13', 'fwd', (x, w13)), ('fwd_w2', 'fwd', (h, w2)),
        ('dx_w2', 'fwd', (gy, w2.transpose(1, 2))), ('dx_w13', 'fwd', (gh, w13.transpose(1, 2))),
        ('dw_w2', 'wgrad', (h, gy)), ('dw_w13', 'wgrad', (x, gh)),
    ]


def run_op(kind, args, offs, E):
    if kind == 'fwd':
        return G.grouped_mm_triton(args[0], args[1], offs)
    return G.grouped_wgrad_triton(args[0], args[1], offs, E)


def ref_op(kind, args, offs, E):
    if kind == 'fwd':
        return G.grouped_mm_ref(args[0], args[1], offs)
    return G.grouped_wgrad_ref(args[0], args[1], offs, E).float()


def flops(kind, args, M):
    if kind == 'fwd':
        return 2 * M * args[0].shape[1] * args[1].shape[2]
    return 2 * M * args[0].shape[1] * args[1].shape[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--experts', type=int, default=16)
    ap.add_argument('--rows', type=int, default=2048, help='mean rows per local expert')
    ap.add_argument('--d-model', type=int, default=2048)
    ap.add_argument('--d-expert', type=int, default=512)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--quick', action='store_true', help='4 configs only')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    ap_cuda = torch.cuda.is_available()
    dev, dtype = (torch.device('cuda'), torch.bfloat16) if ap_cuda else (torch.device('cpu'), torch.float32)
    gen = torch.Generator().manual_seed(0)
    E = a.experts
    space = FWD_SPACE[::12] if a.quick else FWD_SPACE
    res = dict(gpu=torch.cuda.get_device_name() if ap_cuda else 'cpu',
               capability=torch.cuda.get_device_capability() if ap_cuda else None,
               torch=torch.__version__, shapes=vars(a), cases={})
    totals = {'fwd': {}, 'wgrad': {}}
    for skew in (False, True):
        offs, M, oplist = ops(E, a.rows, a.d_model, a.d_expert, skew, dev, dtype, gen)
        case = res['cases']['skewed' if skew else 'balanced'] = {}
        for name, kind, args in oplist:
            ref = ref_op(kind, args, offs, E).float()
            fl = flops(kind, args, M)
            entry = case[name] = dict(gflop=fl / 1e9)
            best = None
            for cfg in space:
                tgt = G.BLOCK if kind == 'fwd' else G.BLOCK_WGRAD
                tgt.update(cfg)
                try:
                    out = run_op(kind, args, offs, E).float()
                    err = ((out - ref).norm() / ref.norm()).item()
                    if err > 2e-2:
                        raise RuntimeError(f'relative error {err:.3g}')
                    t = cuda_time(lambda: run_op(kind, args, offs, E), a.iters)
                except Exception as e:                    # out of shared memory, compile failure, wrong result
                    entry.setdefault('failed', []).append(dict(cfg=cfg, error=str(e)[:200]))
                    continue
                key = json.dumps(cfg, sort_keys=True)
                tot, n = totals[kind].get(key, (0.0, 0))
                totals[kind][key] = (tot + t, n + 1)
                if best is None or t < best[1]:
                    best = (cfg, t, err)
            if best:
                entry.update(best_cfg=best[0], triton_s=best[1], triton_tflops=fl / best[1] / 1e12, rel_err=best[2])
            entry['loop_s'] = cuda_time(lambda: ref_op(kind, args, offs, E), max(3, a.iters // 4))
            entry['loop_tflops'] = fl / entry['loop_s'] / 1e12
            if kind == 'fwd' and ap_cuda:
                try:
                    entry['torch_grouped_mm_s'] = cuda_time(
                        lambda: torch._grouped_mm(args[0], args[1], offs=offs), max(3, a.iters // 4))
                except Exception as e:
                    entry['torch_grouped_mm_error'] = str(e)[:200]
    # one config per kernel family: best total time among configs that ran every op of that family
    res['best'] = {}
    for k, v in totals.items():
        full = max((n for _, n in v.values()), default=0)
        ok = {key: tot for key, (tot, n) in v.items() if n == full}
        if ok:
            res['best'][k] = json.loads(min(ok, key=ok.get))
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        with open(a.out.replace('.json', '.config.json'), 'w') as f:
            json.dump(dict(fwd=res['best'].get('fwd', {}), wgrad=res['best'].get('wgrad', {})), f, indent=1)
    return res


if __name__ == '__main__':
    main()

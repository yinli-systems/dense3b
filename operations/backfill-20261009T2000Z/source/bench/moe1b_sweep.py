"""One-GPU checks for the 1B-active MoE: (1) local-expert kernels vs a PyTorch reference (values and every gradient),
(2) kernel-level timing of the expert path, (3) whole-model fwd+bwd throughput / memory per candidate config.

  python -m bench.moe1b_sweep --out results.json [--configs A,C,D,E] [--check-only]

Memory note: optimizer state is owner-sharded over the 8 training ranks, so a rank holds 2N (bf16 params) + 2N
(bf16 grads) + 8N/8 (fp32 master + Muon momentum, its 1/8 share) bytes. The sweep measures params + grads +
activations and adds the N-byte optimizer share analytically.
"""
import argparse
import json
import os
import time
import traceback
from dataclasses import replace

import torch
import torch.nn.functional as F

from model.config import MoEConfig, param_counts, get_config
from model.model import Transformer, init_weights
from model.moe_local import local_experts, local_experts_ref
from train.train_dense import flops_per_token, ckpt_indices

BASE = MoEConfig(tie_embeddings=True, fuse_proj=True, doc_mask=True, route_groups=0, max_seq_len=4096)
CANDIDATES = {
    'A': replace(BASE, d_model=1536, n_layers=24, n_heads=12, n_kv_heads=4, d_ff_dense=4096, n_experts=32, top_k=4,
                 d_expert=512, d_shared=1024),
    'B': replace(BASE, d_model=1536, n_layers=24, n_heads=12, n_kv_heads=4, d_ff_dense=4096, n_experts=32, top_k=8,
                 d_expert=512, d_shared=0),
    'C': replace(BASE, d_model=1536, n_layers=20, n_heads=12, n_kv_heads=4, d_ff_dense=4096, n_experts=64, top_k=8,
                 d_expert=384, d_shared=768),
    'D': replace(BASE, d_model=1280, n_layers=24, n_heads=10, n_kv_heads=2, d_ff_dense=3584, n_experts=64, top_k=8,
                 d_expert=384, d_shared=768),
    'E': replace(BASE, d_model=1536, n_layers=20, n_heads=12, n_kv_heads=4, d_ff_dense=4096, n_experts=48, top_k=6,
                 d_expert=512, d_shared=1024),
    'R3S': get_config('r3b_s'),
    'MR': get_config('moe1b_r'), 'DL': get_config('dense1b_l'), 'DS': get_config('dense1b_s'),
}
PEAK = float(os.environ.get('PEAK_TFLOPS', '165.2')) * 1e12


def check_kernels(dev):
    out = {}
    torch.manual_seed(0)
    for (N, d, E, k, f) in [(4096, 1536, 64, 8, 384), (8192, 1536, 32, 4, 512), (2048, 1280, 48, 6, 512)]:
        x = (torch.randn(N, d, device=dev) * 0.5).bfloat16().requires_grad_()
        logits = torch.randn(N, E, device=dev)
        idx = logits.topk(k, -1).indices
        gate = torch.softmax(torch.randn(N, k, device=dev), -1).requires_grad_()
        w13 = (torch.randn(E, d, 2 * f, device=dev) * d ** -0.5).bfloat16().requires_grad_()
        w2 = (torch.randn(E, f, d, device=dev) * f ** -0.5).bfloat16().requires_grad_()
        dout = torch.randn(N, d, device=dev).bfloat16()
        res = {}
        for name, fn in (('fast', local_experts), ('ref', local_experts_ref)):
            for t in (x, gate, w13, w2):
                t.grad = None
            y = fn(x, gate, idx, w13, w2)
            y.backward(dout)
            res[name] = [y.float()] + [t.grad.float() for t in (x, gate, w13, w2)]
        errs = {}
        for i, nm in enumerate(['out', 'dx', 'dgate', 'dw13', 'dw2']):
            a, b = res['fast'][i], res['ref'][i]
            errs[nm] = ((a - b).norm() / b.norm().clamp(min=1e-12)).item()
        out[f'N{N}_d{d}_E{E}_k{k}_f{f}'] = errs
        assert all(v < 2e-2 for v in errs.values()), errs
    return out


def fp8_eval(dev, c, n_tokens=16384):
    """Block-scaled FP8 expert path vs BF16: relative error of output and every gradient, and timing."""
    torch.manual_seed(0)
    d, E, k, f = c.d_model, c.n_experts, c.top_k, c.d_expert
    x = (torch.randn(n_tokens, d, device=dev)).bfloat16().requires_grad_()
    idx = torch.randn(n_tokens, E, device=dev).topk(k, -1).indices
    gate = torch.softmax(torch.randn(n_tokens, k, device=dev), -1).requires_grad_()
    w13 = (torch.randn(E, d, 2 * f, device=dev) * d ** -0.5).bfloat16().requires_grad_()
    w2 = (torch.randn(E, f, d, device=dev) * f ** -0.5).bfloat16().requires_grad_()
    dout = torch.randn(n_tokens, d, device=dev).bfloat16()
    res = {}
    for fp8 in (False, True):
        for t in (x, gate, w13, w2):
            t.grad = None
        y = local_experts(x, gate, idx, w13, w2, fp8=fp8)
        y.backward(dout)
        res[fp8] = [y.float()] + [t.grad.float() for t in (x, gate, w13, w2)]
    err = {nm: ((res[True][i] - res[False][i]).norm() / res[False][i].norm()).item()
           for i, nm in enumerate(['out', 'dx', 'dgate', 'dw13', 'dw2'])}
    times = {}
    for fp8 in (False, True):
        for _ in range(3):
            local_experts(x, gate, idx, w13, w2, fp8=fp8).backward(dout)
        torch.cuda.synchronize()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        for _ in range(10):
            y = local_experts(x, gate, idx, w13, w2, fp8=fp8)
        e1.record()
        for _ in range(10):
            local_experts(x, gate, idx, w13, w2, fp8=fp8).backward(dout)
        e2.record(); torch.cuda.synchronize()
        times['fp8' if fp8 else 'bf16'] = dict(fwd_ms=e0.elapsed_time(e1) / 10, fwd_bwd_ms=e1.elapsed_time(e2) / 10)
    return dict(rel_err=err, times=times)


def check_fused(dev):
    """(1) wgrad accumulation into a preallocated .grad equals autograd accumulation; (2) fused linear CE equals the
    checkpointed reference (loss, dh, dW)."""
    from model.model import LMHead
    torch.manual_seed(0)
    out = {}
    N, d, E, k, f = 4096, 1536, 64, 8, 384
    x = torch.randn(N, d, device=dev).bfloat16().requires_grad_()
    idx = torch.randn(N, E, device=dev).topk(k, -1).indices
    gate = torch.softmax(torch.randn(N, k, device=dev), -1).requires_grad_()
    w13 = torch.nn.Parameter((torch.randn(E, d, 2 * f, device=dev) * d ** -0.5).bfloat16())
    w2 = torch.nn.Parameter((torch.randn(E, f, d, device=dev) * f ** -0.5).bfloat16())
    dout = torch.randn(N, d, device=dev).bfloat16()
    grads = {}
    for mode in ('autograd', 'inplace'):
        w13.grad = None if mode == 'autograd' else torch.full_like(w13, 0.01)
        w2.grad = None if mode == 'autograd' else torch.full_like(w2, 0.01)
        for _ in range(2):
            local_experts(x, gate, idx, w13, w2).backward(dout)
        g13 = w13.grad.float() - (0.01 if mode == 'inplace' else 0)
        g2 = w2.grad.float() - (0.01 if mode == 'inplace' else 0)
        grads[mode] = (g13, g2)
    out['wgrad_inplace_rel_err'] = [((grads['inplace'][i] - grads['autograd'][i]).norm() / grads['autograd'][i].norm()).item()
                                    for i in range(2)]
    from model.config import get_config
    c = get_config('moe1b')
    head = LMHead(c).to(dev, torch.bfloat16)
    torch.nn.init.normal_(head.weight, std=0.02)
    h = torch.randn(2, 4096, c.d_model, device=dev).bfloat16().requires_grad_()
    t = torch.randint(0, 50000, (2, 4096), device=dev); t[0, :100] = -100
    res = {}
    for fused in (False, True):
        head.fused_ce = fused; h.grad = None; head.weight.grad = None
        loss = head(h, t); loss.backward()
        res[fused] = (loss.float(), h.grad.float(), head.weight.grad.float())
    out['fused_ce_rel_err'] = [((res[True][i] - res[False][i]).norm() / res[False][i].norm().clamp(min=1e-12)).item()
                               for i in range(3)]
    return out


def check_wgrad_balance(dev):
    """Skewed routing (one expert takes ~30% of pairs): balanced chunked wgrad == per-expert wgrad, and timing."""
    from model.moe_local import wgrad, wgrad_balanced, _route
    torch.manual_seed(0)
    N, d, E, k, f = 16384, 1536, 64, 8, 384
    logits = torch.randn(N, E, device=dev); logits[:, 0] += 3.0          # expert 0 is hot
    idx = logits.topk(k, -1).indices
    tok, order, pos, offs = _route(idx, E)
    counts = torch.diff(offs, prepend=offs.new_zeros(1))
    x = torch.randn(N, d, device=dev).bfloat16()
    dh = torch.randn(N * k, 2 * f, device=dev).bfloat16()
    out = {'hot_expert_share': (counts.max() / counts.sum()).item(), 'max_over_mean': (counts.max() / counts.float().mean()).item()}
    a = wgrad(x, dh, offs, E, x_idx=tok)
    b = wgrad_balanced(x, dh, offs, E, x_idx=tok)
    out['rel_err'] = ((a - b).norm() / a.norm()).item()
    for name, fn in (('per_expert', wgrad), ('balanced', wgrad_balanced)):
        for _ in range(3):
            fn(x, dh, offs, E, x_idx=tok)
        torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            fn(x, dh, offs, E, x_idx=tok)
        e1.record(); torch.cuda.synchronize()
        out[name + '_ms'] = e0.elapsed_time(e1) / 10
    # balanced routing for reference
    idx2 = torch.randn(N, E, device=dev).topk(k, -1).indices
    tok2, _, _, offs2 = _route(idx2, E)
    for name, fn in (('per_expert_uniform', wgrad), ('balanced_uniform', wgrad_balanced)):
        for _ in range(3):
            fn(x, dh, offs2, E, x_idx=tok2)
        torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            fn(x, dh, offs2, E, x_idx=tok2)
        e1.record(); torch.cuda.synchronize()
        out[name + '_ms'] = e0.elapsed_time(e1) / 10
    return out


def time_experts(dev, c, n_tokens):
    torch.manual_seed(0)
    d, E, k, f = c.d_model, c.n_experts, c.top_k, c.d_expert
    x = torch.randn(n_tokens, d, device=dev).bfloat16().requires_grad_()
    idx = torch.randn(n_tokens, E, device=dev).topk(k, -1).indices
    gate = torch.softmax(torch.randn(n_tokens, k, device=dev), -1).requires_grad_()
    w13 = (torch.randn(E, d, 2 * f, device=dev) * 0.02).bfloat16().requires_grad_()
    w2 = (torch.randn(E, f, d, device=dev) * 0.02).bfloat16().requires_grad_()
    dout = torch.randn(n_tokens, d, device=dev).bfloat16()
    for _ in range(3):
        local_experts(x, gate, idx, w13, w2).backward(dout)
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    n = 10
    ev[0].record()
    for _ in range(n):
        y = local_experts(x, gate, idx, w13, w2)
    ev[1].record()
    for _ in range(n):
        y = local_experts(x, gate, idx, w13, w2)
        y.backward(dout)
    ev[2].record()
    torch.cuda.synchronize()
    fwd = ev[0].elapsed_time(ev[1]) / n
    fb = ev[1].elapsed_time(ev[2]) / n
    flop = 2 * n_tokens * k * d * 3 * f
    return dict(fwd_ms=fwd, fwd_bwd_ms=fb, fwd_tflops=flop / fwd / 1e9, fwd_bwd_tflops=3 * flop / fb / 1e9)


def bench_model(dev, name, c, micro, seq, ckpt_layers, steps=8):
    torch.manual_seed(0)
    model = Transformer(c)
    init_weights(model, 0)
    model = model.to(dev, torch.bfloat16)
    model.cos, model.sin = model.cos.float(), model.sin.float()
    model.ckpt_layers = ckpt_indices(c.n_layers, ckpt_layers)
    if os.environ.get('FP8') == '1':
        from model.fp8_linear import convert_to_fp8
        convert_to_fp8(model)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    for blk in model.layers:
        blk.compile(dynamic=False)
    g = torch.Generator().manual_seed(1)
    tok = torch.randint(1, min(50000, c.vocab_size), (micro, seq + 1), generator=g)
    tok[torch.rand(tok.shape, generator=g) < 1 / 900] = c.eos_id       # ~900-token documents
    x, y = tok[:, :-1].to(dev), tok[:, 1:].to(dev)
    times = []
    torch.cuda.reset_peak_memory_stats(dev)
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.time()
        out = model(x, y)
        out['loss'].backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        for m in model.moe_layers():
            m.router.update_bias(m.router.load)
    dt = sorted(times[3:])[len(times[3:]) // 2]
    fpt = flops_per_token(c, seq)
    tps = micro * seq / dt
    pc = param_counts(c)
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    return dict(config=name, micro=micro, seq=seq, ckpt_layers=ckpt_layers, step_s=dt, tok_per_s=tps,
                mfu=tps * fpt / PEAK, gflop_per_tok=fpt / 1e9, total_B=pc['total'] / 1e9, active_B=pc['active'] / 1e9,
                peak_fwd_bwd_gb=peak, est_rank_gb=peak + pc['total'] / 1e9, loss=out['ce'].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--configs', default='A,C,D,E')
    p.add_argument('--micro', default='1,2')
    p.add_argument('--ckpt', default='0,6,12')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--skip-check', action='store_true')
    p.add_argument('--check-fused', action='store_true')
    p.add_argument('--check-wgrad', action='store_true')
    p.add_argument('--fp8', action='store_true', help='only the FP8 vs BF16 expert-path comparison')
    p.add_argument('--one', default=None, help='CONFIG,MICRO,CKPT: run one model benchmark in this process')
    a = p.parse_args()
    dev = torch.device('cuda')
    if a.one:
        n, micro, ck = a.one.split(',')
        try:
            seq = int(os.environ.get('SEQ', 4096))
            c = CANDIDATES[n]
            c = replace(c, max_seq_len=max(seq, c.max_seq_len))
            r = bench_model(dev, n, c, int(micro), seq, int(ck))
        except Exception as e:
            r = dict(config=n, micro=int(micro), ckpt_layers=int(ck), error=f'{type(e).__name__}: {str(e)[:200]}')
        print('RESULT ' + json.dumps(r), flush=True)
        return
    res = dict(gpu=torch.cuda.get_device_name(dev), torch=torch.__version__)
    if a.check_wgrad:
        res['wgrad_balance'] = check_wgrad_balance(dev)
        print(json.dumps(res, indent=1), flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)
        return
    if a.check_fused:
        res['fused'] = check_fused(dev)
        print(json.dumps(res, indent=1), flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)
        return
    if a.fp8:
        res['fp8'] = {n: fp8_eval(dev, CANDIDATES[n]) for n in a.configs.split(',')}
        print(json.dumps(res, indent=1), flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)
        return
    if not a.skip_check:
        res['kernel_check'] = check_kernels(dev)
        print(json.dumps(res), flush=True)
    if not a.check_only:
        res['experts'] = {n: time_experts(dev, CANDIDATES[n], 16384) for n in a.configs.split(',')}
        print(json.dumps(res['experts']), flush=True)
        torch.cuda.empty_cache()
        import subprocess, sys
        res['model'] = []
        for n in a.configs.split(','):
            for micro in map(int, a.micro.split(',')):
                for ck in map(int, a.ckpt.split(',')):
                    out = subprocess.run([sys.executable, '-m', 'bench.moe1b_sweep', '--out', '/dev/null',
                                          '--one', f'{n},{micro},{ck}'], capture_output=True, text=True)
                    line = [l for l in out.stdout.splitlines() if l.startswith('RESULT ')]
                    r = json.loads(line[-1][7:]) if line else dict(config=n, micro=micro, ckpt_layers=ck,
                                                                   error=out.stderr[-300:])
                    print(json.dumps(r), flush=True)
                    res['model'].append(r)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

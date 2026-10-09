"""Exhaustive Triton config search for the expert kernels at the production shapes (moe1b, 2 x 4096 tokens per
micro-batch, top-8 -> 65,536 (token, expert) pairs), plus attention: flex (doc mask) vs flash-attn varlen.

  MOE_GMM_WIDE=1 python -m bench.moe1b_kernels --out kern.json
"""
import argparse, json, time
import torch
import torch.nn.functional as F


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    import model.moe_local as ml
    dev = torch.device('cuda'); torch.manual_seed(0)
    N, d, E, k, f = 8192, 1536, 64, 8, 384
    idx = torch.randn(N, E, device=dev).topk(k, -1).indices
    tok, order, pos, offs = ml._route(idx, E)
    P = N * k
    x = torch.randn(N, d, device=dev).bfloat16()
    w13 = (torch.randn(E, d, 2 * f, device=dev) * 0.02).bfloat16()
    w2 = (torch.randn(E, f, d, device=dev) * 0.02).bfloat16()
    h = torch.randn(P, 2 * f, device=dev).bfloat16(); a_ = torch.randn(P, f, device=dev).bfloat16()
    dout = torch.randn(N, d, device=dev).bfloat16()
    gs = torch.rand(P, device=dev)
    res = {}
    cases = {
        'up_fwd   x[tok]@W13': lambda: ml.gmm(x, w13, offs, a_idx=tok),
        'down_fwd a@W2': lambda: ml.gmm(a_, w2, offs),
        'G2  dout[tok]@W2^T': lambda: ml.gmm(dout, w2, offs, a_idx=tok, trans_b=True),
        'dxp dh@W13^T': lambda: ml.gmm(h, w13, offs, trans_b=True),
        'dw13 x[tok]^T@dh': lambda: ml.wgrad(x, h, offs, E, x_idx=tok, out_dtype=torch.bfloat16),
        'dw2 (g*a)^T@dout[tok]': lambda: ml.wgrad(a_, dout, offs, E, g_idx=tok, x_scale=gs, out_dtype=torch.bfloat16),
    }
    flops = {'up_fwd   x[tok]@W13': 2 * P * d * 2 * f, 'down_fwd a@W2': 2 * P * f * d, 'G2  dout[tok]@W2^T': 2 * P * d * f,
             'dxp dh@W13^T': 2 * P * 2 * f * d, 'dw13 x[tok]^T@dh': 2 * P * d * 2 * f, 'dw2 (g*a)^T@dout[tok]': 2 * P * f * d}
    for name, fn in cases.items():
        t = timeit(fn)
        kern = ml._wgrad_kernel if name.startswith('dw') else ml._gmm_kernel
        best = {str(k_): str(v) for k_, v in getattr(kern, 'cache', {}).items()}
        res[name] = dict(ms=round(t, 3), tflops=round(flops[name] / t / 1e9, 1), best=best)
        print(name, res[name]['ms'], 'ms', res[name]['tflops'], 'TF', flush=True)
    # attention: flex with a document mask vs flash-attn varlen (same documents)
    B, T, H, KV, D = 2, 4096, 12, 4, 128
    q = torch.randn(B, H, T, D, device=dev).bfloat16().requires_grad_()
    kk = torch.randn(B, KV, T, D, device=dev).bfloat16().requires_grad_()
    v = torch.randn(B, KV, T, D, device=dev).bfloat16().requires_grad_()
    toks = torch.randint(0, 50000, (B, T), device=dev); toks[torch.rand(B, T, device=dev) < 1 / 900] = 50279
    from model.layers import doc_block_mask, flex_attention
    bm = doc_block_mask(toks)
    go = torch.randn(B, H, T, D, device=dev).bfloat16()

    def flex_fb():
        o = flex_attention(q, kk, v, bm, enable_gqa=True); o.backward(go)
    res['attn_flex_fwd_bwd_ms'] = round(timeit(flex_fb, 10), 3)
    try:
        from flash_attn import flash_attn_varlen_func
        is_eos = (toks == 50279).int(); doc = torch.cumsum(is_eos, 1) - is_eos
        lens = []
        for b in range(B):
            lens += torch.bincount(doc[b]).tolist()
        cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), device=dev, dtype=torch.int32)
        qv = q.detach().transpose(1, 2).reshape(B * T, H, D).requires_grad_()
        kv_ = kk.detach().transpose(1, 2).reshape(B * T, KV, D).requires_grad_()
        vv = v.detach().transpose(1, 2).reshape(B * T, KV, D).requires_grad_()
        gv = go.transpose(1, 2).reshape(B * T, H, D)

        def fa_fb():
            o = flash_attn_varlen_func(qv, kv_, vv, cu, cu, max(lens), max(lens), causal=True); o.backward(gv)
        res['attn_flash_varlen_fwd_bwd_ms'] = round(timeit(fa_fb, 10), 3)
        import flash_attn; res['flash_attn_version'] = flash_attn.__version__
    except Exception as e:
        res['flash_attn_error'] = f'{type(e).__name__}: {str(e)[:200]}'
    print(json.dumps(res, indent=1)); json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

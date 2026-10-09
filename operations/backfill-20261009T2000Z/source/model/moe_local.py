"""Routed experts on one device (no expert parallelism): sync-free sort -> gathered grouped GEMMs -> weighted combine.

Every rank holds all experts, so dispatch is a permutation instead of an all-to-all. The (token, expert) pairs are
sorted by expert with a fixed-size argsort; no .item()/nonzero/bincount host sync anywhere, so the whole layer is
CUDA-graph and torch.compile friendly and never stalls the CPU.

Memory: the gathered input x[tok] (N*k x d) is never materialised -- the up-projection kernel gathers rows while
loading A, and the weight gradient re-gathers them. Saved for backward: x, the sorted pair -> token map, the gates,
and the pre-activation h (N*k x 2f); the SwiGLU output is recomputed.

Forward  (pairs sorted by expert; tok[i] = token of pair i, g[i] = its gate)
  h   = x[tok] @ W13[e]                       (gathered grouped GEMM)
  a   = silu(h1) * h2
  y   = a @ W2[e]
  out[t] = sum_j g[t,j] * y[pos[t,j]]         (combine kernel, deterministic, no atomics)
Backward
  G2  = dout[tok] @ W2[e]^T                   (gathered grouped GEMM)
  dg  = rowsum(a * G2);  da = g * G2
  dW2 = (g*a)^T @ dout[tok]                   (gathered, row-scaled weight gradient)
  dh  = swiglu'(h) * da
  dxp = dh @ W13[e]^T;   dx[t] = sum_j dxp[pos[t,j]]
  dW13 = x[tok]^T @ dh
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _fwd_configs():
    import os
    if os.environ.get('MOE_GMM_WIDE') == '1':        # exhaustive search space (bench/moe1b_kernels.py)
        return [triton.Config({'BM': bm, 'BN': bn, 'BK': bk}, num_warps=w, num_stages=st)
                for bm in (64, 128, 256) for bn in (64, 128, 256) for bk in (32, 64) for w in (4, 8) for st in (2, 3, 4, 5)
                if bm * bn <= 128 * 256 and not (bm == 256 and bn == 256) and (bm * bk + bk * bn) * 2 * st <= 99 * 1024]
    cfgs = []
    # (64, 128, 32, 4, 4) won every expert GEMM in the exhaustive search at the moe1b shapes (122-160 TF on a 4090)
    for bm, bn, bk, w, s in [(64, 128, 32, 4, 4), (64, 128, 32, 4, 3), (128, 128, 32, 8, 4), (128, 128, 64, 8, 3), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4),
                             (128, 256, 32, 8, 3), (256, 128, 32, 8, 3), (64, 64, 64, 4, 5), (128, 128, 32, 4, 4)]:
        cfgs.append(triton.Config({'BM': bm, 'BN': bn, 'BK': bk}, num_warps=w, num_stages=s))
    return cfgs


def _wgrad_configs():
    import os
    if os.environ.get('MOE_GMM_WIDE') == '1':
        return [triton.Config({'BK': bk, 'BN': bn, 'BM': bm}, num_warps=w, num_stages=st)
                for bk in (64, 128, 256) for bn in (64, 128, 256) for bm in (32, 64) for w in (4, 8) for st in (2, 3, 4, 5)
                if bk * bn <= 128 * 256 and not (bk == 256 and bn == 256) and (bk * bm + bm * bn) * 2 * st <= 99 * 1024]
    cfgs = []
    for bk, bn, bm, w, s in [(64, 128, 32, 4, 5), (64, 128, 32, 4, 4), (128, 128, 32, 8, 4), (128, 128, 64, 8, 3), (64, 128, 64, 4, 4), (128, 64, 64, 4, 4),
                             (64, 64, 64, 4, 5), (128, 256, 32, 8, 3)]:
        cfgs.append(triton.Config({'BK': bk, 'BN': bn, 'BM': bm}, num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_fwd_configs(), key=['N', 'K', 'MB', 'GATHER', 'TRANS_B'])
@triton.jit
def _gmm_kernel(A, AIDX, W, Y, OFFS, N, K, MB, sam, sak, swe, swk, swn, sym, syn,
                E: tl.constexpr, EP2: tl.constexpr, GATHER: tl.constexpr, TRANS_B: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Y[rows of e] = A[(gathered) rows of e] @ W[e]; tile -> expert map computed on device from OFFS."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ar = tl.arange(0, EP2)
    end = tl.load(OFFS + ar, mask=ar < E, other=0)
    start = tl.load(OFFS + ar - 1, mask=(ar > 0) & (ar < E), other=0)
    tiles = tl.where(ar < E, (end - start + BM - 1) // BM, 0)
    tile_end = tl.cumsum(tiles, 0)
    e = tl.sum((tile_end <= pid_m).to(tl.int32), 0)
    if e < E:
        sel = ar == e
        t0 = tl.sum(tl.where(sel, tile_end - tiles, 0), 0)
        r0 = tl.sum(tl.where(sel, start, 0), 0)
        r1 = tl.sum(tl.where(sel, end, 0), 0)
        rm = r0 + (pid_m - t0) * BM + tl.arange(0, BM)
        mmask = rm < r1
        if GATHER:
            src = tl.load(AIDX + rm, mask=mmask, other=0).to(tl.int64)
        else:
            src = rm.to(tl.int64)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptr = A + src[:, None] * sam + rk[None, :] * sak
        b_ptr = W + e.to(tl.int64) * swe + rk[:, None] * swk + rn[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptr, mask=mmask[:, None], other=0.0)
            b = tl.load(b_ptr, mask=rn[None, :] < N, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptr += BK * sak
            b_ptr += BK * swk
        tl.store(Y + rm[:, None].to(tl.int64) * sym + rn[None, :] * syn, acc.to(Y.dtype.element_ty),
                 mask=mmask[:, None] & (rn[None, :] < N))


@triton.autotune(configs=_wgrad_configs(), key=['K', 'N', 'MB', 'XG', 'GG', 'XS', 'ACC'],
                 reset_to_zero=None, restore_value=['DW'])
@triton.jit
def _wgrad_kernel(X, XIDX, XSCALE, G, GIDX, DW, OFFS, K, N, MB, sxm, sxk, sgm, sgn, swe, swk, swn,
                  XG: tl.constexpr, GG: tl.constexpr, XS: tl.constexpr, ACC: tl.constexpr,
                  BK: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr):
    """DW[e] = (s * X[gathered rows of e])^T @ G[gathered rows of e], fp32 accumulation."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.program_id(2)
    r0 = tl.load(OFFS + e - 1, mask=e > 0, other=0)
    r1 = tl.load(OFFS + e)
    rk = pid_k * BK + tl.arange(0, BK)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BK, BN), dtype=tl.float32)
    for m0 in range(r0, r1, BM):
        rm = m0 + tl.arange(0, BM)
        mm = rm < r1
        xs = tl.load(XIDX + rm, mask=mm, other=0).to(tl.int64) if XG else rm.to(tl.int64)
        gs = tl.load(GIDX + rm, mask=mm, other=0).to(tl.int64) if GG else rm.to(tl.int64)
        x = tl.load(X + xs[None, :] * sxm + rk[:, None] * sxk, mask=mm[None, :] & (rk[:, None] < K), other=0.0)
        if XS:
            s = tl.load(XSCALE + rm, mask=mm, other=0.0)
            x = (x.to(tl.float32) * s[None, :]).to(X.dtype.element_ty)
        g = tl.load(G + gs[:, None] * sgm + rn[None, :] * sgn, mask=mm[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(x, g, acc)
    dw_ptr = DW + e.to(tl.int64) * swe + rk[:, None] * swk + rn[None, :] * swn
    dmask = (rk[:, None] < K) & (rn[None, :] < N)
    if ACC:
        acc += tl.load(dw_ptr, mask=dmask, other=0.0).to(tl.float32)
    tl.store(dw_ptr, acc.to(DW.dtype.element_ty), mask=dmask)


def _wgrad_chunk_configs():
    return [triton.Config({'BK': bk, 'BN': bn, 'BM': bm, 'CH': ch}, num_warps=w, num_stages=st)
            for bk, bn, bm, ch, w, st in [(128, 128, 64, 1024, 8, 3), (128, 128, 32, 1024, 8, 4), (64, 128, 64, 1024, 4, 4),
                                          (128, 128, 64, 2048, 8, 3), (128, 64, 64, 1024, 4, 4), (128, 128, 64, 512, 8, 3)]]


@triton.autotune(configs=_wgrad_chunk_configs(), key=['K', 'N', 'MB', 'XG', 'GG', 'XS', 'E'],
                 reset_to_zero=['DW32'])
@triton.jit
def _wgrad_chunk_kernel(X, XIDX, XSCALE, G, GIDX, DW32, OFFS, K, N, MB, sxm, sxk, sgm, sgn, swe, swk, swn,
                        E: tl.constexpr, EP2: tl.constexpr, XG: tl.constexpr, GG: tl.constexpr, XS: tl.constexpr,
                        BK: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr, CH: tl.constexpr):
    """Load-balanced weight gradient: program (chunk, k-tile, n-tile) handles <= CH rows of ONE expert and atomically
    adds its fp32 partial into DW32[e]. Work per program is bounded whatever the routing imbalance."""
    pid_c = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)
    ar = tl.arange(0, EP2)
    end = tl.load(OFFS + ar, mask=ar < E, other=0)
    start = tl.load(OFFS + ar - 1, mask=(ar > 0) & (ar < E), other=0)
    chunks = tl.where(ar < E, (end - start + CH - 1) // CH, 0)
    chunk_end = tl.cumsum(chunks, 0)
    e = tl.sum((chunk_end <= pid_c).to(tl.int32), 0)
    if e < E:
        sel = ar == e
        c0 = tl.sum(tl.where(sel, chunk_end - chunks, 0), 0)
        r_e0 = tl.sum(tl.where(sel, start, 0), 0)
        r_e1 = tl.sum(tl.where(sel, end, 0), 0)
        r0 = r_e0 + (pid_c - c0) * CH
        r1 = tl.minimum(r0 + CH, r_e1)
        rk = pid_k * BK + tl.arange(0, BK)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BK, BN), dtype=tl.float32)
        for m0 in range(r0, r1, BM):
            rm = m0 + tl.arange(0, BM)
            mm = rm < r1
            xs = tl.load(XIDX + rm, mask=mm, other=0).to(tl.int64) if XG else rm.to(tl.int64)
            gs = tl.load(GIDX + rm, mask=mm, other=0).to(tl.int64) if GG else rm.to(tl.int64)
            x = tl.load(X + xs[None, :] * sxm + rk[:, None] * sxk, mask=mm[None, :] & (rk[:, None] < K), other=0.0)
            if XS:
                sc = tl.load(XSCALE + rm, mask=mm, other=0.0)
                x = (x.to(tl.float32) * sc[None, :]).to(X.dtype.element_ty)
            g = tl.load(G + gs[:, None] * sgm + rn[None, :] * sgn, mask=mm[:, None] & (rn[None, :] < N), other=0.0)
            acc = tl.dot(x, g, acc)
        tl.atomic_add(DW32 + e.to(tl.int64) * swe + rk[:, None] * swk + rn[None, :] * swn, acc,
                      mask=(rk[:, None] < K) & (rn[None, :] < N), sem='relaxed')


@triton.jit
def _combine_kernel(Y, POS, GATE, OUT, D, sy, so, K: tl.constexpr, WEIGHTED: tl.constexpr, BD: tl.constexpr):
    """OUT[t] = sum_j GATE[t, j] * Y[POS[t, j]] (fp32 accumulation, fixed order: deterministic)."""
    t = tl.program_id(0)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    dm = rd < D
    acc = tl.zeros((BD,), dtype=tl.float32)
    for j in tl.static_range(K):
        p = tl.load(POS + t * K + j).to(tl.int64)
        y = tl.load(Y + p * sy + rd, mask=dm, other=0.0).to(tl.float32)
        if WEIGHTED:
            acc += tl.load(GATE + t * K + j) * y
        else:
            acc += y
    tl.store(OUT + t.to(tl.int64) * so + rd, acc.to(OUT.dtype.element_ty), mask=dm)


def _mb(m):
    """Autotune bucket for the row count (rows change every step; tiles only care about the scale)."""
    return 1 << max(10, (m - 1).bit_length())


def gmm(a, w, offs, a_idx=None, trans_b=False):
    """a (Ma, K) [rows gathered by a_idx (M,)], w (E, K, N) or (E, N, K) if trans_b; offs int32 (E,) -> (M, N)."""
    M = a_idx.shape[0] if a_idx is not None else a.shape[0]
    E = w.shape[0]
    if trans_b:
        N, K = w.shape[1], w.shape[2]
        swk, swn = w.stride(2), w.stride(1)
    else:
        K, N = w.shape[1], w.shape[2]
        swk, swn = w.stride(1), w.stride(2)
    assert K % 64 == 0 and N % 64 == 0, (K, N)
    y = a.new_empty(M, N)
    grid = lambda META: (triton.cdiv(M, META['BM']) + E, triton.cdiv(N, META['BN']))
    _gmm_kernel[grid](a, a_idx if a_idx is not None else offs, w, y, offs, N, K, _mb(M),
                      a.stride(0), a.stride(1), w.stride(0), swk, swn, y.stride(0), y.stride(1),
                      E=E, EP2=triton.next_power_of_2(E), GATHER=a_idx is not None, TRANS_B=trans_b)
    return y


def wgrad_balanced(x, g, offs, E, x_idx=None, g_idx=None, x_scale=None, out_dtype=torch.float32, out=None):
    """Same result as wgrad() but split over row chunks (fp32 atomics), so a hot expert does not serialise the
    kernel. out: add into this tensor in place."""
    K, N = x.shape[1], g.shape[1]
    M = (x_idx if x_idx is not None else g_idx if g_idx is not None else g).shape[0]
    dw32 = torch.zeros(E, K, N, device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['CH']) + E, triton.cdiv(K, META['BK']), triton.cdiv(N, META['BN']))
    dummy = offs
    _wgrad_chunk_kernel[grid](x, x_idx if x_idx is not None else dummy, x_scale if x_scale is not None else dummy,
                              g, g_idx if g_idx is not None else dummy, dw32, offs, K, N, _mb(M),
                              x.stride(0), x.stride(1), g.stride(0), g.stride(1), dw32.stride(0), dw32.stride(1),
                              dw32.stride(2), E=E, EP2=triton.next_power_of_2(E), XG=x_idx is not None,
                              GG=g_idx is not None, XS=x_scale is not None)
    if out is not None:
        out.add_(dw32.to(out.dtype)) if out.dtype != torch.float32 else out.add_(dw32)
        return out
    return dw32.to(out_dtype)


def wgrad(x, g, offs, E, x_idx=None, g_idx=None, x_scale=None, out_dtype=torch.float32, out=None):
    """dw (E, K, N) = sum over each expert's rows of (x_scale * x[x_idx])^T @ g[g_idx].
    out: add into this tensor in place (e.g. the parameter's gradient buffer) instead of allocating."""
    K, N = x.shape[1], g.shape[1]
    M = (x_idx if x_idx is not None else g_idx if g_idx is not None else g).shape[0]
    assert K % 64 == 0 and N % 64 == 0, (K, N)
    dw = out if out is not None else torch.empty(E, K, N, device=x.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(K, META['BK']), triton.cdiv(N, META['BN']), E)
    dummy = offs
    _wgrad_kernel[grid](x, x_idx if x_idx is not None else dummy, x_scale if x_scale is not None else dummy,
                        g, g_idx if g_idx is not None else dummy, dw, offs, K, N, _mb(M),
                        x.stride(0), x.stride(1), g.stride(0), g.stride(1), dw.stride(0), dw.stride(1), dw.stride(2),
                        XG=x_idx is not None, GG=g_idx is not None, XS=x_scale is not None, ACC=out is not None)
    return dw


def combine(y, pos, gate=None):
    """y (P, D) pair rows, pos (N, k) row of each (token, slot), gate (N, k) fp32 or None -> (N, D)."""
    N, k = pos.shape
    D = y.shape[1]
    out = y.new_empty(N, D)
    BD = 512
    _combine_kernel[(N, triton.cdiv(D, BD))](y, pos, gate if gate is not None else pos, out, D, y.stride(0),
                                             out.stride(0), K=k, WEIGHTED=gate is not None, BD=BD, num_warps=4)
    return out


@torch.compile(dynamic=False)
def _swiglu_fwd(h):
    f = h.shape[1] // 2
    return (F.silu(h[:, :f].float()) * h[:, f:].float()).to(h.dtype)


@torch.compile(dynamic=False)
def _swiglu_bwd(h, G2, gs):
    """-> a (recomputed SwiGLU output), dh, d(gate) for every sorted pair."""
    f = h.shape[1] // 2
    h1, h2, G2 = h[:, :f].float(), h[:, f:].float(), G2.float()
    s1 = torch.sigmoid(h1)
    silu = h1 * s1
    a = silu * h2
    dg = (a.to(h.dtype).float() * G2).sum(-1)
    da = G2 * gs[:, None]
    dh = torch.cat([da * h2 * (s1 * (1 + h1 * (1 - s1))), da * silu], -1).to(h.dtype)
    return a.to(h.dtype), dh, dg


def _route(idx, E):
    """Sort (token, slot) pairs by expert, sync-free. Returns tok (P,), order (P,), pos (N,k), offs (E,)."""
    N, k = idx.shape
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = (order // k).to(torch.int32)
    counts = torch.zeros(E, dtype=torch.int32, device=idx.device).index_add_(
        0, flat, torch.ones_like(flat, dtype=torch.int32))
    offs = torch.cumsum(counts, 0, dtype=torch.int32)
    pos = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=idx.device))
    return tok, order, pos.view(N, k).to(torch.int32), offs


def WGRAD(*args, **kw):
    # per-expert wgrad measured faster even at 7.7x routing imbalance (2.14 vs 2.56 ms); chunked version is opt-in
    import os
    return (wgrad_balanced if os.environ.get('MOE_WGRAD') == 'balanced' else wgrad)(*args, **kw)


def _grad_buffer(p):
    """A leaf parameter whose .grad is a preallocated contiguous buffer (owner-ZeRO's flat gradient views) gets its
    weight gradient added in place by the wgrad kernel; autograd then receives None and does no extra add."""
    g = getattr(p, 'grad', None)
    if (p.requires_grad and p.is_leaf and g is not None and g.dtype == p.dtype and g.is_contiguous()
            and not torch.is_grad_enabled()):
        return g
    return None


class LocalExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, idx, w13, w2):
        E, d, f2 = w13.shape
        f = f2 // 2
        tok, order, pos, offs = _route(idx, E)
        gs = gate.reshape(-1)[order].float().contiguous()                 # gates in pair (sorted) order
        w13c, w2c = w13.to(x.dtype), w2.to(x.dtype)
        h = gmm(x, w13c, offs, a_idx=tok)                                  # (P, 2f)
        a = _swiglu_fwd(h)
        y = gmm(a, w2c, offs)                                              # (P, d)
        out = combine(y, pos, gate.float().contiguous())
        ctx.save_for_backward(x, h, tok, order, pos, offs, gs, w13c, w2c)
        ctx.shapes = (E, f, idx.shape, w13.dtype, w2.dtype)
        ctx.params = (w13, w2)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, h, tok, order, pos, offs, gs, w13c, w2c = ctx.saved_tensors
        E, f, idx_shape, w13_dtype, w2_dtype = ctx.shapes
        dout = dout.contiguous()
        G2 = gmm(dout, w2c, offs, a_idx=tok, trans_b=True)                # (P, f) = dout[tok] @ W2^T
        a, dh, dg_sorted = _swiglu_bwd(h, G2, gs)
        dgate = torch.empty_like(dg_sorted).scatter_(0, order, dg_sorted).view(idx_shape)
        w13, w2 = ctx.params
        acc2, acc13 = _grad_buffer(w2), _grad_buffer(w13)
        dw2 = WGRAD(a, dout, offs, E, g_idx=tok, x_scale=gs, out_dtype=w2_dtype, out=acc2)
        dxp = gmm(dh, w13c, offs, trans_b=True)                           # (P, d)
        dx = combine(dxp, pos)
        dw13 = WGRAD(x, dh, offs, E, x_idx=tok, out_dtype=w13_dtype, out=acc13)
        for p_, acc in ((w2, acc2), (w13, acc13)):
            hook = getattr(p_, '_grad_ready_hook', None) if acc is not None else None
            if hook is not None:
                hook(p_)
        return dx, dgate, None, None if acc13 is not None else dw13, None if acc2 is not None else dw2


def local_experts(x, gate, idx, w13, w2, fp8=None):
    """x (N, d) bf16, gate (N, k), idx (N, k) int64, w13 (E, d, 2f), w2 (E, f, d) -> (N, d).
    fp8=None reads MOE_FP8 (1 = block-scaled FP8 forward/input-gradient GEMMs)."""
    import os
    if fp8 is None:
        fp8 = os.environ.get('MOE_FP8') == '1'
    return (LocalExpertsFP8 if fp8 else LocalExperts).apply(x, gate, idx, w13, w2)


def local_experts_ref(x, gate, idx, w13, w2):
    """Plain PyTorch reference (loops over experts; for tests)."""
    N, k = idx.shape
    E, d, f2 = w13.shape
    f = f2 // 2
    out = torch.zeros(N, d, dtype=torch.float32, device=x.device)
    for e in range(E):
        t, j = (idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        hh = x[t] @ w13[e].to(x.dtype)
        aa = F.silu(hh[:, :f]) * hh[:, f:]
        out.index_add_(0, t, (aa @ w2[e].to(x.dtype)).float() * gate[t, j].float()[:, None])
    return out.to(x.dtype)


class LocalExpertsFP8(torch.autograd.Function):
    """Same math as LocalExperts with the forward and input-gradient GEMMs in block-scaled FP8 (model/fp8_gmm.py).
    Weight gradients stay BF16. Expert weights are quantised on the fly in forward and again in backward (no cache:
    a cached FP8 copy of all experts would cost ~2 GB per GPU)."""

    @staticmethod
    def forward(ctx, x, gate, idx, w13, w2):
        from .fp8_gmm import quant_rows, quant_weight, gmm_fp8
        E, d, f2 = w13.shape
        tok, order, pos, offs = _route(idx, E)
        gs = gate.reshape(-1)[order].float().contiguous()
        w13c, w2c = w13.to(x.dtype), w2.to(x.dtype)
        xq, xs = quant_rows(x, tok)
        w13q, w13s = quant_weight(w13c)
        h = gmm_fp8(xq, xs, w13q, w13s, offs)
        del xq, xs, w13q, w13s
        a = _swiglu_fwd(h)
        aq, as_ = quant_rows(a)
        w2q, w2s = quant_weight(w2c)
        y = gmm_fp8(aq, as_, w2q, w2s, offs)
        out = combine(y, pos, gate.float().contiguous())
        ctx.save_for_backward(x, h, tok, order, pos, offs, gs, w13c, w2c)
        ctx.shapes = (E, idx.shape, w13.dtype, w2.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        from .fp8_gmm import quant_rows, quant_weight, gmm_fp8
        x, h, tok, order, pos, offs, gs, w13c, w2c = ctx.saved_tensors
        E, idx_shape, w13_dtype, w2_dtype = ctx.shapes
        dout = dout.contiguous()
        dq, ds = quant_rows(dout, tok)
        w2q, w2s = quant_weight(w2c)
        G2 = gmm_fp8(dq, ds, w2q, w2s, offs, trans_b=True)
        del dq, ds, w2q, w2s
        a, dh, dg_sorted = _swiglu_bwd(h, G2, gs)
        dgate = torch.empty_like(dg_sorted).scatter_(0, order, dg_sorted).view(idx_shape)
        dw2 = wgrad(a, dout, offs, E, g_idx=tok, x_scale=gs, out_dtype=w2_dtype)
        hq, hs = quant_rows(dh)
        w13q, w13s = quant_weight(w13c)
        dxp = gmm_fp8(hq, hs, w13q, w13s, offs, trans_b=True)
        del hq, hs, w13q, w13s
        dx = combine(dxp, pos)
        dw13 = wgrad(x, dh, offs, E, x_idx=tok, out_dtype=w13_dtype)
        return dx, dgate, None, dw13, dw2

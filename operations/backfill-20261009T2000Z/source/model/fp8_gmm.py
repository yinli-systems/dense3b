"""Block-scaled FP8 (e4m3) grouped GEMMs for the local experts, DeepSeek-V3 style, for sm_89 (330 TF FP8 vs 165 TF BF16).

Scaling granularity (amax / 448 per block):
  activations  1 x 128 along K (per row, per 128 columns)        -> a_q (M, K) fp8, a_s (M, K/128) fp32
  weights      128 x 128 blocks of every expert matrix (E, K, N)  -> w_q (E, K, N) fp8, w_s (E, K/128, N/128)
A 128 x 128 block has the same amax transposed, so one quantised copy of each weight serves both the forward GEMM
(x @ W) and the input-gradient GEMM (g @ W^T). Weights are quantised once per optimizer step (cached on the
parameter version). Every K-block partial product is rescaled in fp32 before accumulation (no fp8 accumulation
error). Weight gradients stay BF16 (they reduce over tokens, where 1 x 128 scaling does not apply).
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
QB = 128


@triton.jit
def _quant_rows_kernel(X, XQ, XS, IDX, M, K, sxm, sqm, ssm, GATHER: tl.constexpr, QB_: tl.constexpr, BM: tl.constexpr):
    """Per (row, 128-column block): scale = amax/448; optionally gathers rows (XQ row i = X row IDX[i])."""
    pid_m = tl.program_id(0)
    kb = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    mm = rm < M
    src = tl.load(IDX + rm, mask=mm, other=0).to(tl.int64) if GATHER else rm.to(tl.int64)
    rk = kb * QB_ + tl.arange(0, QB_)
    x = tl.load(X + src[:, None] * sxm + rk[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    s = tl.maximum(amax, 1e-12) / 448.0
    q = (x / s[:, None]).to(XQ.dtype.element_ty)
    tl.store(XQ + rm[:, None].to(tl.int64) * sqm + rk[None, :], q, mask=mm[:, None])
    tl.store(XS + rm.to(tl.int64) * ssm + kb, s, mask=mm)


@triton.jit
def _quant_w_kernel(W, WQ, WS, K, N, swe, swk, sse, ssk, QB_: tl.constexpr):
    e = tl.program_id(2)
    kb = tl.program_id(0)
    nb = tl.program_id(1)
    rk = kb * QB_ + tl.arange(0, QB_)
    rn = nb * QB_ + tl.arange(0, QB_)
    off = e.to(tl.int64) * swe + rk[:, None] * swk + rn[None, :]
    w = tl.load(W + off).to(tl.float32)
    s = tl.maximum(tl.max(tl.max(tl.abs(w), 1), 0), 1e-12) / 448.0
    tl.store(WQ + off, (w / s).to(WQ.dtype.element_ty))
    tl.store(WS + e.to(tl.int64) * sse + kb * ssk + nb, s)


def quant_rows(x, idx=None):
    M = idx.shape[0] if idx is not None else x.shape[0]
    K = x.shape[1]
    assert K % QB == 0
    xq = torch.empty(M, K, device=x.device, dtype=FP8)
    xs = torch.empty(M, K // QB, device=x.device, dtype=torch.float32)
    BM = 32
    _quant_rows_kernel[(triton.cdiv(M, BM), K // QB)](x, xq, xs, idx if idx is not None else xs, M, K, x.stride(0),
                                                       xq.stride(0), xs.stride(0), GATHER=idx is not None, QB_=QB, BM=BM,
                                                       num_warps=4)
    return xq, xs


def quant_weight(w):
    E, K, N = w.shape
    assert K % QB == 0 and N % QB == 0 and w.stride(2) == 1
    wq = torch.empty(E, K, N, device=w.device, dtype=FP8)
    ws = torch.empty(E, K // QB, N // QB, device=w.device, dtype=torch.float32)
    _quant_w_kernel[(K // QB, N // QB, E)](w, wq, ws, K, N, w.stride(0), w.stride(1), ws.stride(0), ws.stride(1),
                                          QB_=QB, num_warps=8)
    return wq, ws


def _configs():
    return [triton.Config({'BM': bm, 'BN': bn}, num_warps=w, num_stages=s)
            for bm, bn, w, s in [(128, 128, 8, 3), (128, 128, 4, 4), (64, 128, 4, 4), (128, 64, 4, 4), (128, 256, 8, 3),
                                 (256, 128, 8, 3), (64, 64, 4, 5)]]


@triton.autotune(configs=_configs(), key=['N', 'K', 'MB', 'TRANS_B'])
@triton.jit
def _gmm_fp8_kernel(A, AS, W, WS, Y, OFFS, N, K, MB, sam, sas, swe, swk, swn, sse, ssk, ssn, sym,
                    E: tl.constexpr, EP2: tl.constexpr, TRANS_B: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Y[rows of e] = sum_kb (A_q[rows, kb] @ W_q[e, kb, :]) * a_s[rows, kb] * w_s[e, kb, nb]  (BN <= 128 per nb:
    BN is a multiple/divisor of 128, w_s is looked up per output column block)."""
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
        rm64 = rm.to(tl.int64)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        nb = rn // 128
        a_ptr = A + rm64[:, None] * sam + rk[None, :]
        b_ptr = W + e.to(tl.int64) * swe + rk[:, None] * swk + rn[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, K // BK):
            a = tl.load(a_ptr, mask=mmask[:, None], other=0.0)
            b = tl.load(b_ptr, mask=rn[None, :] < N, other=0.0)
            sa = tl.load(AS + rm64 * sas + kb, mask=mmask, other=0.0)
            if TRANS_B:
                sb = tl.load(WS + e.to(tl.int64) * sse + nb * ssk + kb * ssn, mask=rn < N, other=0.0)
            else:
                sb = tl.load(WS + e.to(tl.int64) * sse + kb * ssk + nb * ssn, mask=rn < N, other=0.0)
            acc += tl.dot(a, b) * sa[:, None] * sb[None, :]
            a_ptr += BK
            b_ptr += BK * swk
        tl.store(Y + rm64[:, None] * sym + rn[None, :], acc.to(Y.dtype.element_ty), mask=mmask[:, None] & (rn[None, :] < N))


def gmm_fp8(aq, as_, wq, ws, offs, trans_b=False, out_dtype=torch.bfloat16):
    """aq (M, K) fp8 + as_ (M, K/128); wq (E, K, N) [or used transposed: (E, N, K) -> K, N swapped] + ws."""
    M = aq.shape[0]
    E = wq.shape[0]
    if trans_b:
        N, K = wq.shape[1], wq.shape[2]
        swk, swn = wq.stride(2), wq.stride(1)
        sse, ssk, ssn = ws.stride(0), ws.stride(1), ws.stride(2)     # ws (E, N/128, K/128): index [e, nb, kb]
    else:
        K, N = wq.shape[1], wq.shape[2]
        swk, swn = wq.stride(1), wq.stride(2)
        sse, ssk, ssn = ws.stride(0), ws.stride(1), ws.stride(2)     # ws (E, K/128, N/128): index [e, kb, nb]
    assert K % QB == 0
    y = torch.empty(M, N, device=aq.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(M, META['BM']) + E, triton.cdiv(N, META['BN']))
    mb = 1 << max(10, (M - 1).bit_length())
    _gmm_fp8_kernel[grid](aq, as_, wq, ws, y, offs, N, K, mb, aq.stride(0), as_.stride(0), wq.stride(0), swk, swn,
                          sse, ssk, ssn, y.stride(0), E=E, EP2=triton.next_power_of_2(E), TRANS_B=trans_b, BK=QB)
    return y

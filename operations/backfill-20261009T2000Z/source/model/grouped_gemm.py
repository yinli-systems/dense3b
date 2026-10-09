"""Sync-free grouped GEMM for MoE experts: y[rows of e] = x[rows of e] @ w[e].

`offs` is the int32 inclusive cumsum of rows per expert (like torch._grouped_mm). On sm_89 / sm_120
torch._grouped_mm falls back to a per-expert loop that syncs the device, so CUDA tensors use the Triton
kernels below: the tile -> expert map is computed on device and the grid is a host-side upper bound
(cdiv(M, BM) + E tiles), so no .item()/.tolist() is ever needed. Everything else uses the reference.
"""
import json
import os
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None


def grouped_mm_ref(x, w, offs):
    """Reference: per-expert matmul (host sync via tolist). x (M,K), w (E,K,N), offs (E,) -> (M,N)."""
    out = x.new_empty(x.shape[0], w.shape[2])
    start = 0
    for e, end in enumerate(offs.tolist()):
        out[start:end] = x[start:end] @ w[e]
        start = end
    return out


def grouped_wgrad_ref(x, g, offs, E):
    """Reference: dw[e] = x[rows_e]^T @ g[rows_e]."""
    out = x.new_zeros(E, x.shape[1], g.shape[1])
    start = 0
    for e, end in enumerate(offs.tolist()):
        out[e] = x[start:end].T @ g[start:end]
        start = end
    return out


if triton is not None:
    @triton.jit
    def _gmm_kernel(X, W, Y, TILE_END, OFFS, N, K, sxm, sxk, swe, swk, swn, sym, syn,
                    E: tl.constexpr, EP2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        ar = tl.arange(0, EP2)
        tile_end = tl.load(TILE_END + ar, mask=ar < E, other=2 ** 30)
        e = tl.sum((tile_end <= pid_m).to(tl.int32), axis=0)       # expert owning this M tile
        if e < E:
            tile_start = tl.load(TILE_END + e - 1, mask=e > 0, other=0)
            row_start = tl.load(OFFS + e - 1, mask=e > 0, other=0)
            row_end = tl.load(OFFS + e)
            rm = row_start + (pid_m - tile_start) * BM + tl.arange(0, BM)
            rn = pid_n * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, K, BK):
                kk = k0 + rk
                a = tl.load(X + rm[:, None] * sxm + kk[None, :] * sxk,
                            mask=(rm[:, None] < row_end) & (kk[None, :] < K), other=0.0)
                b = tl.load(W + e * swe + kk[:, None] * swk + rn[None, :] * swn,
                            mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
                acc += tl.dot(a, b)
            tl.store(Y + rm[:, None] * sym + rn[None, :] * syn, acc.to(Y.dtype.element_ty),
                     mask=(rm[:, None] < row_end) & (rn[None, :] < N))

    @triton.jit
    def _gmm_wgrad_kernel(X, G, DW, OFFS, K, N, sxm, sxk, sgm, sgn, swe, swk, swn,
                          BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        pid_k = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.program_id(2)
        start = tl.load(OFFS + e - 1, mask=e > 0, other=0)
        end = tl.load(OFFS + e)
        rk = pid_k * BK + tl.arange(0, BK)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BK, BN), dtype=tl.float32)
        for m0 in range(start, end, BM):
            rm = m0 + tl.arange(0, BM)
            a = tl.load(X + rm[None, :] * sxm + rk[:, None] * sxk,
                        mask=(rm[None, :] < end) & (rk[:, None] < K), other=0.0)
            b = tl.load(G + rm[:, None] * sgm + rn[None, :] * sgn,
                        mask=(rm[:, None] < end) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(DW + e * swe + rk[:, None] * swk + rn[None, :] * swn, acc.to(DW.dtype.element_ty),
                 mask=(rk[:, None] < K) & (rn[None, :] < N))


# Tile configs; bench/gemm.py autotunes them per GPU and writes a JSON file that MOE_GMM_CONFIG points to.
BLOCK = dict(BM=64, BN=128, BK=64, num_warps=4, num_stages=3)          # forward / dX (M-grouped)
BLOCK_WGRAD = dict(BM=64, BN=128, BK=64, num_warps=4, num_stages=3)    # dW (K-grouped)
if os.environ.get('MOE_GMM_CONFIG'):
    with open(os.environ['MOE_GMM_CONFIG']) as _f:
        _cfg = json.load(_f)
    BLOCK.update(_cfg.get('fwd', {}))
    BLOCK_WGRAD.update(_cfg.get('wgrad', {}))


def _launch_opts(b):
    return dict(num_warps=b['num_warps'], num_stages=b['num_stages'])


def grouped_mm_triton(x, w, offs):
    M, K = x.shape
    E, _, N = w.shape
    BM, BN, BK = BLOCK['BM'], BLOCK['BN'], BLOCK['BK']
    counts = torch.diff(offs, prepend=offs.new_zeros(1))
    tile_end = torch.cumsum((counts + BM - 1) // BM, 0).to(torch.int32)
    y = x.new_empty(M, N)
    grid = (triton.cdiv(M, BM) + E, triton.cdiv(N, BN))
    _gmm_kernel[grid](x, w, y, tile_end, offs, N, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                      w.stride(2), y.stride(0), y.stride(1), E=E, EP2=triton.next_power_of_2(E), BM=BM, BN=BN, BK=BK,
                      **_launch_opts(BLOCK))
    return y


def grouped_wgrad_triton(x, g, offs, E):
    K, N = x.shape[1], g.shape[1]
    BM, BN, BK = BLOCK_WGRAD['BM'], BLOCK_WGRAD['BN'], BLOCK_WGRAD['BK']
    dw = torch.empty(E, K, N, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(K, BK), triton.cdiv(N, BN), E)
    _gmm_wgrad_kernel[grid](x, g, dw, offs, K, N, x.stride(0), x.stride(1), g.stride(0), g.stride(1),
                            dw.stride(0), dw.stride(1), dw.stride(2), BM=BM, BK=BK, BN=BN, **_launch_opts(BLOCK_WGRAD))
    return dw


def _use_triton(x):
    return triton is not None and (x.is_cuda or os.environ.get('TRITON_INTERPRET') == '1') \
        and os.environ.get('MOE_GMM', 'triton') == 'triton'


class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offs):
        ctx.save_for_backward(x, w, offs)
        return grouped_mm_triton(x, w, offs) if _use_triton(x) else grouped_mm_ref(x, w, offs)

    @staticmethod
    def backward(ctx, g):
        x, w, offs = ctx.saved_tensors
        g = g.contiguous()
        tri = _use_triton(g)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            wt = w.transpose(1, 2)
            dx = grouped_mm_triton(g, wt, offs) if tri else grouped_mm_ref(g, wt, offs)
        if ctx.needs_input_grad[1]:
            dw = grouped_wgrad_triton(x, g, offs, w.shape[0]) if tri else grouped_wgrad_ref(x, g, offs, w.shape[0])
            dw = dw.to(w.dtype)
        return dx, dw, None


def grouped_mm(x, w, offs):
    """Differentiable grouped GEMM. x (M,K) rows sorted by expert, w (E,K,N), offs int32 (E,)."""
    return _GroupedMM.apply(x, w, offs.to(torch.int32))

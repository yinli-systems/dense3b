"""Router (sigmoid + aux-loss-free bias, group-limited), dropless dispatch/combine with optional per-rank token
dedup, and the MoE layer (shared + routed experts).

Dispatch with dedup sends each token at most once to every EP rank that hosts >= 1 of its experts, together
with a small fp32 meta row (gate weight + local expert id per top-k slot). The receiving rank expands the row
to its local experts, runs the grouped GEMM, sums the gate-weighted outputs per (token, rank) and sends one
row back. Without dedup every (token, expert) pair travels separately (the classic layout).
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .fp8 import FP8Policy, quant_rowwise, dequant_rowwise
from .grouped_gemm import grouped_mm
from .layers import SwiGLU


def _a2a(x, out_splits, in_splits, group):
    out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits, group=group)
    return out


class AllToAll(torch.autograd.Function):
    """Differentiable variable-split all-to-all; identity when group is None. Optional FP8 forward payload."""

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group, fp8=False):
        ctx.args = (in_splits, out_splits, group)
        if fp8:
            q, inv = quant_rowwise(x)
            if group is not None:
                q = _a2a(q.view(torch.uint8), out_splits, in_splits, group).view(torch.float8_e4m3fn)
                inv = _a2a(inv, out_splits, in_splits, group)
            return dequant_rowwise(q, inv, x.dtype)
        return x.clone() if group is None else _a2a(x, out_splits, in_splits, group)

    @staticmethod
    def backward(ctx, g):
        in_splits, out_splits, group = ctx.args
        g = g if group is None else _a2a(g, in_splits, out_splits, group)  # gradient is sent in BF16
        return g, None, None, None, None


class Router(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.E, self.k = c.n_experts, c.top_k
        self.kind = getattr(c, 'router_kind', 'sigmoid_bias')
        self.G, self.M = c.route_groups, c.route_topk_groups
        self.z_coef, self.aux_coef, self.rate = c.z_loss_coef, c.seq_aux_coef, c.bias_update_rate
        self.weight = nn.Parameter(torch.empty(c.n_experts, c.d_model))
        self.register_buffer('bias', torch.zeros(c.n_experts))                 # persistent: checkpointed
        self.register_buffer('load', torch.zeros(c.n_experts), persistent=False)  # counts since last update

    def forward(self, x, n_seq):
        logits = F.linear(x.float(), self.weight.float())          # router math in fp32
        if self.kind == 'softmax':                                  # OLMo-core: softmax over experts, raw top-k scores
            scores = logits.softmax(-1)
            w, idx = scores.topk(self.k, -1)
            return idx, w, logits.new_zeros(())
        scores = logits.sigmoid()
        sel = scores.detach() + self.bias                           # bias steers selection only
        if self.G:
            N = sel.shape[0]
            g = sel.view(N, self.G, -1).topk(max(1, self.k // self.M), -1).values.sum(-1)
            keep = torch.zeros_like(g, dtype=torch.bool).scatter_(1, g.topk(self.M, -1).indices, True)
            sel = sel.masked_fill(~keep.repeat_interleave(self.E // self.G, 1), float('-inf'))
        idx = sel.topk(self.k, -1).indices
        w = scores.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)
        z = torch.logsumexp(logits, -1).square().mean()
        cnt = torch.zeros(n_seq, self.E, device=x.device).scatter_add_(
            1, idx.view(n_seq, -1), torch.ones_like(idx, dtype=torch.float).view(n_seq, -1))
        T = x.shape[0] // n_seq
        f = cnt * self.E / (self.k * T)
        p = (scores / scores.sum(-1, keepdim=True)).view(n_seq, T, -1).mean(1)
        aux = (f * p).sum(-1).mean()
        if self.training:
            self.load += cnt.sum(0)
        return idx, w, self.z_coef * z + self.aux_coef * aux

    @torch.no_grad()
    def update_bias(self, load):
        """Aux-loss-free balancing: b_i += u * sign(mean_load - load_i), on global-batch counts."""
        self.bias += self.rate * torch.sign(load.mean() - load)
        self.load.zero_()


def _experts(rx, r, gate, w13, w2, offs):
    h = grouped_mm(rx[r], w13.to(rx.dtype), offs)
    a, b = h.chunk(2, -1)
    return grouped_mm(F.silu(a) * b, w2.to(rx.dtype), offs) * gate.unsqueeze(1).to(rx.dtype)


def dispatch_combine(x, idx, w, w13, w2, group=None, dedup=True, fp8=False, recompute=False):
    """Dropless routed-expert computation. x (N,d), idx/w (N,k); w13 (El,d,2f), w2 (El,f,d) are this rank's
    experts; global expert e lives on rank e // El. Returns (out (N,d), rows sent, slots computed).
    recompute=True re-runs only the local expert GEMMs in backward (never the all-to-all)."""
    N, d = x.shape
    k = idx.shape[1]
    P = 1 if group is None else dist.get_world_size(group)
    El = w13.shape[0]
    dest = idx // El
    if dedup:
        sel = torch.zeros(N, P, dtype=torch.bool, device=x.device).scatter_(1, dest, True)
        p_idx, tok = sel.T.nonzero(as_tuple=True)                   # sorted by destination rank
        slot_ok = dest[tok] == p_idx[:, None]
    else:
        flat = dest.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok, p_idx = order // k, flat[order]
        slot_ok = torch.zeros(order.numel(), k, dtype=torch.bool, device=x.device)
        slot_ok[torch.arange(order.numel(), device=x.device), order % k] = True
    lidx = torch.where(slot_ok, idx[tok] - p_idx[:, None] * El, -1)
    meta = torch.cat([torch.where(slot_ok, w[tok], 0.0), lidx.float()], 1)
    send_counts = torch.bincount(p_idx, minlength=P)
    if group is None:
        in_splits = out_splits = [int(tok.numel())]
    else:
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        in_splits, out_splits = send_counts.tolist(), recv_counts.tolist()   # one host sync per layer
    rx = AllToAll.apply(x[tok], out_splits, in_splits, group, fp8)
    rmeta = AllToAll.apply(meta, out_splits, in_splits, group)
    rw, rl = rmeta[:, :k], rmeta[:, k:].detach().long()
    r, j = (rl >= 0).nonzero(as_tuple=True)
    le = rl[r, j]
    order = torch.argsort(le, stable=True)
    r, j, le = r[order], j[order], le[order]
    offs = torch.cumsum(torch.bincount(le, minlength=El), 0)
    args = (rx, r, rw[r, j], w13, w2, offs)
    y = checkpoint(_experts, *args, use_reentrant=False) if recompute else _experts(*args)
    back = torch.zeros(rx.shape[0], d, dtype=x.dtype, device=x.device).index_add(0, r, y)
    back = AllToAll.apply(back, in_splits, out_splits, group)
    out = torch.zeros(N, d, dtype=x.dtype, device=x.device).index_add(0, tok, back)
    return out, tok.numel(), slot_ok.sum()


@torch.compiler.disable
def _local_experts(x, gate, idx, w13, w2):
    """No expert parallelism: sync-free permutation + gathered grouped GEMMs (model/moe_local.py)."""
    from .moe_local import local_experts
    return local_experts(x, gate, idx, w13, w2)


class MoE(nn.Module):
    def __init__(self, c, ep_size=1):
        super().__init__()
        assert c.n_experts % ep_size == 0
        El = c.n_experts // ep_size
        self.router = Router(c)
        self.w13 = nn.Parameter(torch.empty(El, c.d_model, 2 * c.d_expert))
        self.w2 = nn.Parameter(torch.empty(El, c.d_expert, c.d_model))
        self.w13._muon_split = 2          # Muon orthogonalises w1 and w3 separately
        self.shared = SwiGLU(c.d_model, c.d_shared) if c.d_shared else None
        self.scale = c.routed_scale
        self.combine = getattr(c, 'shared_combine', 'sum')
        self.ep_group, self.dedup, self.fp8, self.recompute = None, True, FP8Policy(), False
        self.stats = {}

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        idx, w, aux = self.router(flat, B)
        if self.ep_group is None and flat.is_cuda:
            y = _local_experts(flat, w, idx, self.w13, self.w2)
        else:
            y, rows, slots = dispatch_combine(flat, idx, w, self.w13, self.w2, self.ep_group, self.dedup,
                                              self.fp8.dispatch, self.recompute)
            self.stats = dict(a2a_rows=rows, slots=slots, pairs=idx.numel())
        if self.scale != 1.0:
            y = y * self.scale
        if self.shared is not None:
            if self.combine == 'average':
                k = idx.shape[1]
                y = self.shared(flat) / (k + 1) + y * (k / (k + 1))
            else:
                y = y + self.shared(flat)
        return y.view(B, T, d), aux

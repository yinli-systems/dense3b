"""Decoder-only MoE transformer (pre-norm, RMSNorm, RoPE, GQA + QK-norm + output gate, shared+routed experts)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import Attention, RMSNorm, SwiGLU, doc_block_mask, rope_tables
from .moe import MoE


class Block(nn.Module):
    def __init__(self, c, layer_id, ep_size=1):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model, c.norm_eps)
        self.attn = Attention(c)
        self.ffn_norm = RMSNorm(c.d_model, c.norm_eps)
        self.is_moe = layer_id >= c.n_dense_layers
        self.ffn = MoE(c, ep_size) if self.is_moe else SwiGLU(c.d_model, c.d_ff_dense, fused=c.fuse_proj)

    def forward(self, x, cos, sin, block_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, block_mask)
        h = self.ffn_norm(x)
        if self.is_moe:
            y, aux = self.ffn(h)
        else:
            y, aux = self.ffn(h), h.new_zeros((), dtype=torch.float32)
        return x + y, aux


class LMHead(nn.Module):
    """Output projection + chunked fp32 cross-entropy; logits are recomputed in backward, never stored."""

    def __init__(self, c):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        self.weight = nn.Parameter(torch.empty(c.vocab_size, c.d_model))
        self.chunk = 2048
        self.fused_ce = False             # set by the trainer (--fused-ce); off = the checkpointed reference path

    def forward(self, h, targets=None):
        h = self.norm(h)
        if targets is None:
            return F.linear(h, self.weight.to(h.dtype)).float()
        h, t = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
        if h.is_cuda and self.fused_ce and torch.is_grad_enabled():
            return FusedLinearCE.apply(h, self.weight, t, self.chunk, (t != -100).sum().clamp(min=1).float())
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, h.shape[0], self.chunk):
            total = total + checkpoint(_ce_sum, h[i:i + self.chunk], self.weight, t[i:i + self.chunk],
                                       use_reentrant=False)
        return total / (t != -100).sum().clamp(min=1)


@torch.compile(dynamic=False)
def _ce_chunk_grad(logits, t):
    """-> (sum of CE over valid rows, d(sum CE)/d(logits) in logits' dtype) for one chunk; fp32 math."""
    lf = logits.float()
    valid = t != -100
    tt = t.clamp(min=0)
    lse = torch.logsumexp(lf, -1)
    tgt = lf.gather(1, tt[:, None]).squeeze(1)
    loss = ((lse - tgt) * valid).sum()
    g = torch.exp(lf - lse[:, None])
    g = g - torch.nn.functional.one_hot(tt, lf.shape[1]).to(g.dtype)
    g = g * valid[:, None]
    return loss, g.to(logits.dtype)


class FusedLinearCE(torch.autograd.Function):
    """Chunked LM head + cross-entropy that computes d/dh and d/dW during the forward pass (Liger-style): every
    chunk's logits are formed once (no recompute in backward) and never stored; 3 GEMMs per chunk instead of 4."""

    @staticmethod
    def forward(ctx, h, w, t, chunk, n_valid):
        wc = w.to(h.dtype)
        gh = torch.empty_like(h)
        gw = torch.zeros(w.shape, dtype=torch.float32, device=w.device)
        loss = torch.zeros((), dtype=torch.float32, device=h.device)
        for i in range(0, h.shape[0], chunk):
            hc = h[i:i + chunk]
            l, g = _ce_chunk_grad(hc @ wc.T, t[i:i + chunk])
            loss += l
            gh[i:i + chunk] = g @ wc
            gw += (g.T @ hc).float()
        ctx.save_for_backward(gh, gw, n_valid)
        ctx.w_dtype = w.dtype
        return loss / n_valid

    @staticmethod
    def backward(ctx, dloss):
        gh, gw, n_valid = ctx.saved_tensors
        s = dloss / n_valid
        return gh * s.to(gh.dtype), (gw * s).to(ctx.w_dtype), None, None, None


def _ce_sum(h, w, t):
    return F.cross_entropy(F.linear(h, w.to(h.dtype)).float(), t, ignore_index=-100, reduction='sum')


class Transformer(nn.Module):
    def __init__(self, c, ep_size=1):
        super().__init__()
        self.cfg = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Block(c, i, ep_size) for i in range(c.n_layers))
        self.head = LMHead(c)
        if c.tie_embeddings:
            self.head.weight = self.embed.weight
        cos, sin = rope_tables(c.head_dim, c.max_seq_len, c.rope_theta)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        self.ckpt_layers = set()          # indices of blocks recomputed in backward (activation checkpointing)
        self.pre_block_hook = None        # called (uncompiled) with 'embed' / block index / 'head' before each part runs

    def moe_layers(self):
        return [b.ffn for b in self.layers if b.is_moe]

    def forward(self, tokens, targets=None):
        bm = doc_block_mask(tokens, self.cfg.eos_id) if self.cfg.doc_mask else None
        hook = self.pre_block_hook
        if hook is not None:
            hook('embed')
        x = self.embed(tokens)
        aux = x.new_zeros((), dtype=torch.float32)
        for i, blk in enumerate(self.layers):
            if hook is not None:
                hook(i)
            if i in self.ckpt_layers and torch.is_grad_enabled():
                x, a = checkpoint(blk, x, self.cos, self.sin, bm, use_reentrant=False)
            else:
                x, a = blk(x, self.cos, self.sin, bm)
            aux = aux + a
        if hook is not None:
            hook('head')
        if targets is None:
            return self.head(x)
        ce = self.head(x, targets)
        return dict(loss=ce + aux, ce=ce.detach(), aux=aux.detach())


@torch.no_grad()
def init_weights(model, seed=0, ep_rank=0):
    """Truncated-normal init (std 0.02, output projections / sqrt(2L)). Every expert is drawn from its own
    generator keyed by (seed, layer, global expert id), so weights do not depend on the EP layout."""
    c = model.cfg
    g = torch.Generator().manual_seed(seed)
    std, out_std = c.init_std, c.init_std / math.sqrt(2 * c.n_layers)

    def tn(t, s, gen=g):
        t.copy_(torch.nn.init.trunc_normal_(torch.empty(t.shape), 0.0, s, -3 * s, 3 * s, generator=gen))

    tn(model.embed.weight, std)
    if not c.tie_embeddings:
        tn(model.head.weight, std)
    for li, blk in enumerate(model.layers):
        for name, m in blk.named_modules():
            if isinstance(m, nn.Linear):
                tn(m.weight, out_std if name.endswith(('wo', 'w2')) else std)
            elif isinstance(m, RMSNorm):
                m.weight.fill_(1.0)
        if blk.is_moe:
            moe = blk.ffn
            tn(moe.router.weight, std)
            moe.router.bias.zero_()
            El = moe.w13.shape[0]
            for e in range(El):
                ge = ep_rank * El + e
                gen = torch.Generator().manual_seed(seed * 1_000_003 + li * 10_007 + ge + 1)
                tn(moe.w13[e], std, gen)
                tn(moe.w2[e], out_std, gen)
    model.head.norm.weight.fill_(1.0)

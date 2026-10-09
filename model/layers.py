"""Dense building blocks: RMSNorm, RoPE, GQA attention with QK-norm and output gate, SwiGLU."""
import torch
import torch.nn as nn
import torch.nn.functional as F

EOS_ID = 50279
_flex = None


def flex_attention(q, k, v, block_mask, enable_gqa):
    """flex_attention must run compiled to be fast (and to use the block-sparse mask); compile once, lazily."""
    global _flex
    if _flex is None:
        from torch.nn.attention.flex_attention import flex_attention as fa
        _flex = torch.compile(fa, dynamic=False) if q.is_cuda else fa
    return _flex(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)


def doc_block_mask(tokens, eos=EOS_ID):
    """Causal mask restricted to the same document. A document's EOS belongs to that document, so the token
    after an EOS starts a new one and sees nothing before it."""
    B, T = tokens.shape
    is_eos = (tokens == eos).int()
    doc = torch.cumsum(is_eos, dim=1) - is_eos
    if not tokens.is_cuda:            # CPU (tests): dense boolean mask for SDPA; flex has no CPU backward
        causal = torch.ones(T, T, dtype=torch.bool, device=tokens.device).tril()
        return (causal & (doc[:, :, None] == doc[:, None, :]))[:, None]
    from torch.nn.attention.flex_attention import create_block_mask

    def same_doc_causal(b, h, q, kv):
        return (q >= kv) & (doc[b, q] == doc[b, kv])

    return create_block_mask(same_doc_causal, B, None, T, T, device=tokens.device,
                             BLOCK_SIZE=128 if T % 128 == 0 else T)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


def rope_tables(head_dim, max_len, theta, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim))
    ang = torch.outer(torch.arange(max_len, device=device, dtype=torch.float64), inv)
    return ang.cos().float(), ang.sin().float()


def apply_rope(x, cos, sin):
    # x: (B, T, H, D); rotate-half convention, computed in fp32 (bf16 angles lose precision at long positions)
    T = x.shape[1]
    c, s = cos[:T, None, :].float(), sin[:T, None, :].float()
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        d = c.d_model
        self.fused = c.fuse_proj
        if self.fused:
            self.sizes = [self.h * self.hd, self.kv * self.hd, self.kv * self.hd] + \
                         ([self.h * self.hd] if c.attn_gate else [])
            self.wqkvg = nn.Linear(d, sum(self.sizes), bias=False)
            self.wqkvg.weight._muon_rows = list(self.sizes)       # orthogonalise q, k, v, g separately
            self.wg = None
        else:
            self.wq = nn.Linear(d, self.h * self.hd, bias=False)
            self.wk = nn.Linear(d, self.kv * self.hd, bias=False)
            self.wv = nn.Linear(d, self.kv * self.hd, bias=False)
            self.wg = nn.Linear(d, self.h * self.hd, bias=False) if c.attn_gate else None
        self.wo = nn.Linear(self.h * self.hd, d, bias=False)
        self.q_norm = RMSNorm(self.hd, c.norm_eps) if c.qk_norm else None
        self.k_norm = RMSNorm(self.hd, c.norm_eps) if c.qk_norm else None

    def forward(self, x, cos, sin, block_mask=None):
        B, T, _ = x.shape
        if self.fused:
            q, k, v, *g = self.wqkvg(x).split(self.sizes, dim=-1)
            g = g[0] if g else None
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            g = self.wg(x) if self.wg is not None else None
        q, k, v = q.view(B, T, self.h, self.hd), k.view(B, T, self.kv, self.hd), v.view(B, T, self.kv, self.hd)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if isinstance(block_mask, torch.Tensor):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask, enable_gqa=self.kv != self.h)
        elif block_mask is not None:
            o = flex_attention(q, k, v, block_mask, enable_gqa=self.kv != self.h)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.kv != self.h)
        o = o.transpose(1, 2).reshape(B, T, self.h * self.hd)
        if g is not None:
            o = o * torch.sigmoid(g)
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, d, ff, fused=False):
        super().__init__()
        self.fused = fused
        if fused:
            self.w13 = nn.Linear(d, 2 * ff, bias=False)
            self.w13.weight._muon_rows = [ff, ff]
        else:
            self.w1 = nn.Linear(d, ff, bias=False)
            self.w3 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        if self.fused:
            a, b = self.w13(x).chunk(2, dim=-1)
            return self.w2(F.silu(a) * b)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

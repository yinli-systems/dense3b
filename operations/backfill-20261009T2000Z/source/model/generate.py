"""Batched greedy decoding with a KV cache for evaluation (no training code path changes).

Prompts of different lengths are left-padded; every row gets its own RoPE positions (0 at its first real token)
and an attention mask that hides the padding, so each row decodes exactly as it would alone.
"""
import torch
import torch.nn.functional as F



def _rope(x, cos, sin, pos):
    # x (B, T, H, D); pos (B, T) absolute positions per row
    c, s = cos[pos][:, :, None, :].float(), sin[pos][:, :, None, :].float()
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def _attn(blk_attn, x, cos, sin, pos, cache, mask):
    a = blk_attn
    B, T, _ = x.shape
    if a.fused:
        q, k, v, *g = a.wqkvg(x).split(a.sizes, dim=-1)
        g = g[0] if g else None
    else:
        q, k, v = a.wq(x), a.wk(x), a.wv(x)
        g = a.wg(x) if a.wg is not None else None
    qk_full = getattr(a, 'qk_full', False)
    if a.q_norm is not None and qk_full:                 # OLMo-core: RMSNorm over all heads' dims at once
        q, k = a.q_norm(q), a.k_norm(k)
    q, k, v = q.view(B, T, a.h, a.hd), k.view(B, T, a.kv, a.hd), v.view(B, T, a.kv, a.hd)
    if a.q_norm is not None and not qk_full:
        q, k = a.q_norm(q), a.k_norm(k)
    q, k = _rope(q, cos, sin, pos), _rope(k, cos, sin, pos)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=a.kv != a.h)
    o = o.transpose(1, 2).reshape(B, T, a.h * a.hd)
    if g is not None:
        o = o * torch.sigmoid(g)
    return a.wo(o), (k, v)


@torch.no_grad()
def forward_cached(model, tokens, pos, valid, caches):
    """tokens/pos (B, T); valid (B, Tk) bool over all keys so far (incl. these); caches: list per layer or None."""
    B, T = tokens.shape
    Tk = valid.shape[1]
    q_abs = torch.arange(Tk - T, Tk, device=tokens.device)
    k_abs = torch.arange(Tk, device=tokens.device)
    mask = (k_abs[None, :] <= q_abs[:, None])[None, None] & valid[:, None, None, :]      # (B,1,T,Tk)
    x = model.embed(tokens)
    new = []
    for i, blk in enumerate(model.layers):
        h, kv = _attn(blk.attn, blk.attn_norm(x), model.cos, model.sin, pos, caches[i] if caches else None, mask)
        x = x + h
        y = blk.ffn(blk.ffn_norm(x))
        x = x + (y[0] if isinstance(y, tuple) else y)        # MoE layers return (output, aux loss)
        new.append(kv)
    logits = F.linear(model.head.norm(x[:, -1]), model.head.weight.to(x.dtype)).float()
    return logits, new


@torch.no_grad()
def greedy(model, prompts, max_new, eos, pad=1, stop_fn=None, ban_from=None, check_every=8):
    """prompts: list of token-id lists. Returns list of generated id lists (EOS excluded, cut where stop_fn says).
    stop_fn(row_index, generated_ids) -> bool, checked every `check_every` tokens."""
    dev = model.embed.weight.device
    B = len(prompts)
    L = max(len(p) for p in prompts)
    toks = torch.full((B, L), pad, dtype=torch.long, device=dev)
    valid = torch.zeros(B, L, dtype=torch.bool, device=dev)
    for b, p in enumerate(prompts):
        toks[b, L - len(p):] = torch.tensor(p, device=dev)
        valid[b, L - len(p):] = True
    pos = (valid.long().cumsum(1) - 1).clamp(min=0)
    logits, caches = forward_cached(model, toks, pos, valid, None)
    out = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    next_pos = pos[:, -1] + 1
    for step in range(max_new):
        if ban_from is not None:
            logits[:, ban_from:] = -float('inf')
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        done |= nxt == eos
        for b in range(B):
            if not done[b]:
                out[b].append(int(nxt[b]))
        if stop_fn is not None and (step + 1) % check_every == 0:
            for b in range(B):
                if not done[b] and stop_fn(b, out[b]):
                    done[b] = True
        if bool(done.all()):
            break
        valid = torch.cat([valid, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=1)
        logits, caches = forward_cached(model, nxt[:, None], next_pos[:, None], valid, caches)
        next_pos = next_pos + 1
    return out

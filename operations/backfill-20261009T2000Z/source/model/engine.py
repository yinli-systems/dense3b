"""Sampling engine for long reasoning rollouts (GRPO) and pass@k evals: static KV cache + continuous batching.

* KV cache preallocated once: per layer (rows, kv_heads, max_len, head_dim) BF16. No torch.cat per token.
* Continuous batching: `max_rows` slots; when a row finishes (stop token or length) the next request is prefilled into
  that slot, so the GPU always decodes a full batch even though reasoning lengths vary 10x.
* Decode attends over keys [0, Tk) with Tk = max active length rounded up to 256 (fewer distinct SDPA shapes); the
  per-row mask hides keys >= that row's length.
* Sampling: temperature, top-p, per-request seed (torch.Generator per step from (seed, request, position)) so a rollout
  is reproducible given its request id.

  eng = Engine(model, max_rows=32, max_len=16384)
  outs = eng.generate([ids, ...], max_new=8192, temperature=0.6, top_p=0.95, stop_ids={im_end, eos}, seed=0)
  -> list of dict(ids=[...], finish='stop'|'length', logprobs=[...]) in request order
"""
import math
import torch
import torch.nn.functional as F

from model.generate import _rope


class Engine:
    def __init__(self, model, max_rows=32, max_len=16384, block=256):
        self.m, self.R, self.T, self.block = model, max_rows, max_len, block
        p = next(model.parameters())
        self.dev, self.dt = p.device, p.dtype
        a = model.layers[0].attn
        self.kv, self.hd = a.kv, a.hd
        self.k = [torch.zeros(max_rows, a.kv, max_len, a.hd, device=self.dev, dtype=self.dt) for _ in model.layers]
        self.v = [torch.zeros_like(x) for x in self.k]
        self.len = torch.zeros(max_rows, dtype=torch.long, device=self.dev)
        assert max_len <= model.cos.shape[0], f'max_len {max_len} > RoPE table {model.cos.shape[0]}'

    # ------------------------------------------------------------------ attention with the static cache
    def _qkv(self, a, x, pos):
        B, T, _ = x.shape
        if a.fused:
            q, k, v, *g = a.wqkvg(x).split(a.sizes, dim=-1)
            g = g[0] if g else None
        else:
            q, k, v = a.wq(x), a.wk(x), a.wv(x)
            g = a.wg(x) if a.wg is not None else None
        if a.q_norm is not None and getattr(a, 'qk_full', False):
            q, k = a.q_norm(q), a.k_norm(k)
        q, k, v = q.view(B, T, a.h, a.hd), k.view(B, T, a.kv, a.hd), v.view(B, T, a.kv, a.hd)
        if a.q_norm is not None and not getattr(a, 'qk_full', False):
            q, k = a.q_norm(q), a.k_norm(k)
        q, k = _rope(q, self.m.cos, self.m.sin, pos), _rope(k, self.m.cos, self.m.sin, pos)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), g

    def _out(self, a, o, g):
        B, H, T, D = o.shape
        o = o.transpose(1, 2).reshape(B, T, H * D)
        if g is not None:
            o = o * torch.sigmoid(g)
        return a.wo(o)

    @torch.no_grad()
    def prefill(self, row, ids):
        """Write the KV of prompt `ids` into slot `row`; returns logits of the last position (V,) float32."""
        m = self.m
        n = len(ids)
        assert n < self.T, 'prompt longer than the cache'
        tok = torch.tensor(ids, device=self.dev)[None]
        pos = torch.arange(n, device=self.dev)[None]
        x = m.embed(tok)
        for i, blk in enumerate(m.layers):
            a = blk.attn
            q, k, v, g = self._qkv(a, blk.attn_norm(x), pos)
            self.k[i][row, :, :n] = k[0]
            self.v[i][row, :, :n] = v[0]
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=a.kv != a.h)
            x = x + self._out(a, o, g)
            y = blk.ffn(blk.ffn_norm(x))
            x = x + (y[0] if isinstance(y, tuple) else y)
        self.len[row] = n
        return F.linear(m.head.norm(x[0, -1]), m.head.weight.to(x.dtype)).float()

    @torch.no_grad()
    def decode(self, rows, tok):
        """One token for each slot in `rows` (LongTensor), inputs `tok` (B,). Returns logits (B, V) float32."""
        m = self.m
        B = rows.numel()
        pos = self.len[rows]                                            # position of the new token = current length
        Tk = int(pos.max().item()) + 1
        Tk = min(self.T, ((Tk + self.block - 1) // self.block) * self.block)
        mask = (torch.arange(Tk, device=self.dev)[None, :] <= pos[:, None])[:, None, None, :]   # (B,1,1,Tk)
        x = m.embed(tok[:, None])
        for i, blk in enumerate(m.layers):
            a = blk.attn
            q, k, v, g = self._qkv(a, blk.attn_norm(x), pos[:, None])
            self.k[i][rows, :, pos] = k[:, :, 0]
            self.v[i][rows, :, pos] = v[:, :, 0]
            kk, vv = self.k[i][rows, :, :Tk], self.v[i][rows, :, :Tk]
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, enable_gqa=a.kv != a.h)
            x = x + self._out(a, o, g)
            y = blk.ffn(blk.ffn_norm(x))
            x = x + (y[0] if isinstance(y, tuple) else y)
        self.len[rows] += 1
        return F.linear(m.head.norm(x[:, 0]), m.head.weight.to(x.dtype)).float()

    # ------------------------------------------------------------------ sampling loop
    @staticmethod
    def _sample(logits, temperature, top_p, gen):
        if temperature <= 0:
            return logits.argmax(-1)
        p = torch.softmax(logits / temperature, -1)
        if top_p < 1.0:
            sp, si = p.sort(-1, descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = sp * keep
            idx = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)[:, 0]
            return si.gather(-1, idx[:, None])[:, 0]
        return torch.multinomial(p, 1, generator=gen)[:, 0]

    @torch.no_grad()
    def generate(self, prompts, max_new, temperature=0.6, top_p=0.95, stop_ids=(), seed=0, ban_from=None,
                 return_logprobs=False):
        stop = torch.tensor(sorted(stop_ids), device=self.dev) if stop_ids else None
        queue = list(range(len(prompts)))[::-1]
        res = [None] * len(prompts)
        slot_req = [-1] * self.R
        out = [[] for _ in range(self.R)]
        lps = [[] for _ in range(self.R)]
        last = torch.zeros(self.R, dtype=torch.long, device=self.dev)
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(seed)
        pending_logits = {}

        def admit(s):
            r = queue.pop()
            limit = min(max_new, self.T - len(prompts[r]) - 1)
            slot_req[s] = r; out[s] = []; lps[s] = []
            pending_logits[s] = (self.prefill(s, prompts[r]), limit)

        for s in range(self.R):
            if queue:
                admit(s)
        limits = {}
        while any(r >= 0 for r in slot_req):
            act = [s for s in range(self.R) if slot_req[s] >= 0]
            # slots that were just admitted sample their first token from the prefill logits
            fresh = [s for s in act if s in pending_logits]
            if fresh:
                lg = torch.stack([pending_logits[s][0] for s in fresh])
                for s in fresh:
                    limits[s] = pending_logits.pop(s)[1]
                self._emit(fresh, lg, temperature, top_p, gen, ban_from, out, lps, last, return_logprobs)
            old = [s for s in act if s not in fresh]
            if old:
                rows = torch.tensor(old, device=self.dev)
                lg = self.decode(rows, last[rows])
                self._emit(old, lg, temperature, top_p, gen, ban_from, out, lps, last, return_logprobs)
            # finished rows -> results; refill
            for s in act:
                ids = out[s]
                done_stop = stop is not None and ids and ids[-1] in stop_ids
                if done_stop or len(ids) >= limits[s]:
                    r = slot_req[s]
                    res[r] = dict(ids=ids[:-1] if done_stop else ids, stop=ids[-1] if done_stop else None,
                                  finish='stop' if done_stop else 'length', logprobs=lps[s] if return_logprobs else None)
                    slot_req[s] = -1
                    if queue:
                        admit(s)
        return res

    def _emit(self, slots, logits, temperature, top_p, gen, ban_from, out, lps, last, return_logprobs):
        if ban_from is not None:
            logits[:, ban_from:] = -float('inf')
        nxt = self._sample(logits, temperature, top_p, gen)
        if return_logprobs:
            lp = torch.log_softmax(logits / max(temperature, 1e-6), -1).gather(-1, nxt[:, None])[:, 0].tolist()
        nl = nxt.tolist()
        for j, s in enumerate(slots):
            out[s].append(nl[j])
            if return_logprobs:
                lps[s].append(lp[j])
        last[torch.tensor(slots, device=self.dev)] = nxt

"""Routing-causality study probe (docs/ROUTING_STUDY.md): one checkpoint in, one JSON (+ npz of token-level labels) out.

The probe set is fixed for the whole study: PROBE_WINDOWS windows of PROBE_LEN tokens per domain, read from the mix-m1
shards at seeded offsets, so every snapshot of every twin is measured on exactly the same tokens.

MoE checkpoints (per MoE layer):
  usage_entropy      H(E)/log E of the expert-selection distribution (1 = perfectly balanced)
  load_max_mean      max/mean expert load
  mi_domain          I(E;D)/H(D): how much the selected experts tell about the domain (0 = none, 1 = all)
  margin             mean selection-score gap between the k-th and (k+1)-th expert (router decisiveness)
  top1_weight        mean normalised gate weight of the top expert (router confidence)
  domain_overlap     mean pairwise Jaccard of each domain's top-25% experts (low = domains use different experts)
  expert_cos         mean pairwise cosine between expert weights (w13): low = experts diverged
  geometry_nmi       NMI(k-means of this layer's FFN input, top-1 expert) -- does hidden-state geometry already
                     predict the routing (the "routing just reads geometry" hypothesis)?
  permute_dce        CE increase when expert weights are permuted while the router is kept (router-expert coupling)
Dense checkpoints: k-means (k = number of experts of the MoE twin) of every layer's FFN input; the cluster ids are
saved so bench/routing/compare.py can compute NMI/ARI(dense clusters, MoE routes) on the same tokens.
All: per-domain CE on the probe set.

  python -m bench.routing.probe CKPT --out OUT.json [--permute] [--k 32]
"""
import argparse, glob, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from bench.eval_suite import load

MIX = '/data/run01/scxi253/moe1b/mix-m1/data.txt'
DOMAINS = {'web': ['web-v2'], 'wiki': ['wikipedia'], 'arxiv': ['arxiv'], 'stackexchange': ['stackexchange'],
           'code': ['stackv2-edu', 'swallow-code', 'opc-annealing'],
           'math': ['finemath-4plus', 'nemotron-cc-math-4plus', 'swallow-math']}
PROBE_WINDOWS, PROBE_LEN, SEED = 48, 1024, 20261008


def probe_set():
    paths = [p for p in open(MIX).read().replace('\n', ',').split(',') if p.strip()]
    rng = np.random.default_rng(SEED)
    toks, labels = [], []
    for di, (dom, srcs) in enumerate(DOMAINS.items()):
        ps = sorted(p for p in paths if any(f'/{s}/' in p for s in srcs))
        for w in range(PROBE_WINDOWS):
            mm = np.memmap(ps[rng.integers(len(ps))], dtype='<u2', mode='r')
            off = int(rng.integers(0, len(mm) - PROBE_LEN - 1))
            toks.append(np.asarray(mm[off:off + PROBE_LEN + 1], dtype=np.int64)); labels.append(di)
    return np.stack(toks), np.array(labels)


def nmi(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ca, cb = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    n = len(a); joint = np.zeros((ca.max() + 1, cb.max() + 1)); np.add.at(joint, (ca, cb), 1); joint /= n
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    ha, hb = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(), -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(2 * mi / (ha + hb)) if ha + hb > 0 else 0.0


def kmeans(x, k, iters=25, seed=0):
    """x (N, d) float32 torch on GPU -> labels (N,). Cosine k-means (router logits are dot products)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    x = F.normalize(x.float(), dim=-1)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        lab = (x @ c.T).argmax(1)
        for j in range(k):
            m = lab == j
            c[j] = F.normalize(x[m].mean(0), dim=0) if m.any() else x[torch.randint(len(x), (1,), generator=g, device=x.device)][0]
    return (x @ c.T).argmax(1)


@torch.no_grad()
def run(model, toks, capture=True, batch=2):
    """-> per-window CE (float), per-layer dict: ffn input sample (bf16) and router outputs for MoE layers."""
    cap = {}
    hooks = []
    if capture:
        for li, blk in enumerate(model.layers):
            def ffn_hook(mod, inp, li=li):
                cap.setdefault(('h', li), []).append(inp[0].reshape(-1, inp[0].shape[-1])[::4].to(torch.bfloat16).cpu())
            hooks.append(blk.ffn.register_forward_pre_hook(ffn_hook))
            r = getattr(blk.ffn, 'router', None)
            if r is not None:
                def router_hook(mod, inp, out, li=li):
                    x = inp[0]
                    logits = F.linear(x.float(), mod.weight.float())
                    sel = logits.sigmoid() + mod.bias if mod.kind != 'softmax' else logits.softmax(-1)
                    top = sel.topk(mod.k + 1, -1)
                    idx, w, _ = out
                    cap.setdefault(('idx', li), []).append(idx.to(torch.int16).cpu())
                    cap.setdefault(('margin', li), []).append((top.values[:, -2] - top.values[:, -1]).float().cpu())
                    cap.setdefault(('w1', li), []).append(w.max(-1).values.float().cpu())
                hooks.append(r.register_forward_hook(router_hook))
    ces = []
    for i in range(0, len(toks), batch):
        t = torch.from_numpy(toks[i:i + batch]).cuda()
        x = model.embed(t[:, :-1])
        for blk in model.layers:
            x, _ = blk(x, model.cos, model.sin, None)
        h = model.head.norm(x)
        logits = F.linear(h, model.head.weight.to(h.dtype)).float()
        ce = F.cross_entropy(logits.transpose(1, 2), t[:, 1:], reduction='none').mean(1)
        ces.append(ce.cpu()); del logits, h, x
    for h in hooks:
        h.remove()
    return torch.cat(ces).numpy(), {k: torch.cat(v) for k, v in cap.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=32, help='clusters for Dense FFN inputs (= experts of the MoE twin)')
    ap.add_argument('--permute', action='store_true', help='also measure CE after a random expert permutation')
    a = ap.parse_args()
    t0 = time.time()
    model, meta = load(a.ckpt)
    toks, lab = probe_set()
    dom_names = list(DOMAINS)
    ce, cap = run(model, toks)
    tok_lab = np.repeat(lab, PROBE_LEN)                               # per-token domain (router sees every token)
    res = dict(ckpt=a.ckpt, step=meta['step'], preset=meta['args']['preset'],
               tokens=int((meta['step'] + 1) * meta['args']['global_batch'] * 4096),
               ce={d: float(ce[lab == i].mean()) for i, d in enumerate(dom_names)}, layers={})
    save = {'domain_per_window': lab}
    n_layers = len(model.layers)
    for li in range(n_layers):
        L = {}
        h = cap[('h', li)].cuda()
        sub_lab = np.repeat(lab, PROBE_LEN // 4)
        if ('idx', li) in cap:
            ffn = model.layers[li].ffn
            E, k = ffn.router.E, ffn.router.k
            idx = cap[('idx', li)].long()                              # (N, k)
            cnt = torch.bincount(idx.reshape(-1).cuda(), minlength=E).float()
            p = cnt / cnt.sum()
            L['usage_entropy'] = float(-(p[p > 0] * p[p > 0].log()).sum() / math.log(E))
            L['load_max_mean'] = float(cnt.max() / cnt.mean())
            # I(E;D)/H(D), counting each of the k selections
            joint = torch.zeros(len(dom_names), E, device='cuda')
            dl = torch.from_numpy(np.repeat(tok_lab, k)).cuda()
            joint.index_put_((dl, idx.reshape(-1).cuda()), torch.ones_like(dl, dtype=torch.float), accumulate=True)
            joint /= joint.sum(); pd, pe = joint.sum(1), joint.sum(0); nz = joint > 0
            mi = (joint[nz] * (joint[nz] / (pd[:, None] * pe[None, :])[nz]).log()).sum()
            L['mi_domain'] = float(mi / -(pd[pd > 0] * pd[pd > 0].log()).sum())
            L['margin'] = float(cap[('margin', li)].mean())
            L['top1_weight'] = float(cap[('w1', li)].mean())
            tops = [set(joint[d].topk(max(1, E // 4)).indices.tolist()) for d in range(len(dom_names))]
            jac = [len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i in range(len(tops)) for j in range(i + 1, len(tops))]
            L['domain_overlap'] = float(np.mean(jac))
            w = F.normalize(ffn.w13.float().reshape(E, -1), dim=-1)
            cs = w @ w.T
            L['expert_cos'] = float((cs.sum() - E) / (E * (E - 1)))
            top1 = idx[:, 0].cpu().numpy()[::4]
            km = kmeans(h, E).cpu().numpy()
            L['geometry_nmi'] = nmi(km, top1)
            save[f'top1_l{li}'] = top1.astype(np.int16)
            save[f'km_l{li}'] = km.astype(np.int16)
        else:
            km = kmeans(h, a.k).cpu().numpy()
            L['cluster_domain_nmi'] = nmi(km, sub_lab)
            save[f'km_l{li}'] = km.astype(np.int16)
        res['layers'][li] = L
        del h; torch.cuda.empty_cache()
    if a.permute and any(('idx', li) in cap for li in range(n_layers)):
        g = torch.Generator().manual_seed(SEED)
        saved = {}
        for li, blk in enumerate(model.layers):
            f = blk.ffn
            if hasattr(f, 'router'):
                perm = torch.randperm(f.router.E, generator=g).cuda()
                saved[li] = (f.w13.data.clone(), f.w2.data.clone())
                f.w13.data.copy_(f.w13.data[perm]); f.w2.data.copy_(f.w2.data[perm])
        ce_p, _ = run(model, toks, capture=False)
        res['permute_dce'] = {d: float(ce_p[lab == i].mean() - ce[lab == i].mean()) for i, d in enumerate(dom_names)}
        for li, (w13, w2) in saved.items():
            model.layers[li].ffn.w13.data.copy_(w13); model.layers[li].ffn.w2.data.copy_(w2)
    res['seconds'] = round(time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    np.savez_compressed(a.out.replace('.json', '.npz'), **save)
    print(json.dumps({k: v for k, v in res.items() if k != 'layers'}), flush=True)


if __name__ == '__main__':
    main()

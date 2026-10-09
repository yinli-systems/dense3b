"""Muon (matrix params, incl. per-expert matrices) + AdamW (embeddings, head, router, norms), and the WSD schedule.

Muon follows Moonlight (arXiv 2502.16982): Nesterov momentum, 5-step Newton-Schulz orthogonalisation,
update scaled by 0.2*sqrt(max(A, B)) so its RMS matches AdamW and both share one LR, decoupled weight decay.
Expert matrices are local to their EP rank, so their Newton-Schulz needs no communication; FSDP-sharded
dense matrices are gathered within the node (shard group) once per optimizer step.
"""
import math
import torch
from torch.distributed.tensor import DTensor, distribute_tensor

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def newton_schulz(G, steps=5):
    """Approximate the orthogonal polar factor of every matrix in G (..., A, B)."""
    a, b, c = NS_COEFFS
    X = G.bfloat16() if G.is_cuda else G.float()
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return (X.mT if tr else X).to(G.dtype)


def muon_update(m, split=1):
    """m (..., A, B) -> orthogonalised update with Moonlight RMS scaling; `split` blocks along B handled apart."""
    if split > 1:
        *lead, A, B = m.shape
        u = muon_update(m.reshape(*lead, A, split, B // split).movedim(-2, -3))
        return u.movedim(-3, -2).reshape(m.shape)
    return newton_schulz(m) * (0.2 * math.sqrt(max(m.shape[-2:])))


class MuonAdamW(torch.optim.Optimizer):
    """Each param group sets kind='muon' or 'adamw'. State is stored as local shards (checkpoint-friendly)."""

    def __init__(self, groups, lr, betas=(0.9, 0.95), eps=1e-8, momentum=0.95, weight_decay=0.1):
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, wd = grp['lr'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g, w, st = _local(p.grad), _local(p), self.state[p]
                if grp['kind'] == 'muon':
                    buf = st.setdefault('momentum', torch.zeros_like(g, dtype=torch.float32))
                    buf.mul_(grp['momentum']).add_(g)
                    m = g.float() + grp['momentum'] * buf
                    if isinstance(p, DTensor):
                        full = DTensor.from_local(m, p.device_mesh, p.placements, shape=p.shape,
                                                  stride=p.stride()).full_tensor()
                        u = distribute_tensor(muon_update(full, getattr(p, '_muon_split', 1)), p.device_mesh,
                                              p.placements, src_data_rank=None).to_local()
                    elif getattr(p, '_muon_rows', None):
                        u = torch.cat([muon_update(b) for b in m.split(p._muon_rows, dim=-2)], dim=-2)
                    else:
                        u = muon_update(m, getattr(p, '_muon_split', 1))
                    w.mul_(1 - lr * wd).add_(u.to(w.dtype), alpha=-lr)
                else:
                    b1, b2 = grp['betas']
                    t = st['step'] = st.get('step', 0) + 1
                    m1 = st.setdefault('exp_avg', torch.zeros_like(g, dtype=torch.float32))
                    m2 = st.setdefault('exp_avg_sq', torch.zeros_like(g, dtype=torch.float32))
                    m1.lerp_(g.float(), 1 - b1)
                    m2.mul_(b2).addcmul_(g.float(), g.float(), value=1 - b2)
                    denom = (m2 / (1 - b2 ** t)).sqrt_().add_(grp['eps'])
                    w.mul_(1 - lr * wd).addcdiv_((m1 / (1 - b1 ** t)).to(w.dtype), denom.to(w.dtype), value=-lr)


def param_groups(model, use_muon=True, weight_decay=0.1):
    """Muon: 2D/3D weights inside blocks except the router. AdamW: embeddings, head, router; norms and
    embeddings without WD."""
    muon, adam, no_wd = [], [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or name.startswith('embed'):      # OLMo 2: no weight decay on embeddings
            no_wd.append(p)
        elif use_muon and name.startswith('layers.') and 'router' not in name:
            muon.append(p)
        else:
            adam.append(p)
    groups = [dict(params=muon, kind='muon', weight_decay=weight_decay),
              dict(params=adam, kind='adamw', weight_decay=weight_decay),
              dict(params=no_wd, kind='adamw', weight_decay=0.0)]
    return [g for g in groups if g['params']]


def wsd_lr(step, total, peak, warmup, cooldown_frac=0.2, final_frac=0.0):
    """Warmup-stable-decay with a (1 - sqrt) cooldown over the last `cooldown_frac` of training."""
    if step < warmup:
        return peak * (step + 1) / warmup
    cd = int(total * cooldown_frac)
    start = total - cd
    if step < start or cd == 0:
        return peak
    frac = min(1.0, (step - start) / cd)
    return peak * (final_frac + (1 - final_frac) * (1 - math.sqrt(frac)))

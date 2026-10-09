import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from model.moe import dispatch_combine


def naive(x, idx, w, w13, w2):
    out = torch.zeros_like(x)
    for n in range(x.shape[0]):
        for j in range(idx.shape[1]):
            e = idx[n, j]
            a, b = (x[n] @ w13[e]).chunk(2)
            out[n] += w[n, j] * ((F.silu(a) * b) @ w2[e])
    return out


def routing(N, E, k, gen):
    idx = torch.stack([torch.randperm(E, generator=gen)[:k] for _ in range(N)])
    w = torch.rand(N, k, generator=gen, dtype=torch.float64)
    return idx, w / w.sum(-1, keepdim=True)


def problem(N=40, d=16, f=8, E=8, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, d, generator=g, dtype=torch.float64)
    w13 = torch.randn(E, d, 2 * f, generator=g, dtype=torch.float64) / 4
    w2 = torch.randn(E, f, d, generator=g, dtype=torch.float64) / 4
    idx, w = routing(N, E, k, g)
    return x, idx, w, w13, w2


@pytest.mark.parametrize('dedup', [True, False])
def test_dropless_matches_naive(dedup):
    x, idx, w, w13, w2 = problem()
    out, rows, slots = dispatch_combine(x, idx, w, w13, w2, dedup=dedup)
    assert torch.allclose(out, naive(x, idx, w, w13, w2), atol=1e-12)
    assert int(slots) == idx.numel()                       # dropless: every (token, expert) slot computed
    assert rows == (x.shape[0] if dedup else idx.numel())  # one rank: dedup sends each token once


def test_dispatch_combine_is_exact_permutation_inverse():
    """combine(dispatch(x)) returns every expert output to exactly its source token: perturbing token n
    changes output row n and nothing else."""
    x, idx, w, w13, w2 = problem()
    out = dispatch_combine(x, idx, w, w13, w2)[0]
    for n in [0, 7, 39]:
        x2 = x.clone()
        x2[n] += 1.0
        diff = (dispatch_combine(x2, idx, w, w13, w2)[0] - out).abs().sum(-1)
        assert diff[n] > 0 and torch.count_nonzero(diff) == 1


def test_gradients_match_naive():
    x, idx, w, w13, w2 = problem(N=12)
    ts = [t.clone().requires_grad_() for t in (x, w, w13, w2)]
    dispatch_combine(ts[0], idx, ts[1], ts[2], ts[3])[0].square().sum().backward()
    rs = [t.clone().requires_grad_() for t in (x, w, w13, w2)]
    naive(rs[0], idx, rs[1], rs[2], rs[3]).square().sum().backward()
    for a, b in zip(ts, rs):
        assert torch.allclose(a.grad, b.grad, atol=1e-10)


def test_expert_recompute_same_values_and_grads():
    x, idx, w, w13, w2 = problem(N=12)
    res = []
    for rc in (False, True):
        ts = [t.clone().requires_grad_() for t in (x, w, w13, w2)]
        out = dispatch_combine(ts[0], idx, ts[1], ts[2], ts[3], recompute=rc)[0]
        out.square().sum().backward()
        res.append([out.detach()] + [t.grad for t in ts])
    assert all(torch.equal(a, b) for a, b in zip(*res))


def test_fp8_dispatch_close():
    x, idx, w, w13, w2 = problem()
    x, w, w13, w2 = x.float(), w.float(), w13.float(), w2.float()
    ref = dispatch_combine(x, idx, w, w13, w2)[0]
    q = dispatch_combine(x, idx, w, w13, w2, fp8=True)[0]
    rel = (q - ref).norm() / ref.norm()
    assert 0 < rel < 0.08                                  # e4m3 has a 3-bit mantissa


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _ep_worker(rank, world, port, dedup, q):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    x, idx, w, w13, w2 = problem(N=40 * world, E=8, k=3)
    sl = slice(rank * 40, (rank + 1) * 40)              # each rank owns 40 tokens and E/world experts
    El = 8 // world
    xs, ws = x[sl].clone().requires_grad_(), w[sl].clone().requires_grad_()
    a = w13[rank * El:(rank + 1) * El].clone().requires_grad_()
    b = w2[rank * El:(rank + 1) * El].clone().requires_grad_()
    out, rows, _ = dispatch_combine(xs, idx[sl], ws, a, b, group=dist.group.WORLD, dedup=dedup)
    out.square().sum().backward()
    q.put((rank, out.detach(), xs.grad, ws.grad, a.grad, b.grad, rows))
    dist.destroy_process_group()


@pytest.mark.parametrize('dedup', [True, False])
def test_expert_parallel_matches_single_process(dedup):
    world = 2
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_ep_worker, args=(r, world, port, dedup, q)) for r in range(world)]
    for p in procs:
        p.start()
    res = sorted([q.get(timeout=120) for _ in procs], key=lambda t: t[0])
    for p in procs:
        p.join(timeout=60)
    x, idx, w, w13, w2 = [t.clone() for t in problem(N=40 * world, E=8, k=3)]
    ts = [t.requires_grad_() for t in (x, w, w13, w2)]
    ref = dispatch_combine(ts[0], idx, ts[1], ts[2], ts[3])[0]
    ref.square().sum().backward()
    assert torch.allclose(torch.cat([r[1] for r in res]), ref, atol=1e-10)
    assert torch.allclose(torch.cat([r[2] for r in res]), ts[0].grad, atol=1e-10)
    assert torch.allclose(torch.cat([r[3] for r in res]), ts[1].grad, atol=1e-10)
    # expert grads on their owning rank already sum all ranks' tokens (all-to-all backward)
    assert torch.allclose(torch.cat([r[4] for r in res]), ts[2].grad, atol=1e-10)
    assert torch.allclose(torch.cat([r[5] for r in res]), ts[3].grad, atol=1e-10)
    rows = sum(r[6] for r in res)
    dest_pairs = len({(n, int(e) // 4) for n in range(idx.shape[0]) for e in idx[n]})
    assert rows == (dest_pairs if dedup else idx.numel())

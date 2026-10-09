import torch

from model.config import get_config
from model.moe import Router


def make(**kw):
    c = get_config('tiny', **kw)
    r = Router(c)
    torch.nn.init.normal_(r.weight, std=0.02)
    return c, r


def test_weights_normalised_and_topk_distinct():
    c, r = make()
    x = torch.randn(64, c.d_model)
    idx, w, aux = r(x, 4)
    assert idx.shape == (64, c.top_k)
    assert torch.allclose(w.sum(-1), torch.ones(64))
    assert all(len(set(row.tolist())) == c.top_k for row in idx)
    assert aux > 0


def test_group_limited_routing():
    c, r = make(n_experts=16, top_k=4, route_groups=8, route_topk_groups=2)
    idx, _, _ = r(torch.randn(256, c.d_model), 4)
    groups = idx // (16 // 8)
    assert max(len(set(g.tolist())) for g in groups) <= 2


def test_bias_update_direction():
    c, r = make()
    load = torch.tensor([10., 0, 5, 5, 5, 5, 5, 5])
    r.update_bias(load)
    assert r.bias[0] < 0 < r.bias[1]                    # overloaded down, underloaded up
    assert torch.all(r.bias[2:] == 0)                     # exactly at the mean (5) -> unchanged
    assert r.load.sum() == 0


def test_bias_balances_skewed_router():
    """A router with a strong built-in preference becomes balanced through the bias alone."""
    c, r = make(route_groups=0)      # default u = 1e-3
    r.eval()
    with torch.no_grad():
        r.weight[0] += 0.5                          # expert 0 gets a large positive logit shift
    x = torch.randn(2048, c.d_model).abs()
    def maxvio():
        idx, _, _ = r(x, 1)
        load = torch.bincount(idx.flatten(), minlength=c.n_experts).float()
        return (load.max() / load.mean() - 1).item(), load
    before, _ = maxvio()
    tail = torch.zeros(c.n_experts)
    for i in range(1000):
        _, load = maxvio()
        r.update_bias(load)
        if i >= 800:
            tail += load
    after = (tail.max() / tail.mean() - 1).item()     # sign updates oscillate with amplitude ~u; judge the average load
    assert before > 1.0 and after < 0.1
    assert r.bias[0] == r.bias.min()


def test_load_counted_only_in_training():
    c, r = make()
    x = torch.randn(32, c.d_model)
    r.eval(); r(x, 1)
    assert r.load.sum() == 0
    r.train(); r(x, 1)
    assert r.load.sum() == 32 * c.top_k


def test_bias_does_not_change_gate_values():
    c, r = make(route_groups=0)
    x = torch.randn(16, c.d_model)
    idx, w, _ = r(x, 1)
    r.bias.fill_(0.3)                  # uniform shift: same selection, same weights
    idx2, w2, _ = r(x, 1)
    assert torch.equal(idx, idx2) and torch.allclose(w, w2)

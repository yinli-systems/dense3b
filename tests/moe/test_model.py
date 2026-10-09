import pytest
import torch

from model.config import PRESETS, get_config, param_counts
from model.model import Transformer, init_weights


@pytest.mark.parametrize('name', sorted(PRESETS))
def test_param_counts_match_module(name):
    c = PRESETS[name]
    with torch.device('meta'):
        m = Transformer(c)
    assert sum(p.numel() for p in m.parameters()) == param_counts(c)['total']


def test_design_doc_numbers():
    p = param_counts(PRESETS['moe7b'])
    assert p['total'] == 6_698_637_312
    assert p['active'] == 1_036_326_912
    assert p['active_nonembed'] == 830_281_728
    b = param_counts(PRESETS['olmoe'])            # OLMoE-1B-7B: 6.9B total / 1.3B active
    assert round(b['total'] / 1e9, 1) == 6.9 and round(b['active'] / 1e9, 1) == 1.3


def test_shapes_and_aux():
    c = get_config('tiny')
    m = Transformer(c)
    init_weights(m)
    x = torch.randint(0, c.vocab_size, (2, 32))
    assert m(x).shape == (2, 32, c.vocab_size)
    out = m(x, torch.roll(x, -1, 1))
    assert out['loss'].ndim == 0 and torch.isfinite(out['loss'])
    assert abs(out['ce'].item() - torch.log(torch.tensor(c.vocab_size)).item()) < 0.2   # near-uniform at init
    assert out['aux'] > 0


def test_init_independent_of_ep_layout():
    c = get_config('tiny')
    full = Transformer(c)
    init_weights(full, seed=3)
    for r in range(2):
        half = Transformer(c, ep_size=2)
        init_weights(half, seed=3, ep_rank=r)
        El = c.n_experts // 2
        for a, b in zip(full.moe_layers(), half.moe_layers()):
            assert torch.equal(a.w13[r * El:(r + 1) * El], b.w13)
            assert torch.equal(a.w2[r * El:(r + 1) * El], b.w2)
            assert torch.equal(a.router.weight, b.router.weight)


def test_qk_norm_and_gate_present():
    m = Transformer(get_config('tiny'))
    att = m.layers[0].attn
    assert att.q_norm is not None and att.wg is not None and att.kv < att.h


def test_dense3b_is_dense_and_3b():
    c = PRESETS['dense3b']
    assert c.n_dense_layers == c.n_layers
    p = param_counts(c)
    assert p['routed_experts'] == 0 and p['router'] == 0 and p['shared_experts'] == 0
    assert 3.2e9 < p['total'] < 3.3e9

import torch

from model.config import PRESETS, get_config
from model.moe import Router
from tools.budget import a2a_bytes_per_token, expected_dest_ranks, flops_per_token, markdown, report


def test_expected_dest_ranks_matches_monte_carlo():
    g = torch.Generator().manual_seed(0)
    E, k, P = 128, 8, 8
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(4000)])
    mc = torch.tensor([len(set((row // (E // P)).tolist())) for row in idx]).float().mean().item()
    assert abs(mc - expected_dest_ranks(E, k, P)) < 0.05


def test_group_limited_rows_with_real_router():
    """Group limit M=4 with a real router: distinct ranks per token never exceed M and sit just below it,
    so the budget's use of M is a tight upper bound."""
    c = get_config('moe7b', d_model=256)
    r = Router(c)
    torch.nn.init.normal_(r.weight, std=0.02)
    idx, _, _ = r(torch.randn(4096, 256), 1)
    rows = torch.tensor([len(set((row // 16).tolist())) for row in idx]).float().mean().item()
    assert 3.8 < rows <= expected_dest_ranks(128, 8, 8, 4) == 4


def test_budget_numbers_used_in_doc():
    c = PRESETS['moe7b']
    fl, mat, att = flops_per_token(c, 4096)
    assert round(fl / 1e9, 2) == 6.40 and round(att / 1e9, 2) == 0.81
    rows, by = a2a_bytes_per_token(c, 8, dedup=True)
    rows0, by0 = a2a_bytes_per_token(c, 8, dedup=False)
    assert rows == 4 and rows0 == 8 and by == 0.5 * by0
    assert round(expected_dest_ranks(128, 8, 8), 2) == 5.34       # dedup without the group limit
    assert round(expected_dest_ranks(64, 8, 8), 2) == 5.43        # OLMoE layout with dedup
    rep = report()
    assert rep['moe7b']['mem_4090']['fits'] and not rep['moe7b']['mem_4090_norecompute']['fits']
    assert rep['moe7b']['time_4090_conservative']['tokens_15d'] > 160e9
    assert '### moe7b' in markdown(rep)

"""FP8Linear numerics against BF16 on a CUDA sm_89+ GPU (skipped elsewhere)."""
import pytest
import torch
import torch.nn as nn

from model.fp8_linear import FP8Linear

gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
                         reason='needs an FP8 tensor-core GPU')


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@gpu
@pytest.mark.parametrize('M,K,N', [(4096, 3072, 8192), (4096, 8192, 3072), (2048, 3072, 16384)])
def test_fp8_linear_matches_bf16(M, K, N):
    torch.manual_seed(0)
    ref = nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
    f8 = FP8Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
    f8.weight = nn.Parameter(ref.weight.detach().clone())
    x1 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    g = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    y1, y2 = ref(x1), f8(x2)
    y1.backward(g); y2.backward(g)
    assert rel(y2, y1) < 0.06
    assert rel(x2.grad, x1.grad) < 0.08
    assert rel(f8.weight.grad, ref.weight.grad) < 0.08


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_flex_doc_mask_matches_dense_mask_sdpa():
    import torch.nn.functional as F
    from model.layers import doc_block_mask, flex_attention
    torch.manual_seed(0)
    B, H, KV, T, D = 2, 8, 2, 1024, 128
    tok = torch.randint(2, 1000, (B, T), device='cuda')
    tok[0, 300] = tok[0, 700] = tok[1, 50] = 50279
    q = torch.randn(B, H, T, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, KV, T, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, KV, T, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    o1 = flex_attention(q, k, v, doc_block_mask(tok), enable_gqa=True)
    dense = doc_block_mask(tok.cpu()).cuda()
    o2 = F.scaled_dot_product_attention(q, k, v, attn_mask=dense, enable_gqa=True)
    assert rel(o1, o2) < 0.01
    g = torch.randn_like(o1)
    gq1, = torch.autograd.grad(o1, q, g); gq2, = torch.autograd.grad(o2, q, g)
    assert rel(gq1, gq2) < 0.02


@gpu
@pytest.mark.xfail(reason='FP8 shelved: compiled custom-op path still differs from eager (2026-10-06); not used for training', strict=False)
def test_compiled_fp8_blocks_match_eager_and_are_finite():
    from model.config import get_config
    from model.model import Transformer, init_weights
    from model.fp8_linear import convert_to_fp8
    grads = []
    for comp in (False, True):
        torch.manual_seed(0)
        m = Transformer(get_config('dense3b', n_layers=8, n_dense_layers=8))
        init_weights(m, 0); m = m.cuda().bfloat16(); convert_to_fp8(m)
        if comp:
            for b in m.layers:
                b.compile(dynamic=False)
        x = (torch.arange(4096, device='cuda')[None] * 7919) % 49999 + 2
        m(x, torch.roll(x, -1, 1))['loss'].backward()
        g = {n: p.grad.float() for n, p in m.named_parameters()}
        assert all(torch.isfinite(v).all() for v in g.values()), [n for n, v in g.items() if not torch.isfinite(v).all()]
        grads.append(g)
    for n in grads[0]:
        assert rel(grads[1][n], grads[0][n]) < 0.05, n

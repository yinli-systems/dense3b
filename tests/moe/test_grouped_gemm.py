import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from model.grouped_gemm import grouped_mm, grouped_mm_ref, grouped_wgrad_ref

ROOT = Path(__file__).resolve().parents[2]


def case(counts=(0, 70, 3, 130, 1), K=48, N=40, dtype=torch.float64):
    counts = torch.tensor(counts)
    offs = counts.cumsum(0).to(torch.int32)
    M = int(offs[-1])
    return torch.randn(M, K, dtype=dtype), torch.randn(len(counts), K, N, dtype=dtype), offs, counts


def test_reference_equals_per_expert_matmul():
    x, w, offs, counts = case()
    y = grouped_mm_ref(x, w, offs)
    rows = torch.repeat_interleave(torch.arange(len(counts)), counts)
    expect = torch.einsum('mk,mkn->mn', x, w[rows])
    assert torch.allclose(y, expect, atol=1e-12)


def test_autograd_matches_dense():
    x, w, offs, counts = case()
    rows = torch.repeat_interleave(torch.arange(len(counts)), counts)
    a, b = x.clone().requires_grad_(), w.clone().requires_grad_()
    grouped_mm(a, b, offs).sin().sum().backward()
    c, d = x.clone().requires_grad_(), w.clone().requires_grad_()
    torch.einsum('mk,mkn->mn', c, d[rows]).sin().sum().backward()
    assert torch.allclose(a.grad, c.grad, atol=1e-10) and torch.allclose(b.grad, d.grad, atol=1e-10)
    assert torch.all(b.grad[0] == 0)                       # empty expert gets an exact zero gradient


def test_wgrad_reference():
    x, w, offs, _ = case()
    g = torch.randn(x.shape[0], w.shape[2], dtype=x.dtype)
    dw = grouped_wgrad_ref(x, g, offs, w.shape[0])
    assert torch.allclose(dw[1], x[0:70].T @ g[0:70]) and torch.all(dw[0] == 0)


TRITON_CHECK = '''
import torch
from model.grouped_gemm import grouped_mm_triton, grouped_wgrad_triton, grouped_mm_ref, grouped_wgrad_ref
torch.manual_seed(0)
for counts in [(0, 70, 3, 130, 1), (64, 64), (0, 0, 5)]:
    counts = torch.tensor(counts); offs = counts.cumsum(0).int(); E = len(counts); M = int(offs[-1])
    x = torch.randn(M, 96); w = torch.randn(E, 96, 80); g = torch.randn(M, 80)
    assert torch.allclose(grouped_mm_triton(x, w, offs), grouped_mm_ref(x, w, offs), atol=1e-3)
    wt = w.transpose(1, 2)
    assert torch.allclose(grouped_mm_triton(g, wt, offs), grouped_mm_ref(g, wt, offs), atol=1e-3)
    assert torch.allclose(grouped_wgrad_triton(x, g, offs, E), grouped_wgrad_ref(x, g, offs, E), atol=1e-3)
print('ok')
'''


def test_triton_kernels_in_interpreter():
    pytest.importorskip('triton')
    env = dict(os.environ, TRITON_INTERPRET='1', PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, '-c', TRITON_CHECK], env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 and 'interpret' in r.stderr.lower() and 'not supported' in r.stderr.lower():
        pytest.skip('Triton interpreter unavailable')
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr[-2000:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU-only: real Triton kernel')
def test_triton_kernel_on_gpu_bf16():
    x, w, offs, _ = case(dtype=torch.bfloat16)
    x, w, offs = x.cuda(), w.cuda(), offs.cuda()
    a = grouped_mm(x, w, offs).float()
    assert torch.allclose(a, grouped_mm_ref(x.float(), w.float(), offs), atol=0.5, rtol=2e-2)

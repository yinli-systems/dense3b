"""FP8 hooks (off by default). Row-wise e4m3 quantisation, used for the forward dispatch payload
(DeepSeek-V3 sends dispatch activations in FP8, combine in BF16). The same helpers are the entry point
for an FP8 expert GEMM; embeddings, LM head, router, norms and attention softmax always stay >= BF16."""
from dataclasses import dataclass
import torch

E4M3_MAX = 448.0
HIGH_PRECISION = ('embed', 'head', 'router', 'norm', 'attention softmax', 'optimizer state', 'master weights')


@dataclass(frozen=True)
class FP8Policy:
    dispatch: bool = False        # quantise the forward all-to-all payload (halves dispatch bytes)
    expert_gemm: bool = False     # reserved: FP8 grouped GEMM (needs per-GPU validation, see design doc)


def quant_rowwise(x):
    """x (M, D) -> (q float8_e4m3fn (M, D), inv_scale float32 (M, 1))."""
    amax = x.detach().abs().amax(-1, keepdim=True).float().clamp(min=1e-12)
    scale = E4M3_MAX / amax
    return (x.float() * scale).to(torch.float8_e4m3fn), scale.reciprocal()


def dequant_rowwise(q, inv_scale, dtype):
    return (q.float() * inv_scale).to(dtype)

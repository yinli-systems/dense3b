"""FP8 Linear for Ada/Blackwell GeForce (sm_89+): 2x the BF16 tensor-core rate with FP32 accumulation.

Same recipe as torchao's Float8Linear with dynamic tensorwise scaling: inputs and weights in e4m3, output
gradients in e5m2, every GEMM accumulates in FP32 and returns BF16. The forward saves the FP8 copies of x and w
for backward, so a Linear input costs half the activation memory. Embedding, LM head, norms, attention
softmax, master weights and optimizer state never touch FP8 (see fp8.HIGH_PRECISION).

torch._scaled_mm(a, b): a row-major (M, K), b column-major (K, N), K and N multiples of 16.
"""
import torch
import torch.nn as nn

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
MAX = {E4M3: 448.0, E5M2: 57344.0}


def _quant(t, dtype):
    """Dynamic tensorwise scaling -> (fp8 tensor, inverse scale as fp32 scalar tensor)."""
    amax = t.detach().abs().amax().float().clamp(min=1e-12)
    scale = MAX[dtype] / amax
    return (t.float() * scale).clamp(-MAX[dtype], MAX[dtype]).to(dtype), scale.reciprocal()


def _mm(a, a_inv, b, b_inv):
    return torch._scaled_mm(a, b, scale_a=a_inv, scale_b=b_inv, out_dtype=torch.bfloat16, use_fast_accum=False)


# The FP8 GEMMs are registered as opaque custom ops: under torch.compile, Inductor handling the saved FP8
# activation of the fused QKVG projection produced non-finite weight gradients (eager was correct). Inside the
# ops everything runs eagerly; compile still fuses everything around them.
@torch.library.custom_op('dense3b::fp8_linear_fwd', mutates_args=())
def fp8_linear_fwd(x2: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor]:
    xq, xi = _quant(x2, E4M3)
    wq, wi = _quant(w, E4M3)
    return _mm(xq, xi, wq.t(), wi), xq, xi, wq, wi        # (M,K) @ (K,N): w.t() of row-major (N,K) is col-major


@fp8_linear_fwd.register_fake
def _(x2, w):
    M, N = x2.shape[0], w.shape[0]
    s = x2.new_empty((), dtype=torch.float32)
    return (x2.new_empty(M, N, dtype=torch.bfloat16), x2.new_empty(x2.shape, dtype=E4M3), s,
            w.new_empty(w.shape, dtype=E4M3), s.clone())


@torch.library.custom_op('dense3b::fp8_linear_bwd', mutates_args=())
def fp8_linear_bwd(gy: torch.Tensor, xq: torch.Tensor, xi: torch.Tensor, wq: torch.Tensor,
                   wi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gq, gi = _quant(gy, E5M2)
    gx = _mm(gq, gi, wq.t().contiguous().t(), wi)                    # (M,N) @ (N,K), w column-major
    gw = _mm(gq.t().contiguous(), gi, xq.t().contiguous().t(), xi)  # (N,M) @ (M,K), x column-major
    return gx, gw


@fp8_linear_bwd.register_fake
def _(gy, xq, xi, wq, wi):
    return gy.new_empty(xq.shape, dtype=torch.bfloat16), gy.new_empty(wq.shape, dtype=torch.bfloat16)


def _setup(ctx, inputs, output):
    _, xq, xi, wq, wi = output
    ctx.save_for_backward(xq, xi, wq, wi)


def _backward(ctx, gy, *unused):
    gx, gw = fp8_linear_bwd(gy.contiguous(), *ctx.saved_tensors)
    return gx, gw


fp8_linear_fwd.register_autograd(_backward, setup_context=_setup)


def fp8_matmul(x, w):
    y = fp8_linear_fwd(x.reshape(-1, x.shape[-1]).contiguous(), w)[0]
    return y.view(*x.shape[:-1], w.shape[0])


class FP8Linear(nn.Linear):
    """nn.Linear (no bias) whose forward/backward GEMMs run in FP8 on CUDA; plain BF16 elsewhere."""

    def forward(self, x):
        if x.is_cuda and torch.is_grad_enabled() and self.weight.dtype == torch.bfloat16:
            return fp8_matmul(x, self.weight)
        return super().forward(x)


def convert_to_fp8(model):
    """Swap every bias-free nn.Linear inside the transformer blocks (attention + MLP) for FP8Linear in place."""
    n = 0
    for blk in model.layers:
        for name, mod in list(blk.named_modules()):
            for cname, child in list(mod.named_children()):
                if type(child) is nn.Linear and child.bias is None and child.in_features % 16 == 0 \
                        and child.out_features % 16 == 0:
                    new = FP8Linear(child.in_features, child.out_features, bias=False, device=child.weight.device,
                                    dtype=child.weight.dtype)
                    new.weight = child.weight                       # same Parameter (keeps _muon_rows, views)
                    setattr(mod, cname, new)
                    n += 1
    return n

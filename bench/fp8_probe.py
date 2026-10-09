"""Locate the first non-finite tensor in an FP8 forward/backward of one dense3b block (eager)."""
import torch
from model.config import get_config
from model.model import Transformer, init_weights
from model.fp8_linear import convert_to_fp8, _quant, E4M3, E5M2

torch.manual_seed(0)
cfg = get_config('dense3b', n_layers=1, n_dense_layers=1, doc_mask=False)
m = Transformer(cfg); init_weights(m, 0); m = m.cuda().bfloat16()
convert_to_fp8(m)
bad = []
def hook(name):
    def f(mod, gin, gout):
        for i, g in enumerate(gin):
            if g is not None and not torch.isfinite(g).all():
                bad.append(f'{name} grad_input[{i}] nonfinite; amax={g.float().abs().nan_to_num(0).amax().item():.3e}')
    return f
for n, mod in m.named_modules():
    if isinstance(mod, torch.nn.Linear):
        mod.register_full_backward_hook(hook(n))
x = torch.randint(2, 50000, (1, 4096), device='cuda'); y = torch.roll(x, -1, 1)
out = m(x, y); print('loss', out['loss'].item())
out['loss'].backward()
for n, p in m.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print('nonfinite param grad', n)
print('\n'.join(bad) or 'no nonfinite grad_inputs')
t = torch.randn(64, 64, device='cuda') * 1e-3
for dt in (E4M3, E5M2):
    q, inv = _quant(t, dt); print(dt, 'roundtrip rel err', ((q.float() * inv - t).norm() / t.norm()).item())

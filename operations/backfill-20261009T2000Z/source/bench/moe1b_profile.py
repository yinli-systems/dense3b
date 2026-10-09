"""Kernel-time breakdown of one fwd+bwd of a preset on one GPU (torch.profiler), grouped into categories.

  python -m bench.moe1b_profile --preset moe1b --micro 2 --ckpt 3 --out prof.json
"""
import argparse, collections, json, time
import torch
from model.config import get_config
from model.model import Transformer, init_weights
from train.train_dense import ckpt_indices

CATS = [('expert_gmm', ('_gmm_kernel',)), ('expert_wgrad', ('_wgrad_kernel',)), ('combine', ('_combine_kernel',)),
        ('attention', ('flex', 'attention', 'fmha', 'flash')), ('gemm_dense', ('gemm', 'sm80_xmma', 'cutlass', 'ampere', 'sm89', 'nvjet')),
        ('swiglu_moe', ('_swiglu',)), ('sort_route', ('sort', 'radix', 'scatter', 'index_add', 'cumsum', 'topk')),
        ('loss_head', ('cross_entropy', 'log_softmax', 'nll')), ('triton_fused', ('triton_',)), ('copy', ('copy', 'cat'))]


def cat(name):
    n = name.lower()
    for c, keys in CATS:
        if any(k in n for k in keys):
            return c
    return 'other'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='moe1b'); p.add_argument('--micro', type=int, default=2)
    p.add_argument('--ckpt', type=int, default=3); p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = torch.device('cuda'); c = get_config(a.preset)
    model = Transformer(c); init_weights(model, 0); model = model.to(dev, torch.bfloat16)
    model.cos, model.sin = model.cos.float(), model.sin.float()
    model.ckpt_layers = ckpt_indices(c.n_layers, a.ckpt)
    for prm in model.parameters():
        prm.grad = torch.zeros_like(prm)
    for blk in model.layers:
        blk.compile(dynamic=False)
    g = torch.Generator().manual_seed(1)
    tok = torch.randint(0, 50000, (a.micro, 4097), generator=g); tok[torch.rand(tok.shape, generator=g) < 1 / 900] = c.eos_id
    x, y = tok[:, :-1].to(dev), tok[:, 1:].to(dev)
    for _ in range(5):
        model(x, y)['loss'].backward()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            model(x, y)['loss'].backward()
        torch.cuda.synchronize()
    agg = collections.defaultdict(float); top = collections.defaultdict(float)
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA or getattr(e, 'self_device_time_total', 0) > 0:
            t = getattr(e, 'self_device_time_total', 0) / 3 / 1e3
            agg[cat(e.key)] += t; top[e.key[:90]] += t
    tot = sum(agg.values())
    res = dict(total_ms=tot, tok_per_s=a.micro * 4096 / tot * 1e3,
               cats={k: round(v, 2) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])},
               top=[(k, round(v, 2)) for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:30]])
    json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

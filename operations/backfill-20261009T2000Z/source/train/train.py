"""Training entry point. Single process (CPU smoke runs) or torchrun across nodes:

  torchrun --nnodes 6 --nproc-per-node 8 ... -m train.train --preset moe7b --data '/data/shards/*.u16' \
      --global-batch 960 --micro-batch 1 --steps 40700 --ep 8 --recompute-experts --ckpt-dir /local/ckpt
"""
import argparse
import json
import math
import time

import torch
import torch.distributed as dist

from model.config import get_config
from model.fp8 import FP8Policy
from model.model import Transformer, init_weights
from train import ckpt
from train.data import PackedTokens, shard_paths
from train.optim import MuonAdamW, param_groups, wsd_lr
from train.parallel import (ExpertGradSync, apply_parallel, build_mesh, clip_grad_norm, expert_params,
                            init_distributed, update_router_bias)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='moe7b')
    p.add_argument('--override', default='{}', help='JSON dict of MoEConfig overrides')
    p.add_argument('--data', required=True, help='comma-separated shard globs, optional :R repeat suffix')
    p.add_argument('--steps', type=int, default=40700, help='40,700 x 960 x 4096 = 160B tokens')
    p.add_argument('--global-batch', type=int, default=960, help='sequences per optimizer step')
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--cooldown-frac', type=float, default=0.2)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--optimizer', choices=['muon', 'adamw'], default='muon')
    p.add_argument('--ep', type=int, default=8)
    p.add_argument('--no-dedup', action='store_true')
    p.add_argument('--fp8-dispatch', action='store_true')
    p.add_argument('--recompute-experts', action='store_true', help='needed on 24 GB cards (see budget)')
    p.add_argument('--grad-sync-dtype', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ckpt-dir', default=None)
    p.add_argument('--ckpt-every', type=int, default=200)
    p.add_argument('--async-ckpt', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--metrics', default=None, help='JSONL metrics file (rank 0)')
    p.add_argument('--timing', action='store_true', help='synchronise to split step time into phases')
    p.add_argument('--profile-dir', default=None, help='torch.profiler chrome traces (one per rank)')
    p.add_argument('--profile-steps', default='3,5', help='first,last step to profile')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


def router_stats(load):
    """Per-layer MaxVio (max/mean - 1, as in arXiv 2408.15664) and normalised load entropy."""
    p = load / load.sum(-1, keepdim=True).clamp(min=1)
    ent = -(p * p.clamp(min=1e-12).log()).sum(-1) / math.log(load.shape[-1])
    vio = load.max(-1).values / load.mean(-1).clamp(min=1e-9) - 1
    return dict(maxvio_mean=vio.mean().item(), maxvio_max=vio.max().item(), load_entropy_min=ent.min().item())


def main(argv=None):
    a = parse(argv)
    distributed = init_distributed()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    dev = torch.device(a.device if a.device != 'cuda' else f'cuda:{torch.cuda.current_device()}')
    cfg = get_config(a.preset, **json.loads(a.override))
    ep = a.ep if distributed else 1
    mesh = build_mesh(ep, dev.type) if distributed else None
    ep_rank = mesh['shard'].get_local_rank() if mesh else 0

    torch.manual_seed(a.seed)
    model = Transformer(cfg, ep).to(dev)
    init_weights(model, a.seed, ep_rank)
    for m in model.moe_layers():
        m.dedup, m.fp8, m.recompute = not a.no_dedup, FP8Policy(dispatch=a.fp8_dispatch), a.recompute_experts
    sync, shard_group = None, None
    if mesh is not None:
        apply_parallel(model, mesh, bf16=dev.type == 'cuda')
        sync = ExpertGradSync(expert_params(model), mesh,
                              torch.bfloat16 if a.grad_sync_dtype == 'bf16' else torch.float32)
        shard_group = mesh['shard'].get_group()
    params = [p for p in model.parameters()]
    opt = MuonAdamW(param_groups(model, a.optimizer == 'muon', a.weight_decay), lr=a.lr)
    data = PackedTokens(shard_paths(a.data), cfg.max_seq_len, a.seed)
    n_micro = a.global_batch // (world * a.micro_batch)
    assert n_micro * world * a.micro_batch == a.global_batch, 'global batch must split evenly'

    start = 0
    if a.resume and a.ckpt_dir and (path := ckpt.latest(a.ckpt_dir, world)) is not None:
        start, _ = ckpt.load(path, model, opt, rank)
        start += 1
    logf = open(a.metrics, 'a') if (a.metrics and rank == 0) else None
    history, pending, prof = [], None, None
    p0, p1 = map(int, a.profile_steps.split(','))
    cuda = dev.type == 'cuda'

    def mark():
        if a.timing and cuda:
            torch.cuda.synchronize()
        return time.time()

    model.train()
    for step in range(start, a.steps):
        if a.profile_dir and step == p0:
            prof = torch.profiler.profile(record_shapes=True, with_stack=False)
            prof.__enter__()
        t0 = mark()
        lr = wsd_lr(step, a.steps, a.lr, a.warmup, a.cooldown_frac)
        for g in opt.param_groups:
            g['lr'] = lr
        ce = aux = torch.zeros((), device=dev)
        rows = pairs = dropped = 0
        for mi in range(n_micro):
            last = mi == n_micro - 1
            if mesh is not None:
                model.set_requires_all_reduce(last)      # HSDP: intra-node reduce-scatter each micro, inter-node once
                sync.active = last
            x, y = data.batch(step, rank, world, mi, a.micro_batch, a.global_batch)
            out = model(x.to(dev), y.to(dev))
            (out['loss'] / n_micro).backward()
            ce, aux = ce + out['ce'] / n_micro, aux + out['aux'] / n_micro
            for m in model.moe_layers():
                rows += m.stats['a2a_rows']
                pairs += m.stats['pairs']
                dropped = dropped + (m.stats['pairs'] - m.stats['slots'])      # device tensor: no sync
        t1 = mark()
        if sync is not None:
            sync.wait()
        t2 = mark()
        gnorm = clip_grad_norm(params, a.clip, shard_group)
        opt.step()
        opt.zero_grad(set_to_none=True)
        load = update_router_bias(model)
        if distributed:
            dist.all_reduce(ce, op=dist.ReduceOp.AVG)
        t3 = mark()
        if prof is not None and step == p1:
            prof.__exit__(None, None, None)
            prof.export_chrome_trace(f'{a.profile_dir}/trace_rank{rank:03d}.json')
            prof = None
        rec = dict(step=step, lr=lr, loss=ce.item(), aux=aux.item(), grad_norm=gnorm.item(),
                   tok_per_s=a.global_batch * cfg.max_seq_len / (time.time() - t0),
                   a2a_rows_per_pair=rows / max(pairs, 1), dropped=int(dropped))
        if a.timing:
            rec.update(t_fwd_bwd=t1 - t0, t_grad_sync_wait=t2 - t1, t_optim=t3 - t2)
        if cuda:
            mem = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device=dev)
            if distributed:
                dist.all_reduce(mem, op=dist.ReduceOp.MAX)
            rec['max_mem_gb'] = mem.item()
        if load is not None:
            rec.update(router_stats(load))
        history.append(rec)
        if logf:
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
        if rank == 0 and (step % 10 == 0 or step == a.steps - 1):
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if a.ckpt_dir and ((step + 1) % a.ckpt_every == 0 or step == a.steps - 1):
            if pending is not None:
                pending.join()
            pending = ckpt.save(a.ckpt_dir, step, model, opt, rank, world, dict(config=cfg.to_dict()),
                                async_save=a.async_ckpt)
    if pending is not None:
        pending.join()
    if distributed:
        dist.destroy_process_group()
    return history, model


if __name__ == '__main__':
    main()

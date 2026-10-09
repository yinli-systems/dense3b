"""Dense-model training entry point built for throughput on PCIe-only GPU nodes (see train/zero.py).

  torchrun --nnodes N --nproc-per-node 8 ... -m train.train_dense --preset dense3b --data '<globs>' \
      --seq-len 4096 --global-batch 256 --micro-batch 1 --ckpt-layers 18 --compile --steps 95000

Knobs that trade memory for speed: --ckpt-layers (blocks recomputed in backward), --micro-batch, --seq-len.
Logged per step: tokens/s, MFU (model FLOPs, recompute not counted), phase times, peak memory.
"""
import argparse
import glob
import json
import os
import time

import torch
import torch.distributed as dist

from model.config import get_config
from model.model import Transformer, init_weights
from train.data import PackedTokens, shard_paths
from train.optim import wsd_lr
from train.zero import OwnerZeRO, node_groups

PEAK_TFLOPS = {'NVIDIA GeForce RTX 4090': 165.2, 'NVIDIA GeForce RTX 5090': 209.5}   # dense BF16, FP32 accum


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='dense3b')
    p.add_argument('--override', default='{}', help='JSON dict of config overrides')
    p.add_argument('--data', required=True, help='comma-separated shard globs, optional :R repeat suffix')
    p.add_argument('--seq-len', type=int, default=0, help='default: config max_seq_len')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--global-batch', type=int, default=256, help='sequences per optimizer step')
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--cooldown-frac', type=float, default=0.2)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--optimizer', choices=['muon', 'adamw'], default='muon')
    p.add_argument('--ckpt-layers', type=int, default=0, help='number of blocks recomputed in backward')
    p.add_argument('--compile', action='store_true', help='torch.compile every block')
    p.add_argument('--act-budget', type=float, default=1.0,
                   help='compiled blocks: fraction of activation memory the min-cut partitioner may keep (<1 recomputes '
                        'the cheapest ops instead of whole blocks)')
    p.add_argument('--fp8', action='store_true', help='FP8 (e4m3/e5m2, tensorwise) GEMMs for block Linears')
    p.add_argument('--shard-scope', choices=['node', 'world'], default='node',
                   help='owner-shard optimizer state per node (full nodes) or over all ranks (scattered GPUs)')
    p.add_argument('--sync-every', type=int, default=0,
                   help='reduce partial grads every N micro-batches into fp32 (0 = once per step, bf16 accumulation)')
    p.add_argument('--dtype', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ckpt-dir', default=None)
    p.add_argument('--ckpt-every', type=int, default=500)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--keep-last', type=int, default=3, help='full (resumable) checkpoints kept')
    p.add_argument('--milestone-every', type=int, default=5000, help='also keep the BF16 model of every N-th step')
    p.add_argument('--metrics', default=None)
    p.add_argument('--timing', action='store_true', help='synchronise to split step time into phases')
    p.add_argument('--profile-dir', default=None)
    p.add_argument('--profile-steps', default='3,5')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--debug-nan', action='store_true', help='report the first micro-batch / parameter with a non-finite grad')
    p.add_argument('--val-data', default=None, help='held-out shard globs; evaluated every --val-every steps')
    p.add_argument('--val-every', type=int, default=250)
    p.add_argument('--val-windows', type=int, default=64, help='fixed windows evaluated (split across ranks)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(argv)


def flops_per_token(cfg, seq_len):
    """Training FLOPs per token: 6 x matmul params (incl. LM head) + causal attention scores and values."""
    d, hd = cfg.d_model, cfg.head_dim
    attn_w = d * cfg.n_heads * hd * 2 + d * cfg.n_kv_heads * hd * 2 + (d * cfg.n_heads * hd if cfg.attn_gate else 0)
    mlp_w = 3 * d * cfg.d_ff_dense
    matmul = cfg.n_layers * (attn_w + mlp_w) + cfg.vocab_size * d
    attn = cfg.n_layers * 2 * 2 * cfg.n_heads * hd * seq_len / 2          # QK^T and AV, causal half, forward
    return 6 * matmul + 3 * attn


def ckpt_indices(n_layers, k):
    """k blocks spread evenly over the depth."""
    return {int(j * n_layers / k) for j in range(k)} if k > 0 else set()


def init_dist():
    if 'RANK' not in os.environ:
        return False
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    import datetime
    # a stuck collective (e.g. a bad network path) aborts after 20 min instead of holding the GPUs forever
    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo', timeout=datetime.timedelta(minutes=20))
    return True


# ------------------------------------------------------------------ checkpoints (node 0 writes, all nodes read)

def latest(ckpt_dir):
    done = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*', 'COMPLETE')))
    return os.path.dirname(done[-1]) if done else None


def save(ckpt_dir, step, model, zero, rank, L, meta):
    """Every writer rank stores its owned parameters by name (master + optimizer state), so the checkpoint
    resumes on any number of ranks; rank 0 also stores the BF16 model for evaluation."""
    d = os.path.join(ckpt_dir, f'step_{step:08d}')
    if rank < L:                    # one copy of every owner shard (node 0; in world scope every rank)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f'.rank{rank:03d}.pt.tmp')
        torch.save(zero.named_state(), tmp)
        os.replace(tmp, os.path.join(d, f'rank{rank:03d}.pt'))
    if rank == 0:
        torch.save({k: v for k, v in model.state_dict().items()}, os.path.join(d, 'model.pt'))
        json.dump(dict(meta, step=step, writers=L), open(os.path.join(d, 'meta.json'), 'w'))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        open(os.path.join(d, 'COMPLETE'), 'w').close()


def load(path, zero, rank, L):
    files = sorted(glob.glob(os.path.join(path, 'rank*.pt')))
    zero.load_named_state(torch.load(f, map_location='cpu', mmap=True, weights_only=False) for f in files)
    return json.load(open(os.path.join(path, 'meta.json')))['step']


@torch.no_grad()
def evaluate(model, val, n_windows, rank, world, dev):
    """Mean token CE over the first n_windows windows of the (fixed-seed) validation stream."""
    tot = torch.zeros(2, device=dev, dtype=torch.float64)
    for i in range(rank, n_windows, world):
        w = torch.from_numpy(val.window(i)).to(dev)[None]
        x, y = w[:, :-1], w[:, 1:].clone()
        y[y == 1] = -100
        ce = model(x, y)['ce']
        n = (y != -100).sum()
        tot += torch.stack([ce.double() * n, n.double()])
    if dist.is_initialized():
        dist.all_reduce(tot)
    return (tot[0] / tot[1]).item()


# ------------------------------------------------------------------ main

def main(argv=None):
    a = parse(argv)
    distributed = init_dist()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    dev = torch.device(a.device if a.device != 'cuda' else f'cuda:{torch.cuda.current_device()}')
    cfg = get_config(a.preset, **json.loads(a.override))
    seq = a.seq_len or cfg.max_seq_len
    assert seq <= cfg.max_seq_len
    dtype = torch.bfloat16 if a.dtype == 'bf16' else torch.float32

    torch.manual_seed(a.seed)
    model = Transformer(cfg)
    init_weights(model, a.seed)
    model = model.to(dev, dtype)
    model.cos, model.sin = model.cos.float(), model.sin.float()
    model.ckpt_layers = ckpt_indices(cfg.n_layers, a.ckpt_layers)
    node, cross = node_groups(a.shard_scope) if distributed else (None, None)
    L = dist.get_world_size(node) if distributed else 1
    zero = OwnerZeRO(model, a.lr, a.weight_decay, a.optimizer == 'muon', dtype, node, cross, a.sync_every)
    if a.fp8:
        from model.fp8_linear import convert_to_fp8
        convert_to_fp8(model)
    if a.compile and a.act_budget < 1.0:
        import torch._functorch.config as fconf
        fconf.activation_memory_budget = a.act_budget
    if a.compile:
        for blk in model.layers:
            blk.compile(dynamic=False)

    data = PackedTokens(shard_paths(a.data), seq, a.seed, local_cache=os.environ.get('MOE_DATA_CACHE') or None)
    n_micro = a.global_batch // (world * a.micro_batch)
    assert n_micro * world * a.micro_batch == a.global_batch, 'global batch must split evenly'
    val = PackedTokens(shard_paths(a.val_data), seq, seed=12345) if a.val_data else None
    # per-source held-out sets (one per comma-separated --val-data entry), named after the shard directory
    val_sets = {}
    for spec in (a.val_data.split(',') if a.val_data else []):
        name = os.path.basename(os.path.dirname(spec.split(':')[0])) or spec
        val_sets[name] = PackedTokens(shard_paths(spec), seq, seed=12345)
    fpt = flops_per_token(cfg, seq)
    gpu = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'
    peak = PEAK_TFLOPS.get(gpu, 0) * 1e12

    start = 0
    if a.resume and a.ckpt_dir and (path := latest(a.ckpt_dir)):
        start = load(path, zero, rank, L) + 1
    logf = open(a.metrics, 'a') if (a.metrics and rank == 0) else None
    history, prof = [], None
    p0, p1 = map(int, a.profile_steps.split(','))
    cuda = dev.type == 'cuda'

    def mark():
        if a.timing and cuda:
            torch.cuda.synchronize()
        return time.time()

    if rank == 0:
        print(json.dumps(dict(event='start', gpu=gpu, world=world, nodes=world // L, seq=seq, n_micro=n_micro,
                              tokens_per_step=a.global_batch * seq, flops_per_token=fpt, owner_segment=zero.S,
                              ckpt_layers=sorted(model.ckpt_layers), compile=a.compile, fp8=a.fp8, start=start)), flush=True)
    # Read future micro-batches on CPU threads so random JuiceFS page faults do not stall the GPU loop.
    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(8)

    def fetch(st, mi):
        x, y = data.batch(st, rank, world, mi, a.micro_batch, a.global_batch)
        return (x.pin_memory(), y.pin_memory()) if cuda else (x, y)

    order = ((st, mi) for st in range(start, a.steps) for mi in range(n_micro))
    pending = [pool.submit(fetch, *nxt) for nxt in [next(order, None) for _ in range(32)] if nxt]
    t_data = [0.0]

    def next_batch():
        f = pending.pop(0)
        nxt = next(order, None)
        if nxt:
            pending.append(pool.submit(fetch, *nxt))
        tw = time.time()
        r = f.result()
        t_data[0] += time.time() - tw
        return r

    # Avoid unsynchronised full Python GC pauses turning one rank into a collective straggler.
    import gc
    gc.collect(); gc.freeze(); gc.disable()

    model.train()
    for step in range(start, a.steps):
        if a.profile_dir and step == p0:
            prof = torch.profiler.profile(record_shapes=True)
            prof.__enter__()
        t0 = mark()
        if cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        ce = torch.zeros((), device=dev)
        t_comm = 0.0
        for mi in range(n_micro):
            x, y = next_batch()
            out = model(x.to(dev, non_blocking=True), y.to(dev, non_blocking=True))
            (out['loss'] / n_micro).backward()
            ce += out['ce'] / n_micro
            if a.debug_nan and not torch.isfinite(zero.G).all():
                badp = [n for n, p in model.named_parameters() if not torch.isfinite(p.grad).all()]
                print(json.dumps(dict(event='nonfinite_grad', step=step, micro=mi, loss=out['loss'].item(),
                                      params=badp[:8], n_params=len(badp))), flush=True)
                raise SystemExit(3)
            tc = mark()
            zero.after_backward(mi, n_micro)
            t_comm += mark() - tc
        t1 = mark()
        mem_fb = torch.cuda.max_memory_allocated(dev) / 1e9 if cuda else 0.0      # forward/backward peak
        lr = wsd_lr(step, a.steps, a.lr, a.warmup, a.cooldown_frac)
        gnorm = zero.step(lr, a.clip)
        if distributed:
            dist.all_reduce(ce, op=dist.ReduceOp.SUM)
            ce /= world
        t2 = mark()
        if prof is not None and step == p1:
            prof.__exit__(None, None, None)
            os.makedirs(a.profile_dir, exist_ok=True)
            prof.export_chrome_trace(f'{a.profile_dir}/trace_rank{rank:03d}.json')
            prof = None
        dt = time.time() - t0
        tps = a.global_batch * seq / dt
        rec = dict(step=step, lr=lr, loss=ce.item(), grad_norm=gnorm.item(), tok_per_s=tps,
                   tok_per_s_gpu=tps / world, mfu=(tps / world) * fpt / peak if peak else 0.0, step_s=dt)
        if step % 50 == 0:
            gc.collect(1)
        if a.timing and distributed:
            td = torch.tensor([t_data[0]], device=dev)
            dist.all_reduce(td, op=dist.ReduceOp.MAX)
            rec['t_data_wait_max'] = td.item()
            t_data[0] = 0.0
            tf = torch.tensor([t1 - t0 - t_comm], device=dev)
            allt = [torch.zeros_like(tf) for _ in range(world)]
            dist.all_gather(allt, tf)
            allt = torch.cat(allt).tolist()
            rec.update(t_fwd_bwd_min=min(allt), t_fwd_bwd_max=max(allt),
                       slowest_rank=int(max(range(world), key=lambda i: allt[i])))
        if a.timing:
            rec.update(t_fwd_bwd=t1 - t0 - t_comm, t_grad_comm=t_comm, t_optim_gather=t2 - t1)
        if cuda:
            rec['max_mem_gb'] = torch.cuda.max_memory_allocated(dev) / 1e9
            rec['mem_fwd_bwd_gb'] = mem_fb
        if val is not None and ((step + 1) % a.val_every == 0 or step == a.steps - 1):
            rec['val_loss'] = evaluate(model, val, a.val_windows, rank, world, dev)
            if len(val_sets) > 1:
                for name, vs in val_sets.items():
                    rec[f'val_{name}'] = evaluate(model, vs, max(a.val_windows // 4, world), rank, world, dev)
        history.append(rec)
        if logf:
            logf.write(json.dumps(rec) + '\n'); logf.flush()
        if rank == 0 and (step % a.log_every == 0 or step == a.steps - 1):
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if a.ckpt_dir and ((step + 1) % a.ckpt_every == 0 or step == a.steps - 1):
            save(a.ckpt_dir, step, model, zero, rank, L, dict(config=cfg.to_dict(), args=vars(a)))
            if rank == 0 and a.keep_last > 0:
                from bench.prune_ckpt import prune
                prune(a.ckpt_dir, a.keep_last, a.milestone_every)
    if distributed:
        dist.destroy_process_group()
    return history, model


if __name__ == '__main__':
    main()

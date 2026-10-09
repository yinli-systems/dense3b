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
    p.add_argument('--save-at-samples', default='', help='comma list: also checkpoint the step where the data stream '
                   'first reaches each of these sample counts (token-matched comparisons across batch sizes)')
    p.add_argument('--batch-offset-step', type=int, default=0, help='step at which --global-batch took effect')
    p.add_argument('--prev-global-batch', type=int, default=0,
                   help='resuming a checkpoint without samples_consumed: the global batch it was trained with')
    p.add_argument('--batch-offset-samples', type=int, default=0, help='samples consumed before that step')
    p.add_argument('--zero-buckets', type=int, default=1,
                   help='>1: layer-bucketed ZeRO, comm overlapped with compute; 0: auto (padding <= 2%%)')
    p.add_argument('--fused-ce', action='store_true', help='LM head + CE with gradients computed in the forward pass')
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
    p.add_argument('--keep-full-steps', default='', help='comma list of steps whose full (resumable) checkpoints are never pruned (fork points)')
    p.add_argument('--freeze-router', action='store_true',
                   help='intervention: router weights leave the optimizer and the balancing bias stops updating')
    p.add_argument('--reset-router', action='store_true',
                   help='intervention: after loading, re-initialise router weights (and clear their optimizer state and bias)')
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
    """Training FLOPs per token: 6 x matmul params (incl. LM head; routed experts counted top_k times) + causal
    attention scores and values."""
    d, hd = cfg.d_model, cfg.head_dim
    attn_w = d * cfg.n_heads * hd * 2 + d * cfg.n_kv_heads * hd * 2 + (d * cfg.n_heads * hd if cfg.attn_gate else 0)
    mlp_w = 3 * d * cfg.d_ff_dense
    moe_w = 3 * d * (cfg.top_k * cfg.d_expert + cfg.d_shared) + d * cfg.n_experts
    Lm = cfg.n_layers - cfg.n_dense_layers
    matmul = cfg.n_layers * attn_w + cfg.n_dense_layers * mlp_w + Lm * moe_w + cfg.vocab_size * d
    attn = cfg.n_layers * 2 * 2 * cfg.n_heads * hd * seq_len / 2          # QK^T and AV, causal half, forward
    return 6 * matmul + 3 * attn


@torch.no_grad()
def update_router_bias(model, distributed):
    """Aux-loss-free balancing on GLOBAL-batch expert loads (one all-reduce for every MoE layer per step).
    Returns the max/mean expert-load ratio (1.0 = perfect balance) for logging."""
    moes = model.moe_layers()
    if not moes:
        return {}
    loads = torch.stack([m.router.load for m in moes])
    if distributed:
        dist.all_reduce(loads)
    for m, ld in zip(moes, loads):
        m.router.update_bias(ld)
    ratio = loads.max(1).values / loads.mean(1).clamp(min=1)
    return dict(moe_load_max_mean=ratio.max().item(), moe_load_max_mean_avg=ratio.mean().item())


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


_SAVER = None


def _cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to('cpu', copy=True)
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    return obj


def save(ckpt_dir, step, model, zero, rank, L, meta, world=1, async_write=True, prune_args=None):
    """Every writer rank stores its owned parameters by name (master + optimizer state), so the checkpoint
    resumes on any number of ranks; rank 0 also stores the BF16 model for evaluation.

    Asynchronous: state is copied to host memory (seconds), then a background thread writes it (the shared file
    system takes minutes for ~25 GB) while training continues. Each rank marks its part done; rank 0 writes COMPLETE
    only after every writer's marker exists, so a checkpoint is never resumed from half-written."""
    global _SAVER
    if _SAVER is not None:
        _SAVER.join()                    # at most one checkpoint in flight
    if hasattr(zero, 'wait_gathers'):
        zero.wait_gathers()
    d = os.path.join(ckpt_dir, f'step_{step:08d}')
    shard = _cpu(zero.named_state()) if rank < L else None
    full = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()} if rank == 0 else None

    def write():
        os.makedirs(d, exist_ok=True)
        if shard is not None:
            tmp = os.path.join(d, f'.rank{rank:03d}.pt.tmp')
            torch.save(shard, tmp)
            os.replace(tmp, os.path.join(d, f'rank{rank:03d}.pt'))
        if full is not None:
            tmp = os.path.join(d, '.model.pt.tmp')
            torch.save(full, tmp)
            os.replace(tmp, os.path.join(d, 'model.pt'))
            json.dump(dict(meta, step=step, writers=L), open(os.path.join(d, 'meta.json'), 'w'))
        open(os.path.join(d, f'.done{rank:03d}'), 'w').close()
        if rank == 0:
            import time as _t
            while not all(os.path.exists(os.path.join(d, f'.done{r:03d}')) for r in range(world)):
                _t.sleep(5)
            open(os.path.join(d, 'COMPLETE'), 'w').close()
            if prune_args:                       # deleting old checkpoints takes minutes on the shared FS
                from bench.prune_ckpt import prune
                prune(ckpt_dir, *prune_args)

    if async_write:
        import threading
        _SAVER = threading.Thread(target=write, daemon=False)
        _SAVER.start()
    else:
        write()



def router_interventions(model, zero, a, cfg):
    """Causal interventions on routing (research forks). Works on the owner-sharded optimizer: router weights live as
    fp32 masters on their owner rank and reach every rank through the parameter all-gather."""
    subs = getattr(zero, 'buckets', None) or [zero]
    g = torch.Generator(device='cpu').manual_seed(a.seed + 7919)
    for z in subs:
        rnames = [n for n in z.proxies if '.router.weight' in n]
        if a.reset_router:
            for n in sorted(rnames):
                q = z.proxies[n]
                q.data.copy_((torch.randn(q.shape, generator=g) * cfg.init_std).to(q.device, q.dtype))
                if z.opt is not None:
                    z.opt.state.pop(q, None)
        if a.freeze_router and z.opt is not None:
            frozen = {id(z.proxies[n]) for n in rnames}
            for grp in z.opt.param_groups:
                grp['params'] = [p for p in grp['params'] if id(p) not in frozen]
        z.P[z.r * z.S:(z.r + 1) * z.S].copy_(z.master)
        z._all_gather_params()
    if a.reset_router:
        for m in model.modules():
            if hasattr(m, 'bias') and hasattr(m, 'update_bias') and torch.is_tensor(m.bias):
                m.bias.zero_()


def wait_saves():
    if _SAVER is not None:
        _SAVER.join()


def load(path, zero, rank, L, model=None):
    files = sorted(glob.glob(os.path.join(path, 'rank*.pt')))
    zero.load_named_state(torch.load(f, map_location='cpu', mmap=True, weights_only=False) for f in files)
    if model is not None:              # persistent buffers (router balancing bias) are not optimizer state
        sd = torch.load(os.path.join(path, 'model.pt'), map_location='cpu', mmap=True, weights_only=False)
        for name, buf in model.named_buffers():
            if name.endswith('router.bias') and name in sd:
                buf.copy_(sd[name].to(buf.device, buf.dtype))
    meta = json.load(open(os.path.join(path, 'meta.json')))
    return meta['step'], meta.get('samples_consumed')


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
    model.head.fused_ce = a.fused_ce
    node, cross = node_groups(a.shard_scope) if distributed else (None, None)
    L = dist.get_world_size(node) if distributed else 1
    if a.zero_buckets == 0:                       # auto: as many buckets as owner balance allows at this world size
        from train.zero import auto_buckets
        a.zero_buckets = auto_buckets(model, L)
    if a.zero_buckets > 1:
        from train.zero import BucketedOwnerZeRO
        zero = BucketedOwnerZeRO(model, a.lr, a.weight_decay, a.optimizer == 'muon', dtype, node, cross, a.sync_every,
                                 n_buckets=a.zero_buckets)
    else:
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
        last, consumed = load(path, zero, rank, L, model)
        start = last + 1
        if consumed is not None and a.batch_offset_step == 0 and a.batch_offset_samples == 0:
            # continue the data stream exactly where the checkpoint left it, whatever the global batch is now
            a.batch_offset_step, a.batch_offset_samples = start, consumed
        elif consumed is None and a.batch_offset_step == 0 and a.batch_offset_samples == 0 and a.prev_global_batch:
            a.batch_offset_step, a.batch_offset_samples = start, start * a.prev_global_batch
    if a.freeze_router or a.reset_router:
        router_interventions(model, zero, a, cfg)
    logf = open(a.metrics, 'a') if (a.metrics and rank == 0) else None
    history, prof = [], None
    p0, p1 = map(int, a.profile_steps.split(','))
    cuda = dev.type == 'cuda'

    def mark():
        if a.timing and cuda:
            torch.cuda.synchronize()
        return time.time()

    if distributed:
        import socket
        hosts = [None] * world
        dist.all_gather_object(hosts, socket.gethostname())
        if rank == 0:
            print(json.dumps(dict(event='hosts', hosts=hosts)), flush=True)
    if rank == 0:
        print(json.dumps(dict(event='start', gpu=gpu, world=world, nodes=world // L, seq=seq, n_micro=n_micro,
                              tokens_per_step=a.global_batch * seq, flops_per_token=fpt, owner_segment=zero.S,
                              zero_padding=getattr(zero, 'padding', None), zero_buckets=a.zero_buckets,
                              ckpt_layers=sorted(model.ckpt_layers), compile=a.compile, fp8=a.fp8, start=start)), flush=True)
    # background prefetch of the next micro-batches (shards live on a network file system; reads must not stall the GPU)
    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(8)

    def fetch(st, mi):
        x, y = data.batch(st, rank, world, mi, a.micro_batch, a.global_batch, a.batch_offset_step, a.batch_offset_samples)
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

    # Python GC: a full collection on one rank stalls every rank at the next collective. Freeze everything built
    # during setup, disable automatic collection, collect young generations on all ranks together every 50 steps.
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
            if mi == n_micro - 1 and hasattr(zero, 'arm'):
                zero.arm()
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
        moe_stats = update_router_bias(model, distributed) if not a.freeze_router else {}
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
        rec.update(moe_stats)
        if step % 50 == 0:
            gc.collect(1)
        if a.timing and distributed:
            td = torch.tensor([t_data[0]], device=dev)
            dist.all_reduce(td, op=dist.ReduceOp.MAX)
            rec['t_data_wait_max'] = td.item()
            t_data[0] = 0.0
            # stragglers: every rank's compute time (the gradient collective waits for the slowest rank)
            tf = torch.tensor([t1 - t0 - t_comm], device=dev)
            allt = [torch.zeros_like(tf) for _ in range(world)]
            dist.all_gather(allt, tf)
            allt = torch.cat(allt).tolist()
            rec.update(t_fwd_bwd_min=min(allt), t_fwd_bwd_max=max(allt), slowest_rank=int(max(range(world), key=lambda i: allt[i])))
        if a.timing:
            rec.update(t_fwd_bwd=t1 - t0 - t_comm, t_grad_comm=t_comm, t_optim_gather=t2 - t1)
        if cuda:
            rec['max_mem_gb'] = torch.cuda.max_memory_allocated(dev) / 1e9
            rec['mem_fwd_bwd_gb'] = mem_fb
            ms = torch.cuda.memory_stats(dev)
            rec['alloc_retries'] = ms.get('num_alloc_retries', 0)
            rec['reserved_gb'] = ms.get('reserved_bytes.all.peak', 0) / 1e9
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
        before = a.batch_offset_samples + (step - a.batch_offset_step) * a.global_batch
        after = before + a.global_batch
        targets = [int(x) for x in a.save_at_samples.split(',') if x]
        hit = any(before < t <= after for t in targets)
        if a.ckpt_dir and ((step + 1) % a.ckpt_every == 0 or hit or (step == a.steps - 1 and a.ckpt_every <= a.steps)):
            consumed = a.batch_offset_samples + (step + 1 - a.batch_offset_step) * a.global_batch
            save(a.ckpt_dir, step, model, zero, rank, L, dict(config=cfg.to_dict(), args=vars(a), samples_consumed=consumed),
                 world=world,
                 prune_args=(a.keep_last, a.milestone_every, False,
                             [int(x) for x in a.keep_full_steps.split(',') if x]) if a.keep_last > 0 else None)
    wait_saves()
    if distributed:
        dist.destroy_process_group()
    return history, model


if __name__ == '__main__':
    main()

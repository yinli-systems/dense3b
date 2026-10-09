"""Day-1 communication benchmark, run under torchrun with the training mesh (replicate=nodes, shard=EP).

Measures what tools/budget.py assumes:
  * all-to-all inside the EP group (NCCL SHM; DRAM-bound), at the message sizes training produces;
  * all-reduce across the replicate group (RoCE), at expert-gradient bucket sizes, with the per-node
    figure that `CLUSTER['nic_gbs'] * CLUSTER['nic_eff']` models;
  * both at once (SHM and RoCE share host DRAM bandwidth).

  torchrun ... -m bench.comm --ep 8 --out results/comm_4090.json
"""
import argparse
import json
import time

import torch
import torch.distributed as dist

from train.parallel import build_mesh, init_distributed


def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def timed(fn, dev, warmup, iters, group):
    for _ in range(warmup):
        fn()
    _sync(dev)
    dist.barrier(group=group)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(dev)
    dt = (time.perf_counter() - t) / iters
    dt_t = torch.tensor([dt], device=dev)
    dist.all_reduce(dt_t, op=dist.ReduceOp.MAX, group=group)     # slowest rank sets the pace
    return dt_t.item()


def bench_a2a(group, dev, mb, dtype, warmup, iters):
    P = dist.get_world_size(group)
    n = int(mb * 2 ** 20) // torch.tensor([], dtype=dtype).element_size()
    n -= n % P
    x, y = torch.randn(n, device=dev).to(dtype), torch.empty(n, device=dev, dtype=dtype)
    dt = timed(lambda: dist.all_to_all_single(y, x, group=group), dev, warmup, iters, group)
    remote = x.numel() * x.element_size() * (P - 1) / P
    return dict(mb_per_rank=mb, seconds=dt, remote_gbs_per_gpu=remote / dt / 1e9)


def bench_allreduce(group, dev, mb, dtype, warmup, iters, gpus_per_node):
    R = dist.get_world_size(group)
    n = int(mb * 2 ** 20) // torch.tensor([], dtype=dtype).element_size()
    x = torch.randn(n, device=dev).to(dtype)
    dt = timed(lambda: dist.all_reduce(x, group=group), dev, warmup, iters, group)
    per_gpu = 2 * (R - 1) / R * x.numel() * x.element_size()
    return dict(mb_per_rank=mb, seconds=dt, busbw_gbs_per_gpu=per_gpu / dt / 1e9,
                node_gbs=per_gpu * gpus_per_node / dt / 1e9)


def bench_concurrent(shard, rep, dev, a2a_mb, ar_mb, dtype, n_a2a):
    """One big all-reduce over RoCE while `n_a2a` all-to-alls run over SHM (the last backward of a step).
    Every rank issues the same number of collectives; slowdown = concurrent time / (each alone)."""
    P = dist.get_world_size(shard)
    esz = torch.tensor([], dtype=dtype).element_size()
    n = int(a2a_mb * 2 ** 20) // esz
    n -= n % P
    x, y = torch.randn(n, device=dev).to(dtype), torch.empty(n, device=dev, dtype=dtype)
    g = torch.randn(int(ar_mb * 2 ** 20) // esz, device=dev).to(dtype)

    def both():
        work = dist.all_reduce(g, group=rep, async_op=True)
        for _ in range(n_a2a):
            dist.all_to_all_single(y, x, group=shard)
        work.wait()

    t_both = timed(both, dev, 1, 3, None)
    t_ar = timed(lambda: dist.all_reduce(g, group=rep), dev, 1, 3, None)
    t_a2a = timed(lambda: [dist.all_to_all_single(y, x, group=shard) for _ in range(n_a2a)], dev, 1, 3, None)
    return dict(n_a2a=n_a2a, a2a_mb=a2a_mb, ar_mb=ar_mb, seconds_concurrent=t_both, seconds_allreduce=t_ar,
                seconds_a2a=t_a2a, overlap_efficiency=max(t_ar, t_a2a) / t_both)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', type=int, default=8)
    ap.add_argument('--a2a-mb', default='4,16,32,64,128', help='bytes each rank sends per all-to-all')
    ap.add_argument('--ar-mb', default='64,256,1024')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    assert init_distributed(), 'run under torchrun'
    dev = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')
    mesh = build_mesh(a.ep, dev.type)
    shard, rep = mesh['shard'].get_group(), mesh['replicate'].get_group()
    dtype = torch.bfloat16
    out = dict(world=dist.get_world_size(), ep=a.ep, replicas=mesh['replicate'].size(), device=str(dev),
               gpu=torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu',
               torch=torch.__version__, nccl='.'.join(map(str, torch.cuda.nccl.version())) if dev.type == 'cuda' else None)
    out['a2a'] = [bench_a2a(shard, dev, float(mb), dtype, a.warmup, a.iters) for mb in a.a2a_mb.split(',')]
    if mesh['replicate'].size() > 1:
        out['allreduce'] = [bench_allreduce(rep, dev, float(mb), dtype, a.warmup, a.iters, a.ep)
                            for mb in a.ar_mb.split(',')]
        a2a_mb = float(a.a2a_mb.split(',')[-1])
        ar_mb = float(a.ar_mb.split(',')[-1])
        t_a2a = next(r['seconds'] for r in out['a2a'] if r['mb_per_rank'] == a2a_mb)
        t_ar = next(r['seconds'] for r in out['allreduce'] if r['mb_per_rank'] == ar_mb)
        ratio = torch.tensor([t_ar / t_a2a], device=dev)
        dist.all_reduce(ratio, op=dist.ReduceOp.MAX)   # every rank must issue the same number of all-to-alls
        n = max(1, round(ratio.item()))
        out['concurrent'] = bench_concurrent(shard, rep, dev, a2a_mb, ar_mb, dtype, n)
    if dist.get_rank() == 0:
        print(json.dumps(out, indent=1))
        if a.out:
            with open(a.out, 'w') as f:
                json.dump(out, f, indent=1)
    dist.destroy_process_group()
    return out


if __name__ == '__main__':
    main()

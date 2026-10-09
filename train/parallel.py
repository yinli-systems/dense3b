"""Process groups and sharding for one layout: mesh (replicate=nodes, shard=GPUs in a node).

* Dense params (attention, norms, shared expert, router, embeddings, head): FSDP2 over `shard` and
  replicated over `replicate` (HSDP), BF16 compute / FP32 reduce; the router keeps FP32 params.
* Routed experts: expert-parallel over `shard` (EP group == FSDP shard group == one node or one socket),
  replicated over `replicate`; their grads are all-reduced across nodes once per optimizer step,
  launched from post-accumulate hooks so the transfer overlaps the rest of the last backward.
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor


def init_distributed():
    if 'RANK' not in os.environ:
        return False
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    return True


def build_mesh(ep_size, device_type):
    world = dist.get_world_size()
    assert world % ep_size == 0, 'EP group must tile the world'
    return init_device_mesh(device_type, (world // ep_size, ep_size), mesh_dim_names=('replicate', 'shard'))


def expert_params(model):
    return [p for m in model.moe_layers() for p in (m.w13, m.w2)]


def apply_parallel(model, mesh, bf16=True):
    shard_group = mesh['shard'].get_group()
    for m in model.moe_layers():
        m.ep_group = shard_group if mesh['shard'].size() > 1 else None
    experts = set(expert_params(model))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if bf16 \
        else MixedPrecisionPolicy()
    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    for blk in model.layers:
        if blk.is_moe:
            fully_shard(blk.ffn.router, mesh=mesh, mp_policy=fp32)
        fully_shard(blk, mesh=mesh, mp_policy=mp, ignored_params=experts & set(blk.parameters()))
    fully_shard(model.embed, mesh=mesh, mp_policy=mp)
    fully_shard(model.head, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp, ignored_params=experts)
    return model


class ExpertGradSync:
    """Turns per-rank expert grads into the global-mean gradient. Each expert grad already sums the
    contributions of every rank in its EP group (via the all-to-all backward), so: divide by EP size,
    then average over the replicate group."""

    def __init__(self, params, mesh, dtype=torch.bfloat16):
        self.ep, self.dtype = mesh['shard'].size(), dtype      # BF16 halves the dominant inter-node bytes
        self.rep = mesh['replicate'].get_group() if mesh['replicate'].size() > 1 else None
        self.active, self.works = False, []
        for p in params:
            p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p):
        if not self.active:
            return
        p.grad.div_(self.ep)
        if self.rep is not None:
            buf = p.grad.to(self.dtype)
            self.works.append((dist.all_reduce(buf, op=dist.ReduceOp.AVG, group=self.rep, async_op=True), p, buf))

    def wait(self):
        for work, p, buf in self.works:
            work.wait()
            p.grad.copy_(buf)
        self.works = []


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


@torch.no_grad()
def clip_grad_norm(params, max_norm, shard_group=None):
    """Global L2 norm: dense DTensor shards and expert shards are both disjoint across the shard group and
    identical across replicas, so one all-reduce over the shard group gives the exact total."""
    grads = [_local(p.grad) for p in params if p.grad is not None]
    sq = torch.stack([g.float().pow(2).sum() for g in grads]).sum()
    if shard_group is not None:
        dist.all_reduce(sq, group=shard_group)
    norm = sq.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale.to(g.dtype))
    return norm


@torch.no_grad()
def update_router_bias(model, group=None):
    """Global-batch load counts (all ranks, all micro-batches) -> aux-loss-free bias update. Returns loads."""
    routers = [m.router for m in model.moe_layers()]
    if not routers:
        return None
    load = torch.stack([r.load for r in routers])
    if group is not None or dist.is_initialized():
        dist.all_reduce(load, group=group)
    for r, l in zip(routers, load):
        r.update_bias(l)
    return load

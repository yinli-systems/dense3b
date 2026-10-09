"""Owner-sharded data parallelism for dense models on PCIe-only nodes (ZeRO-1 with whole-matrix ownership).

Measured on these 8x4090 nodes: ~9 GB/s collective bus bandwidth per GPU (no P2P; NCCL goes through host
shared memory) and poor compute/communication overlap. FSDP re-gathers parameters and reduce-scatters
gradients on every micro-batch (~17 GB/GPU per micro-batch for 3B), which would leave the GPUs idle most of
the time. Here communication happens once per optimizer step instead:

* every rank holds the full model in the compute dtype and accumulates gradients locally (no traffic);
* each local rank OWNS whole parameters (balanced by size), so Muon's Newton-Schulz sees full matrices and
  the update is bit-for-bit the single-process algorithm; the owner keeps the fp32 master copy and the
  optimizer state for its parameters only;
* step: intra-node reduce-scatter of the flat gradient buffer (each rank receives the mean gradient of what it
  owns), cross-node all-reduce of that shard, global-norm clipping, owner update, intra-node all-gather of
  the updated parameters. Bytes per GPU per step: ~2 x model size, independent of gradient accumulation.

Parameters and gradients are views into two flat buffers laid out [segment of local rank 0 | ... | L-1], each
segment padded to the same length, so both collectives run in place.
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn

from train.optim import MuonAdamW, param_groups

ALIGN = 128


def assign_owners(sizes, n):
    """Greedy largest-first bin packing of parameter sizes into n owners -> owner index per parameter."""
    load, owner = [0] * n, [0] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        r = min(range(n), key=lambda r: load[r])
        owner[i] = r; load[r] += sizes[i]
    return owner


class OwnerZeRO:
    def __init__(self, model, lr, weight_decay=0.1, use_muon=True, dtype=torch.bfloat16, node_group=None,
                 cross_group=None, sync_every=0):
        self.model, self.node_group, self.cross_group = model, node_group, cross_group
        self.L = dist.get_world_size(node_group) if dist.is_initialized() else 1
        self.r = dist.get_rank(node_group) if dist.is_initialized() else 0
        self.sync_every = sync_every
        named = list(dict(model.named_parameters()).items())           # tied weights appear once
        self.model_named = named
        dev = named[0][1].device
        sizes = [p.numel() for _, p in named]
        owner = assign_owners(sizes, self.L)
        seg = [[i for i in range(len(named)) if owner[i] == r] for r in range(self.L)]
        S = max(sum(sizes[i] for i in s) for s in seg)
        self.S = S = -(-S // ALIGN) * ALIGN
        self.P = torch.zeros(self.L * S, dtype=dtype, device=dev)       # flat params (compute dtype)
        self.G = torch.zeros(self.L * S, dtype=dtype, device=dev)       # flat grads, accumulated locally
        self.slots = {}                                                 # name -> (offset, numel, shape)
        for r in range(self.L):
            off = r * S
            for i in seg[r]:
                name, p = named[i]
                n = p.numel()
                self.P[off:off + n].copy_(p.detach().reshape(-1))
                p.data = self.P[off:off + n].view(p.shape)
                p.grad = self.G[off:off + n].view(p.shape)
                self.slots[name] = (off, n, p.shape)
                off += n
        mine = self.P[self.r * S:(self.r + 1) * S]
        self.master = mine.float().clone()                             # fp32 master of owned params
        # gradient shard the optimizer reads: the rank's own segment of G (reduced in place) when reducing once
        # per step; a separate fp32 accumulator when partial sums are reduced every `sync_every` micro-batches
        self.gmine = self.G[self.r * S:(self.r + 1) * S]
        self.gshard = torch.zeros(S, dtype=torch.float32, device=dev) if sync_every else self.gmine
        # fp32 proxies of the owned parameters, named like the model so param_groups() classifies them
        self.proxies = {}
        for name, p in named:
            off, n, shape = self.slots[name]
            if self.r * S <= off < (self.r + 1) * S:
                lo = off - self.r * S
                q = nn.Parameter(self.master[lo:lo + n].view(shape))
                if self.gshard.dtype != q.dtype and hasattr(q, 'grad_dtype'):
                    q.grad_dtype = self.gshard.dtype            # fp32 master, BF16 reduced gradient
                q.grad = self.gshard[lo:lo + n].view(shape)
                for attr in ('_muon_rows', '_muon_split'):
                    if hasattr(p, attr):
                        setattr(q, attr, getattr(p, attr))
                self.proxies[name] = q
        groups = param_groups(_Named(self.proxies), use_muon, weight_decay)
        self.opt = MuonAdamW(groups, lr=lr) if groups else None

    # ------------------------------------------------------------------ collectives (gloo fallbacks for CPU tests)
    def _nccl(self):
        return self.G.is_cuda

    def _reduce_scatter_avg(self):
        """G (L*S) -> mean over the node group, written in place into this rank's segment."""
        if self.L == 1:
            return
        if self._nccl():
            dist.reduce_scatter_tensor(self.gmine, self.G, op=dist.ReduceOp.AVG, group=self.node_group)
        else:
            dist.all_reduce(self.G, group=self.node_group)
            self.G.div_(self.L)

    def _cross_avg(self, t):
        if self.cross_group is not None and dist.get_world_size(self.cross_group) > 1:
            if self._nccl():
                dist.all_reduce(t, op=dist.ReduceOp.AVG, group=self.cross_group)
            else:
                dist.all_reduce(t, group=self.cross_group)
                t.div_(dist.get_world_size(self.cross_group))

    def _all_gather_params(self):
        if self.L == 1:
            return
        mine = self.P[self.r * self.S:(self.r + 1) * self.S]
        if self._nccl():
            dist.all_gather_into_tensor(self.P, mine, group=self.node_group)
        else:
            parts = list(self.P.view(self.L, self.S).unbind(0))
            got = [torch.empty_like(mine) for _ in range(self.L)]
            dist.all_gather(got, mine.clone(), group=self.node_group)
            for a, b in zip(parts, got):
                a.copy_(b)

    # ------------------------------------------------------------------ gradient path
    def _reduce(self):
        self._reduce_scatter_avg()
        self._cross_avg(self.gmine)
        if self.sync_every:
            self.gshard.add_(self.gmine.float())
            self.G.zero_()

    def after_backward(self, micro, n_micro):
        """Call after each micro-batch backward. Reduces when `sync_every` micro-batches have accumulated
        (fp32 accumulation of partial sums) and always after the last one."""
        last = micro == n_micro - 1
        if last or (self.sync_every and (micro + 1) % self.sync_every == 0):
            self._reduce()

    # ------------------------------------------------------------------ optimizer path
    @torch.no_grad()
    def step(self, lr, clip):
        sq = self.gshard.float().pow(2).sum()
        if self.L > 1:
            dist.all_reduce(sq, group=self.node_group)
        gnorm = sq.sqrt()
        if clip:
            scale = (clip / (gnorm + 1e-6)).clamp(max=1.0)
            self.gshard.mul_(scale.to(self.gshard.dtype))
        if self.opt is not None:
            for g in self.opt.param_groups:
                g['lr'] = lr
            self.opt.step()
        self.P[self.r * self.S:(self.r + 1) * self.S].copy_(self.master)
        self._all_gather_params()
        self.G.zero_()
        if self.sync_every:
            self.gshard.zero_()
        return gnorm

    # ------------------------------------------------------------------ checkpoint (layout-independent)
    def named_state(self):
        """{param name: {'master': fp32 tensor, 'state': optimizer state}} for the parameters this rank owns.
        Keyed by name, so a checkpoint written with W ranks resumes on any other number of ranks."""
        out = {}
        for name, q in self.proxies.items():
            st = self.opt.state.get(q, {}) if self.opt is not None else {}
            out[name] = dict(master=q.detach().clone(),
                             state={k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in st.items()})
        return out

    def load_named_state(self, shards):
        """shards: iterable of named_state() dicts (e.g. memory-mapped files of every writer rank)."""
        found = set()
        for sd in shards:
            for name, ent in sd.items():
                q = self.proxies.get(name)
                if q is None:
                    continue
                q.data.copy_(ent['master'].to(q.device))
                if self.opt is not None and ent['state']:
                    self.opt.state[q] = {k: (v.to(q.device) if torch.is_tensor(v) else v)
                                         for k, v in ent['state'].items()}
                found.add(name)
        missing = set(self.proxies) - found
        assert not missing, f'checkpoint lacks owned parameters: {sorted(missing)[:5]}'
        self.P[self.r * self.S:(self.r + 1) * self.S].copy_(self.master)
        self._all_gather_params()


class _Named:
    """Minimal stand-in exposing named_parameters() over a dict, for param_groups()."""

    def __init__(self, d):
        self.d = d

    def named_parameters(self):
        return iter(self.d.items())


def node_groups(scope='node'):
    """(owner-sharding group, replica group). scope='node': shard inside each node and all-reduce the shards
    across nodes (needs full nodes: owned state is 1/L of the model). scope='world': shard over every rank of
    the job -- for jobs scattered one or two GPUs per node; the step's collectives then cross the network."""
    world, rank = dist.get_world_size(), dist.get_rank()
    if scope == 'world':
        return dist.group.WORLD, None
    L = int(os.environ.get('LOCAL_WORLD_SIZE', world))
    nodes = world // L
    if nodes == 1:
        return dist.group.WORLD, None
    node, cross = None, None
    for n in range(nodes):
        g = dist.new_group(list(range(n * L, (n + 1) * L)))
        if rank // L == n:
            node = g
    for lr in range(L):
        g = dist.new_group([n * L + lr for n in range(nodes)])
        if rank % L == lr:
            cross = g
    return node, cross


def bucket_key(name, n_layers, n_buckets):
    if name.startswith('layers.'):
        return min(n_buckets - 1, int(name.split('.')[1]) * n_buckets // n_layers)
    return 0 if name.startswith('embed') else n_buckets - 1


def auto_buckets(model, world, max_padding=0.02, candidates=(8, 6, 5, 4, 3, 2)):
    """Largest bucket count whose whole-matrix owner assignment wastes <= max_padding of the communicated bytes
    (big expert tensors cannot be split, so more ranks allow fewer buckets)."""
    named = list(dict(model.named_parameters()).items())
    for K in candidates:
        shard = used = 0
        for b in range(K):
            sizes = [p.numel() for n, p in named if bucket_key(n, model.cfg.n_layers, K) == b]
            if not sizes:
                continue
            owner = assign_owners(sizes, world)
            load = [0] * world
            for s_, o in zip(sizes, owner):
                load[o] += s_
            shard += max(load) * world; used += sum(sizes)
        if 1 - used / shard <= max_padding:
            return K
    return 1


class BucketedOwnerZeRO:
    """OwnerZeRO split into K layer-ordered buckets so communication overlaps compute:

    * during the LAST micro-batch's backward, a bucket's reduce-scatter starts as soon as every gradient in it is
      final (later layers finish first), on NCCL's stream while backward continues;
    * the optimizer runs bucket by bucket in forward order and each bucket's all-gather starts right after its update,
      overlapping the next bucket's Newton-Schulz;
    * the next step's forward waits for a bucket's all-gather only right before the first layer that needs it.
    Numerics are identical to OwnerZeRO with whole-matrix owners (same collectives, same per-parameter math).
    Each bucket is an OwnerZeRO over its own parameters, so checkpoints stay name-keyed and layout-independent."""

    def __init__(self, model, lr, weight_decay=0.1, use_muon=True, dtype=torch.bfloat16, node_group=None,
                 cross_group=None, sync_every=0, n_buckets=4):
        assert sync_every == 0, 'bucketed overlap assumes one reduction per step'
        self.model, self.node_group, self.cross_group = model, node_group, cross_group
        named = list(dict(model.named_parameters()).items())
        L = model.cfg.n_layers
        def bucket_of(name):                                           # embed (tied head) first, final norm last
            return bucket_key(name, L, n_buckets)
        groups = [[(n, p) for n, p in named if bucket_of(n) == b] for b in range(n_buckets)]
        self.buckets = [OwnerZeRO(_Sub(model, g), lr, weight_decay, use_muon, dtype, node_group, cross_group, 0)
                        for g in groups if g]
        self.K = len(self.buckets)
        self.S = sum(b.S for b in self.buckets)
        self.L = self.buckets[0].L
        self.G = None
        # gradient readiness: expected AccumulateGrad events per parameter in one backward (tied weights: 2)
        uses = {id(p): 0 for _, p in named}
        for m in model.modules():
            for p in m.parameters(recurse=False):
                uses[id(p)] = uses.get(id(p), 0) + 1
        self.pbucket, self.expected = {}, []
        for bi, b in enumerate(self.buckets):
            exp = 0
            for name, p in b.model_named:
                self.pbucket[id(p)] = bi
                exp += uses[id(p)]
                if p.dim() == 3:          # local experts accumulate in place and signal readiness themselves
                    p._grad_ready_hook = self._ready
                else:
                    p.register_post_accumulate_grad_hook(self._ready)
            self.expected.append(exp)
        self.armed, self.count, self.rs_work, self.ag_work = False, [0] * self.K, [None] * self.K, [None] * self.K
        # the next forward waits for a bucket's all-gather right before the first part that uses it
        need = {}
        for bi, b in enumerate(self.buckets):
            for name, _ in b.model_named:
                key = int(name.split('.')[1]) if name.startswith('layers.') else ('embed' if name.startswith('embed') else 'head')
                need.setdefault(key, set()).add(bi)
        self.need = need
        model.pre_block_hook = self._wait_for
        shard = [b.S * b.L for b in self.buckets]
        used = [sum(p.numel() for _, p in b.model_named) for b in self.buckets]
        self.padding = 1 - sum(used) / sum(shard)

    def _wait_for(self, key):
        for bi in self.need.get(key, ()):
            w = self.ag_work[bi]
            if w is not None:
                w.wait(); self.ag_work[bi] = None

    # ---------------------------------------------------------------- readiness / async reduce-scatter
    def _ready(self, p):
        if not self.armed:
            return
        bi = self.pbucket[id(p)]
        self.count[bi] += 1
        if self.count[bi] == self.expected[bi] and self.rs_work[bi] is None:
            self._launch_rs(bi)

    def _launch_rs(self, bi):
        b = self.buckets[bi]
        if b.L == 1:
            self.rs_work[bi] = _Done()
            return
        self.rs_work[bi] = dist.reduce_scatter_tensor(b.gmine, b.G, op=dist.ReduceOp.AVG, group=b.node_group,
                                                      async_op=True)

    def after_backward(self, micro, n_micro):
        if micro == n_micro - 1:
            for bi in range(self.K):              # anything a hook did not trigger (deterministic order on all ranks)
                if self.rs_work[bi] is None:
                    self._launch_rs(bi)
            for bi in range(self.K):
                self.rs_work[bi].wait()
                if self.buckets[bi].cross_group is not None:
                    self.buckets[bi]._cross_avg(self.buckets[bi].gmine)
            self.armed = False

    def arm(self):
        """Call right before the last micro-batch's backward."""
        self.armed, self.count, self.rs_work = True, [0] * self.K, [None] * self.K

    # ---------------------------------------------------------------- optimizer (pipelined with all-gathers)
    @torch.no_grad()
    def step(self, lr, clip):
        sq = torch.stack([b.gshard.float().pow(2).sum() for b in self.buckets]).sum()
        if self.L > 1:
            dist.all_reduce(sq, group=self.node_group)
        gnorm = sq.sqrt()
        scale = (clip / (gnorm + 1e-6)).clamp(max=1.0) if clip else None
        for bi, b in enumerate(self.buckets):
            if scale is not None:
                b.gshard.mul_(scale.to(b.gshard.dtype))
            if b.opt is not None:
                for g in b.opt.param_groups:
                    g['lr'] = lr
                b.opt.step()
            b.P[b.r * b.S:(b.r + 1) * b.S].copy_(b.master)
            mine = b.P[b.r * b.S:(b.r + 1) * b.S]
            self.ag_work[bi] = (dist.all_gather_into_tensor(b.P, mine, group=b.node_group, async_op=True)
                                if b.L > 1 else None)
            b.G.zero_()
        return gnorm

    def wait_gathers(self):
        for bi in range(self.K):
            if self.ag_work[bi] is not None:
                self.ag_work[bi].wait(); self.ag_work[bi] = None

    # ---------------------------------------------------------------- checkpoint
    def named_state(self):
        out = {}
        for b in self.buckets:
            out.update(b.named_state())
        return out

    def load_named_state(self, shards):
        shards = list(shards)
        for b in self.buckets:
            b.load_named_state(shards)
        self.wait_gathers()


class _Sub:
    """A parameter subset that OwnerZeRO can treat as a model."""

    def __init__(self, model, named):
        self._named = named
        self.cfg = model.cfg

    def named_parameters(self):
        return iter(self._named)


class _Done:
    def wait(self):
        pass

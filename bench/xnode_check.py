"""Cross-node NCCL check for scattered jobs: reduce-scatter / all-gather bandwidth at training sizes."""
import os, time, json
import torch, torch.distributed as dist

dist.init_process_group('nccl')
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
out = {}
for mb in [int(x) for x in os.environ.get('XNODE_MB', '256,1024').split(',')]:
    n = mb * (1 << 20) // 2
    full = torch.ones(n * w, dtype=torch.bfloat16, device='cuda'); part = torch.empty(n, dtype=torch.bfloat16, device='cuda')
    for op in ('reduce_scatter', 'all_gather'):
        f = (lambda: dist.reduce_scatter_tensor(part, full, op=dist.ReduceOp.AVG)) if op == 'reduce_scatter' \
            else (lambda: dist.all_gather_into_tensor(full, part))
        f(); torch.cuda.synchronize(); t = time.time()
        for _ in range(3):
            f()
        torch.cuda.synchronize(); dt = (time.time() - t) / 3
        out[f'{op}_{mb}MB_per_rank'] = dict(seconds=round(dt, 4), busbw_gbs=round(n * 2 * (w - 1) / dt / 1e9, 2))
if r == 0:
    print(json.dumps(dict(world=w, hosts=os.environ.get('SLURM_JOB_NODELIST'), **out)), flush=True)
dist.destroy_process_group()

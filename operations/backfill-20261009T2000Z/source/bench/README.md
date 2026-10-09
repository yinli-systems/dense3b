# Day-1 cluster kit

These jobs replace the assumptions in `tools/budget.py` / `docs/MODEL_DESIGN.md` §11 with measurements.
Every piece is exercised at toy size on CPU by `tests/moe/test_bench.py`.

## Before you submit

Edit `bench/slurm/env.sh`:
* `PYTHON` → the interpreter with torch 2.13.0+cu130.
* `LOCAL_SCRATCH` → node-local NVMe.
* RoCE knobs → `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, if your site needs them.

Then set the partition in the `#SBATCH` lines. The scripts use `gpu_4090`.

## Run

```bash
mkdir -p results
# 1. one node, ~1 h: env diagnostics, grouped-GEMM autotune, all-to-all, memory/step sweep, profile
sbatch bench/slurm/day1_node.sbatch
# 2. after (1): all 6 nodes, ~40 min: inter-node all-reduce, concurrency, real-batch throughput
MOE_GMM_CONFIG=$PWD/results/<date>-<jobid>/gmm.config.json sbatch --export=ALL bench/slurm/day1_cluster.sbatch
```

For a 5090 partition: change `--partition`, and pass `GPU=5090` and `MICRO_BATCH=2` to the cluster job.

## What each job writes to `results/<date>-<jobid>/`

| File | Answers |
|---|---|
| `diag.<host>.txt` | GPU topology, NUMA layout, bond status (both legs up? hash policy?), RoCE device |
| `nccl.*.log` | which NCCL transport each channel actually uses (SHM intra-node, IB/RoCE inter-node) |
| `gmm.json`, `gmm.config.json`, `pytest_gpu.txt` | correctness of the Triton grouped GEMM on this GPU; best tiles vs the per-expert cuBLAS loop and `torch._grouped_mm` |
| `comm_node_ep8.json`, `comm_node_ep4.json` | SHM all-to-all GB/s per GPU at 4–128 MB per rank |
| `step_mb1_recompute.jsonl`, `step_mb1.jsonl`, `step_mb2_recompute.jsonl` | `max_mem_gb`; time split into fwd+bwd / grad-sync wait / optimizer; `failures.txt` lists configs that OOM'd |
| `trace/`, `trace_cluster/` | torch.profiler chrome traces (one per rank) |
| `comm_cluster.json`, `budget_measured.md` | inter-node all-reduce GB/s per node, its overlap with all-to-all, and the budget recomputed with these numbers |
| `throughput.jsonl` | 48-GPU tokens/s at the real 3.93 M-token batch, router balance, a2a rows per (token, expert) |

## Afterwards

Send back the whole `results/` folder, or commit it on a branch without the traces if they are large.
Then:

```bash
python tools/budget.py --measured results/<run>/comm_cluster.json --gpu 4090 --compute-eff <from profile>
```

That recomputes tokens in 15 days. The doc's decision rules (§6.4, §10) say what to change if a number
misses.

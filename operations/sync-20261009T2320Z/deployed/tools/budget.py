"""FLOPs / communication / memory / wall-clock budget for the design doc (docs/MODEL_DESIGN.md).

  python tools/budget.py            # markdown tables used in the doc
  python tools/budget.py --json     # machine-readable

No torch needed. Every hardware number is an input below with its source; every efficiency number is an
explicit assumption that the Day-1 measurements replace (see the doc).
"""
import argparse
import json
import math
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import PRESETS, param_counts  # noqa: E402

GPUS = {
    # dense tensor TFLOPS with FP32 accumulate (GeForce runs FP32-accumulate at half the FP16-accumulate rate)
    '4090': dict(bf16=165.2, fp8=330.3, mem_gb=24, shm_gbs=(15, 18)),    # Ada whitepaper
    '5090': dict(bf16=209.5, fp8=419.0, mem_gb=32, shm_gbs=(22, 27)),    # Blackwell GB202 whitepaper
}
CLUSTER = dict(nodes=6, gpus_per_node=8, nic_gbs=2 * 25 / 8, nic_eff=0.70, days=15, availability=0.90)
SCENARIOS = {  # (matmul-heavy compute efficiency vs bf16 peak, fraction of all-to-all hidden by overlap)
    'conservative': (0.30, 0.0),
    'expected': (0.40, 0.3),
}
TRAIN = dict(seq_len=4096, global_seqs=960, tokens_target=140e9)


def expected_dest_ranks(E, k, P, M=0):
    """Expected number of distinct EP ranks per token for uniform top-k routing (E experts on P ranks).
    With group-limited routing (groups = ranks) we use the bound M: groups are picked by their top-(k/M)
    scores, so a picked group almost always receives an expert (measured 3.98 of M=4, tests/moe/test_budget.py)."""
    if M:
        return float(M)
    El = E // P
    return P * (1 - comb(E - El, k) / comb(E, k))


def flops_per_token(c, T):
    p = param_counts(c)
    attn = 6 * c.n_layers * c.n_heads * c.head_dim * T     # causal QK^T + AV, fwd + bwd (avg context T/2)
    return 6 * p['active_matmul'] + attn, 6 * p['active_matmul'], attn


def a2a_bytes_per_token(c, P, dedup=True, fp8_dispatch=False, M=None):
    """Bytes one GPU sends per token over all MoE layers, per training step (4 all-to-alls per layer:
    fwd dispatch, fwd combine, bwd of combine, bwd of dispatch); only the (P-1)/P remote share moves."""
    M = c.route_topk_groups if (M is None and c.route_groups) else (M or 0)
    rows = expected_dest_ranks(c.n_experts, c.top_k, P, M) if dedup else c.top_k
    d = c.d_model
    meta = 2 * c.top_k * 4                                  # fp32 gate weight + local id per slot
    disp = (d * 1 + 4) if fp8_dispatch else d * 2
    per_row = disp + 3 * d * 2 + 2 * meta
    return rows, rows * per_row * c.n_moe_layers * (P - 1) / P


def memory_gb(c, gpu, P, micro_seqs, T, recompute=True):
    p = param_counts(c)
    expert = p['routed_experts'] / P
    dense = (p['total'] - p['routed_experts']) / P
    adam_dense = (p['embed'] + p['head'] + p['router']) / P
    states = expert * (4 + 4 + 4) + dense * (4 + 4) + (dense - adam_dense) * 4 + adam_dense * 8
    transient = 2 * 2 * p['embed']                              # largest unsharded bf16 unit + prefetch
    d, k, f, ds = c.d_model, c.top_k, c.d_expert, c.d_shared
    rows = expected_dest_ranks(c.n_experts, k, P, c.route_topk_groups if c.route_groups else 0)
    attn = 6 * d + 3 * c.n_heads * c.head_dim + 2 * c.n_kv_heads * c.head_dim
    moe = (2 * rows + 1) * d + 3 * ds + 2 * c.n_experts + (0 if recompute else 2 * k * d + 3 * k * f)
    dense_ffn = 3 * c.d_ff_dense
    per_tok = 2 * (c.n_layers * attn + c.n_moe_layers * moe + c.n_dense_layers * dense_ffn)
    act = per_tok * micro_seqs * T
    logits = 2048 * c.vocab_size * 4 * 2                        # one chunk of fp32 logits + grad
    workspace = 2.0e9                                           # NCCL/SHM buffers, allocator slack, kernels
    total = states + transient + act + logits + workspace
    return dict(states=states / 1e9, activations=act / 1e9, other=(transient + logits + workspace) / 1e9,
                total=total / 1e9, fits=total / 1e9 <= GPUS[gpu]['mem_gb'] * 0.95)


def grad_sync_s(c, gpu_name, sync_bytes=2):
    """Inter-node all-reduce per optimizer step: experts in bf16 (default) + dense FSDP shards in fp32,
    ring over `nodes` replicas; every node pushes 2(R-1)/R of one full model replica."""
    p = param_counts(c)
    R = CLUSTER['nodes']
    by = p['routed_experts'] * sync_bytes + (p['total'] - p['routed_experts']) * 4
    per_node = 2 * (R - 1) / R * by
    return per_node / 1e9, per_node / 1e9 / (CLUSTER['nic_gbs'] * CLUSTER['nic_eff'])


def step_time(c, gpu, scenario, micro_seqs, dedup=True, fp8_dispatch=False, recompute=True):
    eff, hidden = SCENARIOS[scenario]
    T = TRAIN['seq_len']
    world = CLUSTER['nodes'] * CLUSTER['gpus_per_node']
    P = CLUSTER['gpus_per_node']
    tok_gpu_step = TRAIN['global_seqs'] * T / world
    fl, _, _ = flops_per_token(c, T)
    if recompute:                                           # expert fwd GEMMs run twice
        fl += 2 * 3 * c.d_model * c.d_expert * c.top_k * c.n_moe_layers
    g = GPUS[gpu]
    compute = tok_gpu_step * fl / (g['bf16'] * 1e12 * eff)
    rows, by = a2a_bytes_per_token(c, P, dedup, fp8_dispatch)
    shm = g['shm_gbs'][0] * 1e9                             # low end of the measured range
    a2a = tok_gpu_step * by / shm * (1 - hidden)
    _, sync = grad_sync_s(c, gpu)
    last_micro_bwd = compute * micro_seqs * T / tok_gpu_step * 2 / 3
    sync_exposed = max(0.0, sync - last_micro_bwd)
    total = compute + a2a + sync_exposed
    tps = TRAIN['global_seqs'] * T / total
    days_tokens = tps * CLUSTER['days'] * 86400 * CLUSTER['availability']
    return dict(compute_s=compute, a2a_s=a2a, sync_s=sync, sync_exposed_s=sync_exposed, step_s=total,
                tok_per_s=tps, tok_per_s_gpu=tps / world, tokens_15d=days_tokens, a2a_rows=rows,
                a2a_mb_per_token=by / 1e6, mfu=tok_gpu_step * flops_per_token(c, T)[0] / total / (g['bf16'] * 1e12))


def deepseek_hparams(c, D):
    """DeepSeek LLM (arXiv 2401.02954) fits: eta_opt = 0.3118 C^-0.125, B_opt = 0.2920 C^0.3271 (tokens),
    C = M * D with M = non-embedding FLOPs/token (6N + attention), here using active parameters."""
    fl, _, _ = flops_per_token(c, TRAIN['seq_len'])
    C = (fl - 6 * c.vocab_size * c.d_model) * D
    return dict(C=C, lr=0.3118 * C ** -0.125, batch_tokens=0.2920 * C ** 0.3271)


def report():
    out = {}
    for name in ('moe7b', 'olmoe'):
        c = PRESETS[name]
        p = param_counts(c)
        fl, mat, att = flops_per_token(c, TRAIN['seq_len'])
        r = dict(params=p, flops_per_token=fl, flops_matmul=mat, flops_attn=att, hparams=deepseek_hparams(c, TRAIN['tokens_target']))
        for dedup in (True, False):
            rows, by = a2a_bytes_per_token(c, 8, dedup)
            r[f'a2a_dedup={dedup}'] = dict(rows=rows, mb_per_token=by / 1e6)
        r['mem_5090_ep4'] = memory_gb(c, '5090', 4, 1, TRAIN['seq_len'])
        r['a2a_ep4'] = dict(zip(('rows', 'mb_per_token'), a2a_bytes_per_token(c, 4, True, M=2 if c.route_groups else 0)))
        r['a2a_ep4']['mb_per_token'] /= 1e6
        r['grad_sync'] = dict(zip(('gb_per_node', 'seconds'), grad_sync_s(c, '4090')))
        for gpu, micro in (('4090', 1), ('5090', 2)):
            r[f'mem_{gpu}'] = memory_gb(c, gpu, 8, micro, TRAIN['seq_len'])
            r[f'mem_{gpu}_norecompute'] = memory_gb(c, gpu, 8, micro, TRAIN['seq_len'], recompute=False)
            for sc in SCENARIOS:
                r[f'time_{gpu}_{sc}'] = step_time(c, gpu, sc, micro, dedup=(name == 'moe7b'))
                r[f'time_{gpu}_{sc}_dedup'] = step_time(c, gpu, sc, micro, dedup=True)
        out[name] = r
    return out


def fmt(x, nd=2):
    return f'{x:,.{nd}f}'


def markdown(rep):
    L = []
    for name, r in rep.items():
        p = r['params']
        L.append(f'### {name}\n')
        L.append(f"params: total {p['total']:,} | active {p['active']:,} (incl. embed+head) | "
                 f"active non-embed {p['active_nonembed']:,} | routed experts {p['routed_experts']:,}\n")
        L.append(f"FLOPs/token (train): {r['flops_per_token']/1e9:.2f} G (matmul {r['flops_matmul']/1e9:.2f} G, "
                 f"attention {r['flops_attn']/1e9:.2f} G)\n")
        h = r['hparams']
        L.append(f"DeepSeek-LLM fit for D=140B: C={h['C']:.3g}, lr_opt={h['lr']:.2e}, B_opt={h['batch_tokens']/1e6:.2f}M tokens\n")
        for dd in (True, False):
            a = r[f'a2a_dedup={dd}']
            L.append(f"all-to-all (EP=8, dedup={dd}): {a['rows']:.2f} rows/token/layer, {a['mb_per_token']:.3f} MB sent per token per step\n")
        g = r['grad_sync']
        L.append(f"inter-node grad all-reduce: {g['gb_per_node']:.1f} GB per node per step = {g['seconds']:.1f} s at "
                 f"{CLUSTER['nic_gbs']*CLUSTER['nic_eff']:.2f} GB/s effective\n")
        L.append('\n| GPU | micro-batch | states GB | activations GB | other GB | total GB | fits | no-recompute total GB |\n|---|---|---|---|---|---|---|---|')
        for gpu, micro in (('4090', 1), ('5090', 2)):
            m, m2 = r[f'mem_{gpu}'], r[f'mem_{gpu}_norecompute']
            L.append(f"| {gpu} | {micro}x4096 | {fmt(m['states'],1)} | {fmt(m['activations'],1)} | {fmt(m['other'],1)} | "
                     f"{fmt(m['total'],1)} | {m['fits']} | {fmt(m2['total'],1)} |")
        m4, a4 = r['mem_5090_ep4'], r['a2a_ep4']
        L.append(f"\n5090 socket-local EP=4 (micro-batch 1, M=2 of 4 groups): {fmt(m4['total'],1)} GB/GPU (fits={m4['fits']}), "
                 f"{a4['rows']:.2f} rows/token/layer, {a4['mb_per_token']:.3f} MB per token per step")
        L.append('\n| GPU | scenario | compute s | a2a s | grad-sync s (exposed) | step s | tok/s/GPU | MFU | tokens in 15 d |\n|---|---|---|---|---|---|---|---|---|')
        for gpu in ('4090', '5090'):
            for sc in SCENARIOS:
                for tag in ('', '_dedup') if name == 'olmoe' else ('',):
                    t = r[f'time_{gpu}_{sc}{tag}']
                    L.append(f"| {gpu} | {sc}{' +dedup' if tag else ''} | {fmt(t['compute_s'])} | {fmt(t['a2a_s'])} | "
                             f"{fmt(t['sync_s'])} ({fmt(t['sync_exposed_s'])}) | {fmt(t['step_s'])} | "
                             f"{fmt(t['tok_per_s_gpu'],0)} | {t['mfu']:.0%} | {t['tokens_15d']/1e9:,.0f} B |")
        L.append('')
    return '\n'.join(L)


def apply_measured(comm, gpu, compute_eff=None, hidden=0.3, msg_mb=64):
    """Replace assumptions with a bench/comm.py result: SHM bandwidth from the all-to-all closest to the
    training message size (~64 MB/rank/layer at micro-batch 1), NIC efficiency from the largest all-reduce,
    and optionally add a 'measured' scenario with a profiled compute efficiency."""
    a2a = min(comm['a2a'], key=lambda r: abs(r['mb_per_rank'] - msg_mb))
    GPUS[gpu]['shm_gbs'] = (a2a['remote_gbs_per_gpu'],) * 2
    if comm.get('allreduce'):
        ar = max(comm['allreduce'], key=lambda r: r['mb_per_rank'])
        CLUSTER['nic_eff'] = ar['node_gbs'] / CLUSTER['nic_gbs']
    if compute_eff:
        SCENARIOS['measured'] = (compute_eff, hidden)
    return dict(shm_gbs=GPUS[gpu]['shm_gbs'][0], nic_eff=CLUSTER['nic_eff'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--measured', default=None, help='bench/comm.py JSON from the cluster')
    ap.add_argument('--gpu', default='4090', choices=sorted(GPUS))
    ap.add_argument('--compute-eff', type=float, default=None, help='profiled GEMM-weighted efficiency')
    args = ap.parse_args()
    if args.measured:
        with open(args.measured) as f:
            print('measured:', apply_measured(json.load(f), args.gpu, args.compute_eff), file=sys.stderr)
    rep = report()
    print(json.dumps(rep, indent=1, default=float) if args.json else markdown(rep))

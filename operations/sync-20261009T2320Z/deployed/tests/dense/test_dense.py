"""Dense fast path: owner-sharded ZeRO equals single-process training, fused == unfused projections,
document masking == running each document alone, deterministic resume."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from model.config import get_config
from model.layers import doc_block_mask
from model.model import Transformer, init_weights
from train.data import write_shard
from train.optim import MuonAdamW, muon_update, param_groups
from train.train_dense import ckpt_indices, flops_per_token, main
from train.zero import OwnerZeRO, assign_owners

ROOT = Path(__file__).resolve().parents[2]
TINY = dict(n_dense_layers=3, eos_id=255, fuse_proj=True, doc_mask=False)


def make_shards(tmp_path, n=3, docs=40, vocab=256, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        docs_i = [(rng.integers(0, 250) + np.arange(rng.integers(5, 90)) * rng.integers(1, 4)) % (vocab - 3) + 2
                  for _ in range(docs)]
        write_shard(tmp_path / f'shard_{i}.u16', docs_i, eos=vocab - 1)
    return str(tmp_path / 'shard_*.u16')


def argv(data, steps=4, extra=(), override=None):
    return ['--preset', 'tiny', '--override', json.dumps(override or TINY), '--data', data, '--steps', str(steps),
            '--global-batch', '8', '--micro-batch', '2', '--lr', '3e-3', '--warmup', '1', '--device', 'cpu',
            '--dtype', 'fp32', '--log-every', '1000', *extra]


def torchrun(nproc, args, tmp_path, tag):
    out = tmp_path / f'{tag}.jsonl'
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc-per-node', str(nproc), '--master-port',
           str(29500 + hash(tag) % 400), '-m', 'train.train_dense', *args, '--metrics', str(out)]
    env = dict(os.environ, PYTHONPATH=f'{ROOT}:{ROOT}/.deps', OMP_NUM_THREADS='1')
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return [json.loads(l) for l in out.read_text().splitlines()]


def test_assign_owners_balanced():
    sizes = [100, 90, 50, 40, 30, 20, 10, 5]
    own = assign_owners(sizes, 3)
    loads = [sum(s for s, o in zip(sizes, own) if o == r) for r in range(3)]
    assert max(loads) - min(loads) <= 30


@pytest.mark.parametrize('sync_every,scope', [(0, 'node'), (1, 'node'), (0, 'world')])
def test_zero_two_ranks_match_single_process(tmp_path, sync_every, scope):
    data = make_shards(tmp_path)
    ref, _ = main(argv(data, extra=['--sync-every', str(sync_every)]))
    got = torchrun(2, argv(data, extra=['--sync-every', str(sync_every), '--shard-scope', scope]), tmp_path,
                   f'z{sync_every}{scope}')
    for a, b in zip(ref, got):
        assert abs(a['loss'] - b['loss']) < 1e-4 and abs(a['grad_norm'] - b['grad_norm']) < 1e-3 * a['grad_norm']
    assert ref[-1]['loss'] < ref[0]['loss']


def test_zero_step_equals_plain_optimizer():
    torch.manual_seed(0)
    cfg = get_config('tiny', **TINY)
    m1, m2 = Transformer(cfg), Transformer(cfg)
    init_weights(m1, 0); init_weights(m2, 0)
    x = torch.randint(0, 255, (2, 32)); y = torch.roll(x, -1, 1)
    zero = OwnerZeRO(m1, lr=1e-2, dtype=torch.float32)
    opt = MuonAdamW(param_groups(m2), lr=1e-2)
    for _ in range(3):
        m1(x, y)['loss'].backward(); zero.after_backward(0, 1); zero.step(1e-2, 0.0)
        m2(x, y)['loss'].backward(); opt.step(); opt.zero_grad(set_to_none=True)
    for (n, a), (_, b) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(a, b, atol=1e-5), n


def test_fused_projection_equals_unfused():
    torch.manual_seed(0)
    cu, cf = get_config('tiny', **dict(TINY, fuse_proj=False)), get_config('tiny', **TINY)
    mu, mf = Transformer(cu), Transformer(cf)
    init_weights(mu, 0)
    sd = mu.state_dict(); new = {}
    for k, v in sd.items():
        if k.endswith('attn.wq.weight'):
            pre = k[:-len('wq.weight')]
            new[pre + 'wqkvg.weight'] = torch.cat([sd[pre + f'w{s}.weight'] for s in 'qkvg'])
        elif k.endswith('ffn.w1.weight'):
            pre = k[:-len('w1.weight')]
            new[pre + 'w13.weight'] = torch.cat([sd[pre + 'w1.weight'], sd[pre + 'w3.weight']])
        elif not k.endswith(('attn.wk.weight', 'attn.wv.weight', 'attn.wg.weight', 'ffn.w3.weight')):
            new[k] = v
    mf.load_state_dict(new)
    x = torch.randint(0, 255, (2, 32))
    assert torch.allclose(mu(x), mf(x), atol=1e-5)
    w = mf.layers[0].attn.wqkvg.weight
    rows = w._muon_rows
    u = torch.cat([muon_update(b) for b in w.detach().split(rows, dim=-2)], dim=-2)
    assert u.shape == w.shape and not torch.allclose(u, muon_update(w.detach()))


def test_doc_mask_equals_documents_alone():
    torch.manual_seed(0)
    cfg = get_config('tiny', **dict(TINY, doc_mask=True))
    m = Transformer(cfg); init_weights(m, 0); m.eval()
    a, b = torch.randint(2, 250, (1, 20)), torch.randint(2, 250, (1, 12))
    packed = torch.cat([a, torch.tensor([[255]]), b], dim=1)          # doc a + EOS | doc b
    with torch.no_grad():
        out = m(packed)
        alone_a = m(torch.cat([a, torch.tensor([[255]])], dim=1))
        alone_b = m(b)
    assert torch.allclose(out[:, :21], alone_a, atol=1e-4)
    assert torch.allclose(out[:, 21:], alone_b, atol=1e-4)           # RoPE is relative: offset does not matter
    bm = doc_block_mask(packed, 255)
    assert bm is not None


def test_resume_is_deterministic(tmp_path):
    data = make_shards(tmp_path)
    full, _ = main(argv(data, steps=4))
    ck = tmp_path / 'ck'
    main(argv(data, steps=2, extra=['--ckpt-dir', str(ck), '--ckpt-every', '2']))
    rest, _ = main(argv(data, steps=4, extra=['--ckpt-dir', str(ck), '--resume']))
    assert [r['step'] for r in rest] == [2, 3]
    for a, b in zip(full[2:], rest):
        assert abs(a['loss'] - b['loss']) < 1e-5


def test_checkpointed_layers_same_grads():
    torch.manual_seed(0)
    cfg = get_config('tiny', **TINY)
    m = Transformer(cfg); init_weights(m, 0)
    x = torch.randint(0, 255, (2, 32)); y = torch.roll(x, -1, 1)
    m(x, y)['loss'].backward(); g0 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(); m.ckpt_layers = ckpt_indices(cfg.n_layers, 2)
    m(x, y)['loss'].backward()
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(g0, m.parameters()))


def test_flops_formula_dense3b():
    c = get_config('dense3b')
    f = flops_per_token(c, 4096)
    assert 2.0e10 < f < 2.3e10       # 6 x 3.24B matmul params + causal attention


def test_bf16_zero_trains(tmp_path):
    data = make_shards(tmp_path)
    hist, _ = main(argv(data, steps=6, extra=['--dtype', 'bf16']))
    assert all(np.isfinite(h['loss']) for h in hist) and hist[-1]['loss'] < hist[0]['loss']


def test_validation_loss_logged(tmp_path):
    data = make_shards(tmp_path)
    hist, _ = main(argv(data, steps=4, extra=['--val-data', data, '--val-every', '2', '--val-windows', '4']))
    vals = [h['val_loss'] for h in hist if 'val_loss' in h]
    assert len(vals) == 2 and all(np.isfinite(v) for v in vals)


def test_resume_on_different_world_size(tmp_path):
    """Checkpoint written by 2 ranks resumes on 1 rank with the uninterrupted trajectory (elastic GPU count)."""
    data = make_shards(tmp_path)
    full, _ = main(argv(data, steps=4))
    ck = tmp_path / 'ck'
    torchrun(2, argv(data, steps=2, extra=['--ckpt-dir', str(ck), '--ckpt-every', '2', '--shard-scope', 'world']),
             tmp_path, 'elastic')
    rest, _ = main(argv(data, steps=4, extra=['--ckpt-dir', str(ck), '--resume']))
    assert [r['step'] for r in rest] == [2, 3]
    for a, b in zip(full[2:], rest):
        assert abs(a['loss'] - b['loss']) < 1e-4


def test_prune_ckpt_keeps_last_and_slims_milestones(tmp_path):
    from bench.prune_ckpt import prune
    for s in (249, 499, 749, 999, 1249, 1499):
        d = tmp_path / f'step_{s:08d}'; d.mkdir()
        for f in ('COMPLETE', 'meta.json', 'model.pt', 'rank000.pt', 'rank001.pt'):
            (d / f).write_text('x')
    (tmp_path / 'step_00001749').mkdir()                      # being written: no COMPLETE, untouched
    (tmp_path / 'step_00001749' / 'rank000.pt').write_text('x')
    prune(str(tmp_path), keep_last=3, milestone_every=500)
    left = {p.name: sorted(f.name for f in p.iterdir()) for p in tmp_path.iterdir()}
    assert set(left) == {'step_00000499', 'step_00000999', 'step_00001249', 'step_00001499', 'step_00001749'}
    assert left['step_00000499'] == ['MILESTONE', 'meta.json', 'model.pt']
    assert 'COMPLETE' in left['step_00000999'] and 'rank000.pt' in left['step_00000999']   # within last 3
    assert left['step_00001749'] == ['rank000.pt']


def test_per_source_validation(tmp_path):
    a = tmp_path / 'a'; b = tmp_path / 'b'; a.mkdir(); b.mkdir()
    make_shards(a, n=1); make_shards(b, n=1, seed=1)
    data = make_shards(tmp_path)
    hist, _ = main(argv(data, steps=2, extra=['--val-data', f'{a}/shard_*.u16,{b}/shard_*.u16', '--val-every', '2',
                                              '--val-windows', '8']))
    last = hist[-1]
    assert np.isfinite(last['val_loss']) and np.isfinite(last['val_a']) and np.isfinite(last['val_b'])


def test_kv_cache_batched_greedy_equals_naive():
    """Left-padded batched KV-cache decoding gives exactly the per-prompt naive greedy continuation."""
    from model.generate import greedy
    torch.manual_seed(0)
    cfg = get_config('tiny', **dict(TINY, attn_gate=True))
    m = Transformer(cfg); init_weights(m, 0); m.eval()
    prompts = [[5, 9, 13, 2, 7, 7, 30], [100, 3], [8, 8, 8, 8, 41, 2, 9, 9, 17, 4, 3]]

    def naive(p, n):
        ids = list(p)
        with torch.no_grad():
            for _ in range(n):
                logits = m(torch.tensor([ids]))[0, -1]
                logits[250:] = -float('inf')
                t = int(logits.argmax())
                if t == 255:
                    break
                ids.append(t)
        return ids[len(p):]

    got = greedy(m, prompts, 12, eos=255, ban_from=250)
    assert got == [naive(p, 12) for p in prompts]

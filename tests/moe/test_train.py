import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from model.config import get_config
from model.model import Transformer, init_weights
from train import ckpt
from train.data import EOS, PackedTokens, shard_paths, write_shard
from train.optim import MuonAdamW, muon_update, newton_schulz, param_groups, wsd_lr
from train.parallel import update_router_bias
from train.train import main


def tiny_model(seed=0):
    torch.manual_seed(seed)
    m = Transformer(get_config('tiny'))
    init_weights(m, seed)
    return m


def make_shards(tmp_path, n=3, docs=40, vocab=256, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        # learnable documents: arithmetic progressions mod (vocab - 3), offset past PAD
        docs_i = [(rng.integers(0, 250) + np.arange(rng.integers(5, 90)) * rng.integers(1, 4)) % (vocab - 3) + 2
                  for _ in range(docs)]
        write_shard(tmp_path / f'shard_{i}.u16', docs_i, eos=vocab - 1)
    return str(tmp_path / 'shard_*.u16')


def test_newton_schulz_orthogonalises():
    g = torch.randn(3, 64, 32)
    o = newton_schulz(g, steps=10)
    s = torch.linalg.svdvals(o)
    assert s.min() > 0.5 and s.max() < 1.25          # NS quintic maps singular values into ~[0.7, 1.2]
    u = muon_update(torch.randn(2, 64, 2 * 32), split=2)
    rms = u.square().mean().sqrt().item()
    assert 0.1 < rms < 0.3                            # Moonlight scaling: RMS ~0.2


def test_param_groups_route_by_kind():
    m = tiny_model()
    groups = {g['kind'] + str(g['weight_decay']): g['params'] for g in param_groups(m)}
    muon = {id(p) for p in groups['muon0.1']}
    assert id(m.layers[1].ffn.w13) in muon and id(m.layers[0].attn.wq.weight) in muon
    assert id(m.layers[1].ffn.router.weight) not in muon and id(m.embed.weight) not in muon
    assert id(m.embed.weight) in {id(p) for p in groups['adamw0.0']}


def test_wsd_schedule():
    lrs = [wsd_lr(s, 100, 1.0, 10, 0.2) for s in range(100)]
    assert lrs[0] == 0.1 and lrs[9] == 1.0 and lrs[50] == 1.0 and lrs[79] == 1.0
    assert lrs[99] < 0.25 and all(a >= b for a, b in zip(lrs[80:], lrs[81:]))


def test_tiny_model_overfits_one_batch():
    m = tiny_model()
    opt = MuonAdamW(param_groups(m), lr=3e-3)
    x = torch.randint(0, 255, (4, 64))
    y = torch.roll(x, -1, 1)
    losses = []
    for _ in range(60):
        out = m(x, y)
        out['loss'].backward()
        opt.step()
        opt.zero_grad()
        update_router_bias(m)
        losses.append(out['ce'].item())
    assert losses[-1] < 0.3 * losses[0], losses[::10]


def test_checkpoint_round_trip(tmp_path):
    m = tiny_model()
    opt = MuonAdamW(param_groups(m), lr=1e-3)
    x = torch.randint(0, 255, (2, 64))
    for _ in range(2):
        m(x, x)['loss'].backward()
        opt.step()
        opt.zero_grad()
        update_router_bias(m)
    t = ckpt.save(tmp_path, 7, m, opt, extra=dict(note='x'), async_save=True, upload=lambda p: None)
    t.join()
    m2 = tiny_model(seed=1)
    opt2 = MuonAdamW(param_groups(m2), lr=1e-3)
    step, extra = ckpt.load(ckpt.latest(tmp_path), m2, opt2)
    assert step == 7 and extra['note'] == 'x'
    for (k, a), b in zip(m.state_dict().items(), m2.state_dict().values()):
        assert torch.equal(a, b), k
    assert m2.layers[1].ffn.router.bias.abs().sum() > 0          # router bias is part of the checkpoint
    for o in (opt, opt2):                                         # identical next step
        out = (m if o is opt else m2)(x, x)['loss']
        out.backward()
        o.step()
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_incomplete_checkpoint_ignored(tmp_path):
    (tmp_path / 'step_00000009').mkdir()
    (tmp_path / 'step_00000009' / 'rank_00000.tmp').write_bytes(b'')
    assert ckpt.latest(tmp_path) is None


def test_loader_deterministic_and_disjoint(tmp_path):
    paths = shard_paths(make_shards(tmp_path))
    a, b = PackedTokens(paths, 16, seed=5), PackedTokens(paths, 16, seed=5)
    xa, ya = a.batch(3, 1, 2, 0, 4, 16)
    xb, yb = b.batch(3, 1, 2, 0, 4, 16)
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert torch.equal(xa[:, 1:], ya[:, :-1])
    assert not torch.equal(a.batch(3, 0, 2, 0, 4, 16)[0], xa)        # other DP rank, other data
    assert not torch.equal(PackedTokens(paths, 16, seed=6).batch(3, 1, 2, 0, 4, 16)[0], xa)
    assert torch.equal(a.batch(len(a) // 16, 0, 1, 0, 4, 16)[0][0, :], torch.from_numpy(a.window(len(a) // 16 * 16)[:-1]))


def test_loader_epoch_is_permutation_and_repeat(tmp_path):
    paths = shard_paths(make_shards(tmp_path, n=1))
    d = PackedTokens(paths, 16, seed=0)
    perm = d._perm_for(0)
    assert sorted(perm.tolist()) == list(range(len(d)))
    r = PackedTokens([paths[0] + ':3'], 16, seed=0)
    assert len(r) == 3 * len(d)
    raw = np.fromfile(paths[0], dtype='<u2')
    assert raw[-1] == 255 and (raw == 255).sum() == 40              # EOS-separated documents


def test_single_process_training_smoke(tmp_path):
    data = make_shards(tmp_path)
    log = tmp_path / 'log.jsonl'
    argv = ['--preset', 'tiny', '--data', data, '--steps', '20', '--global-batch', '4', '--micro-batch', '2',
            '--lr', '5e-3', '--warmup', '2', '--device', 'cpu', '--metrics', str(log),
            '--ckpt-dir', str(tmp_path / 'ck'), '--ckpt-every', '4']
    hist, _ = main(argv)
    assert len(hist) == 20 and all(np.isfinite(h['loss']) for h in hist)
    assert np.mean([h['loss'] for h in hist[-4:]]) < hist[0]['loss'] - 0.5
    assert all(h['dropped'] == 0 for h in hist)
    assert all(0 <= h['load_entropy_min'] <= 1 for h in hist)
    assert len(log.read_text().splitlines()) == 20 and json.loads(log.read_text().splitlines()[0])['step'] == 0
    # resume from the step-19 checkpoint and run 2 more steps
    hist2, _ = main(argv[:5] + ['22'] + argv[6:] + ['--resume'])
    assert [h['step'] for h in hist2] == [20, 21]


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('sync_dtype,tol', [('fp32', 1e-4), ('bf16', 2e-2)])
def test_hsdp_ep_matches_single_process(tmp_path, sync_dtype, tol):
    """4 CPU processes = 2 replicas x (FSDP2 shard + EP) of size 2 reproduce single-process training
    (exactly with an fp32 cross-replica expert all-reduce, within bf16 noise with the default bf16 one)."""
    data = make_shards(tmp_path, docs=60)
    common = ['--preset', 'tiny', '--data', data, '--steps', '4', '--global-batch', '8', '--lr', '5e-3',
              '--warmup', '2', '--device', 'cpu']
    main(common + ['--micro-batch', '2', '--metrics', str(tmp_path / 'one.jsonl')])
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node', '4',
                        '-m', 'train.train'] + common + ['--micro-batch', '1', '--ep', '2', '--recompute-experts',
                        '--grad-sync-dtype', sync_dtype, '--metrics', str(tmp_path / 'four.jsonl')], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    one = [json.loads(l) for l in (tmp_path / 'one.jsonl').read_text().splitlines()]
    four = [json.loads(l) for l in (tmp_path / 'four.jsonl').read_text().splitlines()]
    for a, b in zip(one, four):
        assert abs(a['loss'] - b['loss']) < tol and abs(a['grad_norm'] - b['grad_norm']) < tol
    assert four[0]['a2a_rows_per_pair'] < 1.0                     # dedup active across 2 EP ranks


def test_all_dense_training_smoke(tmp_path):
    """dense3b path: every layer dense, so no MoE layers, router stats or expert grad sync."""
    data = make_shards(tmp_path)
    argv = ['--preset', 'tiny', '--override', '{"n_dense_layers": 3}', '--data', data, '--steps', '20',
            '--global-batch', '4', '--micro-batch', '2', '--lr', '5e-3', '--warmup', '2', '--device', 'cpu']
    hist, model = main(argv)
    assert not list(model.moe_layers())
    assert len(hist) == 20 and all(np.isfinite(h['loss']) for h in hist)
    assert np.mean([h['loss'] for h in hist[-4:]]) < hist[0]['loss'] - 0.5

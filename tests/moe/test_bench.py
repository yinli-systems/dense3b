"""The Day-1 kit must work the first time it lands on the cluster: run every piece at toy size on CPU."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tools.budget as budget
from bench.synth_data import main as synth
from train.data import EOS, PackedTokens
from train.train import main as train_main

ROOT = Path(__file__).resolve().parents[2]


def run(args, env=None, timeout=900):
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, PYTHONPATH=str(ROOT), **(env or {})))
    assert r.returncode == 0, r.stderr[-3000:]
    return r


def test_comm_benchmark_gloo(tmp_path):
    out = tmp_path / 'comm.json'
    run(['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node', '4', '-m', 'bench.comm', '--ep', '2',
         '--a2a-mb', '0.25,1', '--ar-mb', '0.25,1', '--iters', '2', '--warmup', '1', '--out', str(out)])
    r = json.loads(out.read_text())
    assert r['replicas'] == 2 and len(r['a2a']) == 2 and len(r['allreduce']) == 2
    assert all(x['remote_gbs_per_gpu'] > 0 for x in r['a2a'])
    assert r['concurrent']['n_a2a'] >= 1 and r['concurrent']['overlap_efficiency'] > 0


def test_gemm_autotuner_dry_run(tmp_path):
    pytest.importorskip('triton')
    out = tmp_path / 'gmm.json'
    run(['-m', 'bench.gemm', '--experts', '3', '--rows', '24', '--d-model', '32', '--d-expert', '16',
         '--iters', '1', '--quick', '--out', str(out)], env=dict(TRITON_INTERPRET='1'))
    r = json.loads(out.read_text())
    for case in r['cases'].values():
        assert len(case) == 6 and all(op['rel_err'] < 1e-3 for op in case.values())
    cfg = json.loads((tmp_path / 'gmm.config.json').read_text())
    assert set(cfg) == {'fwd', 'wgrad'} and {'BM', 'BN', 'BK', 'num_warps', 'num_stages'} <= set(cfg['fwd'])
    # the written config is what the model loads through MOE_GMM_CONFIG
    probe = run(['-c', 'from model import grouped_gemm as G; import json; print(json.dumps([G.BLOCK, G.BLOCK_WGRAD]))'],
                env=dict(MOE_GMM_CONFIG=str(tmp_path / 'gmm.config.json')))
    assert json.loads(probe.stdout) == [cfg['fwd'], cfg['wgrad']]


def test_synthetic_shards(tmp_path):
    paths = synth([str(tmp_path), '--shards', '2', '--tokens-per-shard', '20000'])
    assert len(paths) == 2
    raw = np.fromfile(paths[0], dtype='<u2')
    assert raw.max() < 50_280 and raw[-1] == EOS and raw.size >= 20000
    d = PackedTokens(paths, 4096)
    x, y = d.batch(0, 0, 1, 0, 2, 2)
    assert x.shape == (2, 4096)


def test_budget_applies_measurements(monkeypatch):
    monkeypatch.setattr(budget, 'GPUS', copy.deepcopy(budget.GPUS))
    monkeypatch.setattr(budget, 'CLUSTER', dict(budget.CLUSTER))
    monkeypatch.setattr(budget, 'SCENARIOS', dict(budget.SCENARIOS))
    comm = dict(a2a=[dict(mb_per_rank=16, remote_gbs_per_gpu=9.0), dict(mb_per_rank=64, remote_gbs_per_gpu=12.0)],
                allreduce=[dict(mb_per_rank=1024, node_gbs=3.125)])
    got = budget.apply_measured(comm, '4090', compute_eff=0.35)
    assert got == dict(shm_gbs=12.0, nic_eff=0.5)
    t = budget.report()['moe7b']['time_4090_measured']
    assert t['tokens_15d'] > 0 and t['sync_s'] > 5.6            # half the assumed NIC efficiency -> slower sync


def test_train_timing_fields(tmp_path):
    paths = synth([str(tmp_path), '--shards', '1', '--tokens-per-shard', '5000', '--vocab', '255', '--eos', '255'])
    hist, _ = train_main(['--preset', 'tiny', '--data', paths[0], '--steps', '2', '--global-batch', '2',
                          '--device', 'cpu', '--timing'])
    assert {'t_fwd_bwd', 't_grad_sync_wait', 't_optim'} <= set(hist[-1])

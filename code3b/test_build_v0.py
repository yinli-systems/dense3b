"""Unit tests for build_v0: byte-range chunking, dedup union-find, best-member choice, end-to-end tiny run."""
import json, os, sys
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_v0 as B


def test_chunk_partition_is_exact(tmp_path):
    p = tmp_path / 'x.jsonl'
    rows = [{'text': f'doc {i} ' + 'y' * (i % 37)} for i in range(5000)]
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    size = p.stat().st_size
    for chunk in (1, 7, 100, 4096, size):
        got = []
        for b in range(0, size, chunk):
            got += [r['text'] for r in B.read_rows(str(p), 'jsonl', start=b, end=min(b + chunk, size))]
        assert got == [r['text'] for r in rows], chunk


def test_union_find_components():
    keys = np.array([5, 0, 5, 7, 0, 7, 5, 9], np.uint64)
    x, y = B.collision_pairs(keys)
    parent = list(range(8)); B.union_pairs(parent, x, y)
    r = [B.find(parent, i) for i in range(8)]
    assert r[0] == r[2] == r[6] and r[3] == r[5]
    assert len({r[0], r[1], r[3], r[4], r[7]}) == 5          # key 0 = "no key", never joins


def test_ngram_hash_detects_copy_and_ignores_short():
    bench = B.ngram_hashes('def has_close_elements(numbers, threshold): check if in given list of numbers '
                           'are any two numbers closer to each other than given threshold')
    doc = 'x = 1\n# ' + 'Check if in given list of numbers, are any two numbers closer to each other than given threshold'
    assert np.isin(B.ngram_hashes(doc), bench).any()
    assert B.ngram_hashes('too short text').size == 0


def test_end_to_end_tiny(tmp_path):
    raw = tmp_path / 'raw' / 'src'; raw.mkdir(parents=True)
    texts = [f'unique document number {i} ' + 'alpha beta gamma ' * 20 for i in range(30)]
    dup = texts[3]
    contaminated = 'prefix ' + 'check if in given list of numbers are any two numbers closer to each other than given threshold'
    (raw / 'a.jsonl').write_text(''.join(json.dumps({'text': t}) + '\n' for t in texts + [dup, contaminated]))
    bench_raw = tmp_path / 'bench' / 'raw' / 'he'; bench_raw.mkdir(parents=True)
    pq.write_table(pa.table({'prompt': ['Check if in given list of numbers, are any two numbers closer to each '
                                        'other than given threshold. ' * 2]}), bench_raw / 'he.parquet')
    tok = '/data/run01/scxi253/moe7b-quality-stage2-20261002T2327Z/tokenizer.json'
    cfg = dict(raw_root=str(tmp_path / 'raw'), bench_root=str(tmp_path / 'bench'), work=str(tmp_path / 'work'),
               shards=str(tmp_path / 'shards'), tokenizer=tok,
               background=dict(sources=['src'], docs=10, min_docs=1000),
               sources=[dict(name='src', glob='src/*.jsonl', format='jsonl', rank=1)])
    cfgp = tmp_path / 'c.json'; cfgp.write_text(json.dumps(cfg))
    A = lambda **k: type('A', (), dict(dict(only='', max_docs=0, partial=False), **k))
    B.cmd_bench(cfg, A()); B.cmd_map(cfg, A()); B.cmd_reduce(cfg, A()); B.cmd_pack(cfg, A())
    m = json.load(open(next((tmp_path / 'work' / 'map').glob('*.parquet.json'))))
    assert m['counts']['drop/benchmark_13gram'] == 1
    red = json.load(open(tmp_path / 'work' / 'keep' / 'REDUCE.json'))
    assert red['documents'] == 31 and red['kept'] == 30           # exact duplicate removed
    shard = next((tmp_path / 'shards' / 'src').glob('*.u16'))
    arr = np.fromfile(shard, '<u2')
    assert (arr == B.EOS).sum() == 30 and arr[-1] == B.EOS
    meta = json.load(open(str(shard) + '.json'))
    assert meta['documents'] == 30 and meta['tokens'] == arr.size


def test_in_sorted_matches_isin():
    rng = np.random.default_rng(0)
    arr = np.unique(rng.integers(0, 1 << 20, 5000).astype(np.uint64))
    h = rng.integers(0, 1 << 20, 3000).astype(np.uint64)
    assert (B.in_sorted(h, arr) == np.isin(h, arr)).all()
    assert B.in_sorted(np.empty(0, np.uint64), arr).size == 0
    assert not B.in_sorted(h, np.empty(0, np.uint64)).any()


def test_strong_windows_reject_syntax_and_numbers():
    for t in ('for int i 0 i n i for int j 0 j n j', 'dp i 1 j dp i j 1 dp i 1 j 1 dp',
              '1 0 1 0 0 1 0 1 1 1 1 1 1 0', 'zero 0 one 1 two 2 three 3 four 4 five 5 six 6'):
        assert not B.strong_windows(t).any(), t
    assert B.strong_windows('find all possible combinations of k numbers that add up to a number n').any()


def test_bench_drops_boilerplate_shared_across_problems(tmp_path):
    raw = tmp_path / 'raw' / 'src'; raw.mkdir(parents=True)
    (raw / 'a.jsonl').write_text(json.dumps({'text': 'nothing here ' * 30}) + '\n')
    boiler = 'import logging logging basicconfig level logging info format asctime levelname message'
    rows = [{'prompt': f'problem number {i} computes something special {w} ' + boiler + f' then returns value {w} result'}
            for i, w in enumerate(['alpha', 'bravo', 'charlie', 'delta', 'echo'])]
    rows.append({'prompt': 'write a function that finds all possible combinations of numbers adding up to target value '
                           'while avoiding duplicate combinations entirely'})
    b = tmp_path / 'bench' / 'raw' / 'x'; b.mkdir(parents=True)
    pq.write_table(pa.table({'prompt': [r['prompt'] for r in rows]}), b / 'x.parquet')
    cfg = dict(raw_root=str(tmp_path / 'raw'), bench_root=str(tmp_path / 'bench'), work=str(tmp_path / 'work'),
               shards=str(tmp_path / 'shards'), tokenizer='', background=dict(sources=['src'], docs=10, min_docs=1000),
               sources=[dict(name='src', glob='src/*.jsonl', format='jsonl', rank=1)])
    B.cmd_bench(cfg, None)
    bench = np.load(tmp_path / 'work' / 'bench_13gram.npy')
    doc_boiler = 'x = 1\n' + boiler + ' and more'
    doc_specific = 'finds all possible combinations of numbers adding up to target value while avoiding duplicate'
    assert not B.in_sorted(B.ngram_hashes(doc_boiler), bench).any()
    assert B.in_sorted(B.ngram_hashes(doc_specific), bench).any()


def test_strip_boiler_lines():
    t = 'import numpy as np\nfrom sklearn.model_selection import train_test_split\n#include <vector>\nusing namespace std;\nreal line'
    assert B.strip_boiler_lines(t) == 'real line'


def test_release_frees_only_dead_unfinished_claims(tmp_path, monkeypatch):
    import subprocess
    work = tmp_path / 'work'; (work / 'map').mkdir(parents=True)
    (work / 'map' / 'a.claim').write_text('111_1 5')            # dead owner, unfinished -> released
    (work / 'map' / 'b.claim').write_text('222_3 6')            # live owner -> kept
    (work / 'map' / 'c.claim').write_text('111_2 7')            # dead owner but finished -> kept
    (work / 'map' / 'c.parquet.json').write_text('{}')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: type('R', (), dict(stdout='222_3 222_3\n'))())
    B.cmd_release(dict(work=str(work), shards=str(tmp_path / 'shards')), None)
    assert sorted(p.name for p in (work / 'map').glob('*.claim')) == ['b.claim', 'c.claim']


def test_fim_split_roundtrip_and_layout():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('/data/run01/scxi253/moe7b-quality-stage2-20261002T2327Z/tokenizer.json')
    text = 'def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n'
    rng = np.random.default_rng([1, 2])
    pre, mid, suf = B.fim_split(text, rng)
    assert pre + mid + suf == text
    lr, fim = B.encode_docs(tok, [text, (pre, mid, suf)])
    assert lr[-1] == B.EOS and fim[-1] == B.EOS and (lr == B.EOS).sum() == 1 and (fim == B.EOS).sum() == 1
    assert fim[0] == B.FIM_PREFIX and list(fim).index(B.FIM_SUFFIX) < list(fim).index(B.FIM_MIDDLE)
    i_s, i_m = list(fim).index(B.FIM_SUFFIX), list(fim).index(B.FIM_MIDDLE)
    dec = lambda a: tok.decode([int(x) for x in a])
    assert dec(fim[1:i_s]) + dec(fim[i_m + 1:-1]) + dec(fim[i_s + 1:i_m]) == text

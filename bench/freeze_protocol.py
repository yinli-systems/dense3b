"""Freeze evaluation protocol v1: dataset hashes, tokenizer hash, LiveCodeBench post-cutoff slice (problem IDs +
dates, checked one by one against the training corpus crawl cutoff), corpus cutoff evidence.

Runs with the data env (pyarrow):  python bench/freeze_protocol.py OUT.json
"""
import glob
import hashlib
import json
import os
import re
import sys

import pyarrow.parquet as pq

B = '/data/run01/scxi253/code3b-data-20261003/bench'
J = f'{B}/jsonl'
RAW = '/data/run01/scxi253/code3b-data-20261003/raw'
WEB = '/data/run01/scxi253/web-strict-20261003/out'
TOK = '/data/run01/scxi253/moe7b-quality-stage2-20261002T2327Z/tokenizer.json'
CC = re.compile(r'CC-MAIN-(\d{4})-(\d{2})')


def sha(p):
    h = hashlib.sha256()
    with open(os.path.realpath(p), 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def cc_date(dump):
    y, w = map(int, CC.search(dump).groups())       # CC-MAIN-YYYY-WW: ISO week of the crawl
    import datetime
    return datetime.date.fromisocalendar(y, w, 7).isoformat()


def corpus_cutoff():
    ev = {}
    # Nemotron-CC-Math 4plus: latest snapshot named in any record's WARC file name (full scan of metadata)
    latest = None
    for f in sorted(glob.glob(f'{RAW}/nvidia__Nemotron-CC-Math-v1/4plus/*.parquet')):
        for w in pq.read_table(f, columns=['metadata']).column('metadata').combine_chunks().field('warc_filename').to_pylist():
            m = CC.search(w or '')
            if m and (latest is None or m.group(0) > latest):
                latest = m.group(0)
    ev['nemotron-cc-math-4plus'] = dict(latest_snapshot=latest, crawl_end=cc_date(latest))
    # web15b: the FineWeb-Edu dumps of the 332 packed files
    pack = json.load(open('/data/run01/scxi253/code3b-train/web15b/PACK.json'))
    files = sorted(f for f in glob.glob(f'{WEB}/**/*.parquet', recursive=True) if os.path.exists(f + '.json'))
    dumps = sorted({CC.search(f).group(0) for f in files[:pack['files_used']]})
    ev['web15b'] = dict(dumps=f'{dumps[0]}..{dumps[-1]}', n_dumps=len(dumps), crawl_end=cc_date(dumps[-1]))
    # documented upstream cutoffs (no per-record dates in these releases)
    ev['swallow-code'] = dict(basis='The Stack v2 (Software Heritage snapshot 2023-09-06), LLM-rewritten', crawl_end='2023-09-06')
    ev['stackv2-edu'] = dict(basis='The Stack v2 (Software Heritage snapshot 2023-09-06)', crawl_end='2023-09-06')
    ev['opc-annealing'] = dict(basis='OpenCoder annealing corpus, released 2024-11', crawl_end='2024-11-30')
    ev['swallow-math'] = dict(basis='FineMath-4+ (Common Crawl up to 2024), LLM-rewritten', crawl_end='2024-12-31')
    cutoff = max(v['crawl_end'] for v in ev.values())
    return cutoff, ev


def main(out):
    cutoff, evidence = corpus_cutoff()
    v6 = [json.loads(l) for l in open(f'{J}/lcb/test6.jsonl')]
    slice_ = [dict(question_id=r['question_id'], platform=r['platform'], contest_date=r['contest_date'][:10],
                   post_cutoff=r['contest_date'][:10] > cutoff) for r in v6]
    assert all(s['post_cutoff'] for s in slice_), [s for s in slice_ if not s['post_cutoff']][:5]
    files = {n: sha(os.path.join(J, n)) for n in ['humaneval.jsonl', 'humanevalplus.jsonl', 'mbppplus.jsonl', 'cruxeval.jsonl',
                                                   'gsm8k_test.jsonl'] + [f'lcb/test{s}.jsonl' for s in ['', '2', '3', '4', '5', '6']]}
    ids = ','.join(s['question_id'] for s in slice_)
    proto = dict(
        protocol='dense3b-base-suite-v1', frozen_by='bench/freeze_protocol.py',
        corpus_crawl_cutoff=cutoff, corpus_cutoff_evidence=evidence,
        lcb_v6_slice=dict(name='LiveCodeBench v6 slice: the 175 problems added in release_v6 (release_v6 is 1055 cumulative)',
                          n=len(slice_), first_date=min(s['contest_date'] for s in slice_),
                          last_date=max(s['contest_date'] for s in slice_), ids_sha256=hashlib.sha256(ids.encode()).hexdigest(),
                          problems=slice_),
        dataset_sha256=files, tokenizer_sha256=sha(TOK),
        decoding=dict(temperature=0, kv_cache='model/generate.py (token-identical to naive greedy, unit-tested)',
                      eos_id=50279, banned_ids='>=50280 (FIM sentinels, padding)'),
    )
    json.dump(proto, open(out, 'w'), indent=1)
    print(json.dumps(dict(cutoff=cutoff, evidence=evidence, lcb_slice=dict(n=len(slice_), first=proto['lcb_v6_slice']['first_date'],
                                                                             ids_sha256=proto['lcb_v6_slice']['ids_sha256'])), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])

"""Validate every fused part, then build (a) global LSH candidate groups with the unmodified v1 grouping
code, (b) aggregated benchmark counts, (c) per-source DCLM score histograms. No deletions."""
import collections, gzip, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

V1 = Path('/data/run01/scxi253/moe100b-parallel-quality-20261002T1701Z')
sys.path.insert(0, str(V1 / 'source')); sys.path.insert(0, str(V1 / 'deps'))
from quality_core import sha, atomic
from reduce_evidence import groups

ROOT = Path(__file__).resolve().parent.parent
BINS = np.linspace(0.0, 1.0, 1001)


def main():
    plan_path = ROOT / 'config/plan.json'; plan = json.loads(plan_path.read_text()); ph = sha(plan_path)
    out = ROOT / 'global'; out.mkdir(exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        raise RuntimeError('Refuse completed aggregate overwrite')
    expected = {(t['source_id'], t['start'], t['stop']) for p in plan['parts'] for t in p['tasks']}
    seen = set(); total = collections.Counter(); receipts = []
    for part in plan['parts']:
        d = json.loads((ROOT / 'reports' / ('part-%d.COMPLETE.json' % part['part'])).read_text())
        if d['status'] != 'PART_COMPLETE_NOT_FINAL_ADMISSION' or d['identity']['plan_sha256'] != ph:
            raise ValueError('Incomplete parent part %d' % part['part'])
        for rp in d['receipts']:
            r = json.loads(Path(rp).read_text()); i = r['identity']; key = (i['source_id'], i['start'], i['stop'])
            if key in seen or key not in expected:
                raise ValueError('Duplicate or unknown evidence chunk')
            if i['plan_sha256'] != ph or i['mode'] != 'fused_near_benchmark_dclm' or i['canary_documents']:
                raise ValueError('Wrong evidence scope')
            seen.add(key); total.update(r['counts']); receipts.append((rp, r))
    if seen != expected or total['documents'] != plan['target']['documents'] or total['text_tokens'] != plan['target']['text_tokens']:
        raise ValueError('Global document/token coverage differs')
    tmp = out / 'sort-temp'; tmp.mkdir(exist_ok=True)
    sort = subprocess.Popen(['sort', '-T', str(tmp), '-S', '24G', '--parallel=4', '-t', '\t', '-k1,1'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=dict(os.environ, LC_ALL='C'))
    rows = docs = 0; hist = collections.defaultdict(lambda: np.zeros(len(BINS) - 1, dtype=np.int64))
    tok_hist = collections.defaultdict(lambda: np.zeros(len(BINS) - 1, dtype=np.float64)); started = time.monotonic()
    try:
        for rp, r in receipts:
            a = r['artifact']
            if Path(a['path']).stat().st_size != a['bytes'] or sha(a['path']) != a['sha256']:
                raise ValueError('Evidence artifact changed')
            sid = r['identity']['source_id']; src = r['source']
            cols = ['ordinal', 'source_text_tokens', 'signature', 'bands', 'dclm_hq_prob']
            for batch in pq.ParquetFile(a['path']).iter_batches(batch_size=4096, columns=cols):
                b = batch.to_pydict()
                for o, n, sig, bands, p in zip(b['ordinal'], b['source_text_tokens'], b['signature'], b['bands'], b['dclm_hq_prob']):
                    docs += 1
                    if p is not None:
                        k = min(int(p * 1000), 999); hist[src][k] += 1; tok_hist[src][k] += n
                    if sig is None:
                        continue
                    if len(sig) != 896 or len(bands) != 14:
                        raise ValueError('Invalid signature shape')
                    for band, h in enumerate(bands):
                        sort.stdin.write(f"{band:02d}:{h.hex()}\t{sid}\t{o}\t{n}\n".encode()); rows += 1
            if shutil.disk_usage(ROOT).free < 512 * 2**30:
                raise RuntimeError('HDD reserve reached')
            atomic(out / 'progress.json', dict(status='BUILDING', documents=docs, index_rows=rows, training_admitted=False))
        sort.stdin.close(); building = out / 'candidate_members.building.tsv.gz'
        with gzip.open(building, 'wb', compresslevel=1) as f:
            stats = groups(sort.stdout, f)
        if sort.wait() != 0 or stats['index_rows'] != rows or docs != plan['target']['documents']:
            raise ValueError('Global bucket sort or coverage failed')
        final = out / 'candidate_members.tsv.gz'; os.replace(building, final)
    except BaseException:
        if sort.poll() is None:
            sort.kill(); sort.wait()
        raise
    dclm = {}
    for src in hist:
        c = np.cumsum(tok_hist[src][::-1])[::-1] / tok_hist[src].sum()
        dclm[src] = dict(documents=int(hist[src].sum()), doc_hist=hist[src].tolist(),
                         token_share_at_or_above_bin=c.round(6).tolist())
    (out / 'dclm_histograms.json').write_text(json.dumps(dclm))
    report = dict(status='GLOBAL_FUSED_EVIDENCE_NOT_VERIFIED_DUPLICATES_NOT_SEMANTIC_CLEARANCE', counts=dict(total),
                  candidate_stats=dict(stats), candidate_members=dict(path=str(final), bytes=final.stat().st_size, sha256=sha(final)),
                  dclm_histograms=dict(path=str(out / 'dclm_histograms.json'), sha256=sha(out / 'dclm_histograms.json')),
                  source_receipts=[dict(path=p, sha256=sha(p)) for p, _ in receipts], plan_sha256=ph,
                  training_admitted=False, automatic_deletion=False, seconds=time.monotonic() - started)
    atomic(out / 'COMPLETE.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source_receipts'}), flush=True)


if __name__ == '__main__':
    main()

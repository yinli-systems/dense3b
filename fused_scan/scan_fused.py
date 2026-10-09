"""Fused, task-parallel evidence pass: one decode per document yields the v1 near signature, the v1
benchmark evidence (unmodified v1 code, imported by path and pinned by sha) and a DCLM fastText score.
Evidence only: never deletes, never admits training data."""
import argparse, collections, fcntl, hashlib, json, os, shutil, sys, time, uuid
from pathlib import Path

V1 = Path('/data/run01/scxi253/moe100b-parallel-quality-20261002T1701Z')
sys.path.insert(0, str(V1 / 'source')); sys.path.insert(0, str(V1 / 'deps'))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import quality_core as QC
from quality_core import TOK, sha, atomic, text_from_tokens, sketch, bands
from matching import Matcher

ROOT = Path(__file__).resolve().parent.parent
OVERSIZE = 262144  # same cut-off as v1 scan_selected.py
SCHEMA = pa.schema([('ordinal', pa.uint64()), ('source_text_tokens', pa.uint32()), ('text_hash', pa.binary(32)),
    ('metadata_status', pa.uint8()), ('metadata_flags', pa.uint16()), ('status', pa.string()),
    ('near_status', pa.string()), ('signature', pa.binary(896)), ('bands', pa.list_(pa.binary(16))),
    ('units', pa.uint32()), ('shingles', pa.uint32()),
    ('benchmark_status', pa.string()), ('evidence_json', pa.string()), ('evidence_truncated', pa.bool_()),
    ('dclm_hq_prob', pa.float32())])
G = {}


def load_shared(fasttext_path):
    import fasttext
    G['decoder'] = Tokenizer.from_file(str(TOK))
    G['matcher'] = Matcher()
    G['ft'] = fasttext.load_model(fasttext_path)
    labels = G['ft'].get_labels()
    if '__label__hq' not in labels:
        raise RuntimeError('Unexpected fastText labels: %r' % labels)


def dclm_score(text):
    labels, probs = G['ft'].predict(text.replace('\n', ' '), k=-1)
    return float(dict(zip(labels, probs)).get('__label__hq', 0.0))


def scan_task(job):
    task, identity, out_root = job
    sid = task['source_id']; folder = Path(out_root) / sid / ('%09d-%09d' % (task['start'], task['stop']))
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'RUN.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = folder / 'COMPLETE.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['identity'] != identity or sha(old['artifact']['path']) != old['artifact']['sha256']:
                raise ValueError('Resume evidence changed: ' + str(receipt))
            return str(receipt), old['counts'], True
        m = task['mask']; raw = Path(m['mask_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != m['mask_sha256']:
            raise ValueError('Mask changed')
        keep = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little')[:m['documents']]
        if int(keep.sum()) != m['selected_documents']:
            raise ValueError('Mask population mismatch')
        spec = task['metadata']
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Metadata changed')
        payload = Path(m['token_artifact']['path']); before = payload.stat()
        if before.st_size != m['token_artifact']['bytes']:
            raise ValueError('Token payload resized')
        attempt = folder / ('attempt-' + uuid.uuid4().hex); attempt.mkdir(); out = attempt / 'evidence.parquet'
        counts = collections.Counter(); buf = []; scanned = 0; pf = pq.ParquetFile(spec['path']); base = 0
        cols = ['ordinal', 'offset', 'tokens', 'text_hash', 'token_hash', 'status', 'flags']
        with payload.open('rb', buffering=8 << 20) as stream, pq.ParquetWriter(out, SCHEMA, compression='zstd') as w:
            for rg in range(pf.num_row_groups):
                lo = base; base += pf.metadata.row_group(rg).num_rows
                if base <= task['start'] or lo >= task['stop']:
                    continue
                for batch in pf.iter_batches(row_groups=[rg], batch_size=4096, columns=cols, use_threads=False):
                    for row in batch.to_pylist():
                        o = row['ordinal']
                        if not task['start'] <= o < task['stop']:
                            continue
                        if o != task['start'] + scanned:
                            raise ValueError('Address gap or duplication')
                        scanned += 1
                        if not keep[o]:
                            continue
                        rec = dict(ordinal=o, source_text_tokens=row['tokens'], text_hash=row['text_hash'],
                                   metadata_status=row['status'], metadata_flags=row['flags'])
                        text = None if row['tokens'] > OVERSIZE else text_from_tokens(stream, row, G['decoder'])
                        if text is None:
                            rec.update(status='UNSCANNED_OVERSIZE_OR_RAW_RECONSTRUCTION_REVIEW',
                                       near_status=None, benchmark_status=None)
                        else:
                            sig, n, sh = sketch(text)
                            near = 'NEAR_SIGNATURE_ONLY' if sig else 'SHORT_DOCUMENT_NO_NEAR_SIGNATURE'
                            match = G['matcher'].scan(text)
                            rec.update(status='SCANNED', near_status=near, signature=sig, bands=bands(sig),
                                       units=n, shingles=sh, benchmark_status=match['status'],
                                       evidence_json=json.dumps(match), evidence_truncated=match['evidence_truncated'],
                                       dclm_hq_prob=dclm_score(text))
                            counts['near/' + near + '_documents'] += 1
                            counts['benchmark/' + match['status'] + '_documents'] += 1
                            counts['benchmark/' + match['status'] + '_tokens'] += row['tokens']
                        counts['documents'] += 1; counts['text_tokens'] += row['tokens']
                        counts[rec['status'] + '_documents'] += 1; counts[rec['status'] + '_tokens'] += row['tokens']
                        if row['status'] != 0:
                            counts['metadata_review_documents'] += 1
                        buf.append(rec)
                        if len(buf) >= 512:
                            w.write_table(pa.Table.from_pylist(buf, schema=SCHEMA)); buf = []
            if buf:
                w.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
        if scanned != task['stop'] - task['start'] or counts['documents'] != task['documents']:
            raise ValueError('Assigned selected coverage incomplete')
        after = payload.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed')
        with out.open('rb') as f:
            os.fsync(f.fileno())
        result = dict(identity=identity, status='ALL_CHUNK_ADDRESSES_PROCESSED_NOT_ADMITTED', source=task['source'],
                      counts=dict(counts), artifact=dict(path=str(out), bytes=out.stat().st_size, sha256=sha(out)),
                      training_admitted=False, near_duplicate_confirmed=False, benchmark_cleared=False)
        atomic(receipt, result)
        return str(receipt), dict(counts), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--workers', type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument('--canary-tasks', type=int, default=0)
    a = ap.parse_args()
    cfg = json.loads((ROOT / 'config/fused.json').read_text())
    plan_path = ROOT / 'config/plan.json'; plan = json.loads(plan_path.read_text()); ph = sha(plan_path)
    if ph != sha(V1 / 'config/plan.json'):
        raise ValueError('Plan must be byte-identical to the v1 parallel-quality plan')
    for x in plan['pins']:
        if sha(x['path']) != x['sha256']:
            raise ValueError('Frozen input changed: ' + x['path'])
    for name, digest in cfg['pinned_code'].items():
        if sha(V1 / 'source' / name) != digest:
            raise ValueError('v1 code changed: ' + name)
    if sha(cfg['fasttext_path']) != cfg['fasttext_sha256']:
        raise ValueError('fastText model changed')
    part = plan['parts'][a.part]
    identity = dict(plan_sha256=ph, mode='fused_near_benchmark_dclm', implementation_sha256=sha(__file__),
                    v1_code=cfg['pinned_code'], fasttext_sha256=cfg['fasttext_sha256'],
                    canary_documents=0 if not a.canary_tasks else a.canary_tasks)
    out_root = ROOT / ('canary' if a.canary_tasks else 'output')
    progress = ROOT / 'reports' / ('part-%d%s.json' % (a.part, '-canary' if a.canary_tasks else ''))
    with (ROOT / 'reports' / ('part-%d.lock' % a.part)).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        load_shared(cfg['fasttext_path'])  # loaded once, shared copy-on-write by forked workers
        tasks = part['tasks'][:a.canary_tasks] if a.canary_tasks else part['tasks']
        jobs = [(t, dict(identity, source_id=t['source_id'], start=t['start'], stop=t['stop']), str(out_root)) for t in tasks]
        totals = collections.Counter(); receipts = []; resumed = 0; started = time.monotonic(); last = 0
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            for rp, counts, was_resumed in pool.imap_unordered(scan_task, jobs):
                if shutil.disk_usage(ROOT).free < 512 * 2**30:
                    raise RuntimeError('HDD reserve reached')
                receipts.append(rp); totals.update(counts); resumed += was_resumed
                if time.monotonic() - last > 15 or len(receipts) == len(jobs):
                    el = time.monotonic() - started
                    atomic(progress, dict(status='PROCESSING', part=a.part, job_id=os.environ.get('SLURM_JOB_ID'),
                        completed_tasks=len(receipts), total_tasks=len(jobs), resumed_tasks=resumed,
                        committed=dict(totals), seconds=el, docs_per_s=totals['documents'] / max(1e-9, el),
                        workers=a.workers, target_documents=part['documents'], target_text_tokens=part['text_tokens'],
                        training_admitted=False, automatic_deletion=False)); last = time.monotonic()
    if not a.canary_tasks and (totals['documents'] != part['documents'] or totals['text_tokens'] != part['text_tokens']):
        raise ValueError('Part totals differ from frozen plan')
    final = dict(status='CANARY_COMPLETE' if a.canary_tasks else 'PART_COMPLETE_NOT_FINAL_ADMISSION', part=a.part,
                 identity=identity, committed=dict(totals), receipts=sorted(receipts),
                 seconds=time.monotonic() - started, training_admitted=False, automatic_deletion=False)
    atomic(progress.with_name(progress.stem + '.COMPLETE.json'), final)
    print(json.dumps({k: v for k, v in final.items() if k != 'receipts'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        import traceback
        print(json.dumps(dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc())), flush=True)
        raise

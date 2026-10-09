"""v2 of score_all.py: identical model, windows, schema and receipts; CPU work spread over worker
processes so the GPU is fed continuously. Reuses hash-verified v1 receipts. Scores only, never deletes."""
import argparse, collections, contextlib, fcntl, hashlib, json, os, shutil, sys, time, uuid
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_core import sha, atomic, windows_for_length, score_scope, numeric_gate, check_mask, verified_text

V1 = Path('/data/run01/scxi253/moe100b-neural-score-20261002T1550Z')
UNIT_ROWS = 2048
COLS = ['ordinal', 'offset', 'tokens', 'text_hash', 'token_hash', 'status', 'flags']

# ---------------------------------------------------------------- worker side
W = {}

def winit(model_path, decoder_path):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, AutoConfig
    W['decoder'] = Tokenizer.from_file(decoder_path)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    limit = min(512, int(cfg.max_position_embeddings), int(tok.model_max_length))
    from chunk_encode import ChunkEncoder
    W['tok'] = tok; W['width'] = limit - tok.num_special_tokens_to_add(pair=False)
    W['enc'] = ChunkEncoder(tok._tokenizer)
    W['pq'] = {}; W['streams'] = {}


def _rowgroup(meta_path, rg):
    import pyarrow.parquet as pq
    key = (meta_path, rg)
    if W.get('rg_key') != key:
        pf = W['pq'].get(meta_path)
        if pf is None:
            pf = W['pq'][meta_path] = pq.ParquetFile(meta_path)
        W['rg_rows'] = pf.read_row_group(rg, columns=COLS, use_threads=False).to_pylist()
        W['rg_key'] = key
    return W['rg_rows']


def work(unit):
    """unit = (meta_path, rg, row_lo, row_hi, rg_base, task_start, task_stop, keep_bytes, keep_lo, token_path)."""
    import numpy as np, time
    T = collections.Counter(); t_unit = time.perf_counter()
    meta_path, rg, r_lo, r_hi, rg_base, t_start, t_stop, keep_bytes, keep_lo, token_path = unit
    keep = np.frombuffer(keep_bytes, dtype=np.uint8)
    rows = _rowgroup(meta_path, rg)[r_lo:r_hi]
    stream = W['streams'].get(token_path)
    if stream is None:
        stream = W['streams'][token_path] = open(token_path, 'rb', buffering=8 << 20)
    tok, width, decoder, enc = W['tok'], W['width'], W['decoder'], W['enc']
    spans = lambda n: windows_for_length(n, width)[0]
    first = None; scanned = 0; docs = []
    for row in rows:
        o = row['ordinal']
        if not t_start <= o < t_stop:
            continue
        if first is None:
            first = o
        if o != first + scanned:
            raise RuntimeError('Metadata address gap or duplication')
        scanned += 1
        if not keep[o - keep_lo]:
            continue
        t0 = time.perf_counter()
        stream.seek(row['offset'] * 2); raw = stream.read(row['tokens'] * 2); eos = stream.read(2)
        t1 = time.perf_counter(); text = verified_text(raw, eos, row, decoder); t2 = time.perf_counter()
        T['read_s'] += t1 - t0; T['verify_decode_s'] += t2 - t1
        rec = dict(ordinal=o, source_text_tokens=row['tokens'], text_hash=row['text_hash'],
                   metadata_status=row['status'], metadata_flags=row['flags'])
        if text is None:
            rec['status'] = 'HOLD_RAW_RECONSTRUCTION_REQUIRED'
            docs.append((rec, None, None)); continue
        n_ids, intervals, pieces = enc.length_and_slices(text, spans)
        covered = windows_for_length(n_ids, width)[1]
        local = {}; wins = []; idx = []
        for (lo, hi), piece in zip(intervals, pieces):
            if (lo, hi) not in local:
                local[(lo, hi)] = len(wins)
                wins.append(np.asarray(tok.build_inputs_with_special_tokens(piece), dtype=np.int32))
            idx.append(local[(lo, hi)])
        T['encode_s'] += time.perf_counter() - t2; T['bert_tokens'] += n_ids
        rec.update(status='SCORED_NOT_FACT_CHECKED', classifier_tokens=n_ids, covered_classifier_tokens=covered,
                   head_start=intervals[0][0], middle_start=intervals[1][0], tail_start=intervals[2][0])
        docs.append((rec, wins, idx))
    T['unit_s'] = time.perf_counter() - t_unit; T['pid'] = os.getpid()
    T['affinity'] = len(os.sched_getaffinity(0))
    return first, scanned, docs, dict(T)


# ---------------------------------------------------------------- main side
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--part', type=int, required=True)
    ap.add_argument('--workers', type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    ap.add_argument('--batch-windows', type=int, default=256)
    ap.add_argument('--canary-task', default='', help='source_id:start — rescore this task into canary/ only')
    ap.add_argument('--canary-docs', type=int, default=0)
    a = ap.parse_args(); root = a.root.resolve()
    plan_path = root / 'config/plan.json'; plan = json.loads(plan_path.read_text())
    if not 0 <= a.part < len(plan['parts']):
        raise ValueError('Invalid task range')
    for p in plan['pinned_files']:
        if sha(p['path']) != p['sha256']:
            raise ValueError('Pinned file changed: ' + p['path'])
    plan_sha = sha(plan_path)
    if plan_sha != sha(V1 / 'config/plan.json'):
        raise ValueError('v2 plan must be byte-identical to v1 plan')
    lockname = 'canary-%03d.lock' % a.part if a.canary_task else 'part-%03d.lock' % a.part
    with (root / 'reports' / lockname).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, a, plan, plan_sha)


def run(root, a, plan, plan_sha):
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA device unavailable')
    part = plan['parts'][a.part]
    tok = AutoTokenizer.from_pretrained(plan['model_path'], local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForSequenceClassification.from_pretrained(plan['model_path'], local_files_only=True,
        trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise RuntimeError('Model assets did not load exactly')
    model = model.float().cuda().eval()
    limit = min(512, int(model.config.max_position_embeddings), int(tok.model_max_length))
    pad_id = tok.pad_token_id
    schema = pa.schema([('ordinal', pa.uint64()), ('source_text_tokens', pa.uint32()), ('text_hash', pa.binary(32)),
        ('metadata_status', pa.uint8()), ('metadata_flags', pa.uint16()), ('status', pa.string()),
        ('classifier_tokens', pa.uint32()), ('covered_classifier_tokens', pa.uint32()),
        ('head_start', pa.uint32()), ('middle_start', pa.uint32()), ('tail_start', pa.uint32()),
        ('head_score', pa.float32()), ('middle_score', pa.float32()), ('tail_score', pa.float32()),
        ('mean_score', pa.float32()), ('min_score', pa.float32()), ('score_range', pa.float32())])
    worker_id = os.environ.get('SLURM_JOB_ID', 'manual')
    st = dict(precision=None, calibration=None, limit=limit)

    def forward(wins, use_bf16):
        """wins: list of int32 arrays. Length-sorted batches; returns float list in input order."""
        out = np.empty(len(wins), dtype=np.float64)
        order = sorted(range(len(wins)), key=lambda i: len(wins[i]))
        with torch.inference_mode():
            for s in range(0, len(order), a.batch_windows):
                sel = order[s:s + a.batch_windows]; L = max(len(wins[i]) for i in sel)
                ids = np.full((len(sel), L), pad_id, dtype=np.int64); att = np.zeros((len(sel), L), dtype=np.int64)
                for r, i in enumerate(sel):
                    n = len(wins[i]); ids[r, :n] = wins[i]; att[r, :n] = 1
                batch = dict(input_ids=torch.from_numpy(ids).cuda(non_blocking=True),
                             attention_mask=torch.from_numpy(att).cuda(non_blocking=True))
                ctx = torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
                with ctx:
                    logits = model(**batch).logits
                if logits.ndim != 2 or logits.shape[1] != 1 or not bool(torch.isfinite(logits).all()):
                    raise RuntimeError('Invalid regression output')
                out[sel] = logits[:, 0].float().cpu().numpy()
        return out.tolist()

    def score_docs(docs, counts):
        wins = []; links = []
        for rec, w, idx in docs:
            if w is None:
                counts['held_documents'] += 1; counts['held_source_text_tokens'] += rec['source_text_tokens']
                links.append(None); continue
            base = len(wins); wins.extend(w); links.append([base + i for i in idx])
            counts['classifier_tokens'] += rec['classifier_tokens']
            counts['covered_classifier_tokens'] += rec['covered_classifier_tokens']
        if wins:
            if st['precision'] is None:
                sample = wins[:min(32, len(wins))]
                ref = forward(sample, False)
                if torch.cuda.is_bf16_supported():
                    st['calibration'] = numeric_gate(ref, forward(sample, True))
                    st['precision'] = 'bf16' if st['calibration']['passed'] else 'fp32'
                else:
                    st['precision'] = 'fp32'; st['calibration'] = {'passed': False, 'reason': 'bf16_not_supported'}
                atomic(root / 'reports' / ('numeric-%s-part%03d.json' % (worker_id, a.part)), st['calibration'])
            values = forward(wins, st['precision'] == 'bf16')
            counts['neural_windows'] += len(wins)
        out = []
        for (rec, w, idx), ln in zip(docs, links):
            if ln is not None:
                sc = [values[i] for i in ln]
                rec.update(head_score=sc[0], middle_score=sc[1], tail_score=sc[2],
                           mean_score=sum(sc) / 3, min_score=min(sc), score_range=max(sc) - min(sc))
                counts['scored_documents'] += 1; counts['scored_source_text_tokens'] += rec['source_text_tokens']
            counts['processed_documents'] += 1; counts['processed_source_text_tokens'] += rec['source_text_tokens']
            out.append(rec)
        return out

    total = collections.Counter(); committed = []; reused = []; started = time.monotonic()
    canary = None
    if a.canary_task:
        sid, start = a.canary_task.split(':'); canary = (sid, int(start))
    progress = root / 'reports' / (('canary-%03d.json' if canary else 'part-%03d.progress.json') % a.part)

    def report(state, sid=None, counts=None):
        return dict(status=state, job_id=worker_id, part=a.part, selection_sha256=plan['selection_sha256'],
            plan_sha256=plan_sha, committed_counts=dict(total), current_counts=dict(counts or {}),
            current_source=sid, completed_chunks=len(committed), reused_v1_chunks=len(reused),
            seconds=time.monotonic() - started, precision=st['precision'], calibration=st['calibration'],
            gpu=torch.cuda.get_device_name(), workers=a.workers, implementation='score_fast_v2',
            intended_documents=part['documents'], intended_source_text_tokens=part['text_tokens'],
            training_admitted=False, full_long_document_coverage=False)

    atomic(progress, report('MODEL_LOADED_FULL_PART_PENDING'))
    ctx = mp.get_context('forkserver')
    pool = ctx.Pool(a.workers, initializer=winit, initargs=(plan['model_path'], plan['decoder_path']))
    try:
        for task in part['tasks']:
            sid = task['source_id']; rng = '%09d-%09d' % (task['start'], task['stop'])
            if canary and (sid, task['start']) != canary:
                continue
            identity = dict(plan_sha256=plan_sha, source_id=sid, start=task['start'], stop=task['stop'])
            if not canary:
                prior = None
                for base in (root, V1):
                    r = base / 'scores' / sid / rng / 'COMPLETE.json'
                    if r.exists():
                        old = json.loads(r.read_text())
                        if old['identity'] != identity or sha(old['artifact']['path']) != old['artifact']['sha256']:
                            raise RuntimeError('Completed score identity or payload changed: ' + str(r))
                        prior = r; break
                if prior is not None:
                    total.update(old['counts']); committed.append(str(prior))
                    if base == V1:
                        reused.append(str(prior))
                    continue
                if shutil.disk_usage(root).free < 512 * 2**30:
                    raise RuntimeError('HDD reserve reached')
            folder = root / ('canary' if canary else 'scores') / sid / rng
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'RUN.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = score_task(task, identity, folder, pool, score_docs, schema, report, progress, canary, a,
                                    np, pa, pq, plan, st, sid)
            if canary:
                atomic(progress, report('CANARY_DONE', sid, result['counts'])); return
            atomic(folder / 'COMPLETE.json', result); total.update(result['counts'])
            committed.append(str(folder / 'COMPLETE.json'))
            atomic(progress, report('CHUNK_COMMITTED', sid))
            print(json.dumps(dict(source_id=sid, counts=result['counts'])), flush=True)
    finally:
        pool.terminate()
    if total['processed_documents'] != part['documents'] or total['processed_source_text_tokens'] != part['text_tokens']:
        raise RuntimeError('Full partition totals do not match frozen selection')
    result = report('ALL_PART_RECORDS_PROCESSED_NOT_FINAL_ADMISSION'); result['receipts'] = committed
    result['reused_v1_receipts'] = reused
    atomic(root / 'reports' / ('part-%03d.COMPLETE.json' % a.part), result)
    print(json.dumps(report('TASK_EXIT')), flush=True)


def score_task(task, identity, folder, pool, score_docs, schema, report, progress, canary, a, np, pa, pq, plan, st, sid):
    mask = task['mask']; rawmask = Path(mask['mask_path']).read_bytes()
    if hashlib.sha256(rawmask).hexdigest() != mask['mask_sha256']:
        raise RuntimeError('Mask changed')
    check_mask(rawmask, mask['documents'], mask['selected_documents'])
    keep = np.unpackbits(np.frombuffer(rawmask, dtype=np.uint8), bitorder='little')[:mask['documents']]
    meta = task['metadata']
    if sha(meta['path']) != meta['sha256']:
        raise RuntimeError('Metadata payload changed')
    token = mask['token_artifact']
    if Path(token['path']).stat().st_size != token['bytes']:
        raise RuntimeError('Token payload resized')
    pf = pq.ParquetFile(meta['path']); units = []; base = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows; lo, hi = base, base + n; base = hi
        if hi <= task['start'] or lo >= task['stop']:
            continue
        for r in range(0, n, UNIT_ROWS):
            o_lo, o_hi = max(lo + r, task['start']), min(lo + r + UNIT_ROWS, hi, task['stop'])
            if o_lo >= o_hi:
                continue
            units.append((meta['path'], rg, r, min(r + UNIT_ROWS, n), lo, task['start'], task['stop'],
                          keep[o_lo:o_hi].tobytes(), o_lo, token['path']))
    attempt = folder / ('attempt-' + uuid.uuid4().hex); attempt.mkdir()
    target = attempt / 'scores.parquet'; counts = collections.Counter(); scanned = 0; last = 0.0
    t0 = time.monotonic(); cpu_wait = 0.0; gpu_time = 0.0; wtime = collections.Counter(); pids = set(); aff = None
    with pq.ParquetWriter(target, schema, compression='zstd') as writer:
        inflight = collections.deque(); it = iter(units); depth = 3 * a.workers
        for u in it:
            inflight.append(pool.apply_async(work, (u,)))
            if len(inflight) >= depth:
                break
        stop = False
        while inflight and not stop:
            w0 = time.monotonic(); first, n, docs, wt = inflight.popleft().get(); cpu_wait += time.monotonic() - w0
            pids.add(wt.pop('pid')); aff = wt.pop('affinity'); wtime.update(wt)
            nxt = next(it, None)
            if nxt is not None:
                inflight.append(pool.apply_async(work, (nxt,)))
            if n:
                if first != task['start'] + scanned:
                    raise RuntimeError('Metadata address gap or duplication')
                scanned += n
            if canary and a.canary_docs and counts['processed_documents'] + len(docs) > a.canary_docs:
                docs = docs[:a.canary_docs - counts['processed_documents']]; stop = True
            g0 = time.monotonic(); rows = score_docs(docs, counts); gpu_time += time.monotonic() - g0
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            if time.monotonic() - last >= 10:
                rep = report('SCORING_FULL_SELECTION', sid, counts)
                rep['throughput'] = dict(windows_per_s=counts['neural_windows'] / max(1e-9, time.monotonic() - t0),
                                         main_wait_for_cpu_s=cpu_wait, main_gpu_s=gpu_time,
                                         worker_phase_s=dict(wtime), worker_pids=len(pids), worker_affinity=aff)
                atomic(progress, rep); last = time.monotonic()
    elapsed = time.monotonic() - t0
    perf = dict(seconds=elapsed, windows_per_s=counts['neural_windows'] / max(1e-9, elapsed),
                docs_per_s=counts['processed_documents'] / max(1e-9, elapsed),
                main_wait_for_cpu_s=cpu_wait, main_gpu_s=gpu_time, workers=a.workers,
                worker_phase_s=dict(wtime), worker_pids=len(pids), worker_affinity=aff)
    if not canary:
        if scanned != task['stop'] - task['start'] or counts['processed_documents'] != task['documents']:
            raise RuntimeError('Assigned document coverage mismatch')
    if counts['scored_documents'] + counts['held_documents'] != counts['processed_documents']:
        raise RuntimeError('Score/hold accounting mismatch')
    with target.open('rb') as f:
        os.fsync(f.fileno())
    return dict(identity=identity, status='CANARY_ONLY' if canary else 'ALL_CHUNK_RECORDS_PROCESSED_NOT_ADMITTED',
        counts=dict(counts), artifact=dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target)),
        precision=st['precision'], calibration=st['calibration'], model_sha256=plan['model_sha256'],
        model_repository=plan['model_repository'], model_revision=plan['model_revision'],
        classifier_context_tokens=st['limit'], source=task['source'], score_scope=score_scope(task['source']),
        full_long_document_coverage=False, benchmark_cleared=False, near_deduplicated=False,
        training_admitted=False, implementation='score_fast_v2', performance=perf)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        import traceback
        print(json.dumps(dict(status='SCORING_FAILED', error=repr(exc), traceback=traceback.format_exc())), flush=True)
        raise

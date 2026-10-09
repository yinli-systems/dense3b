"""3B code-model data, v0: filter -> benchmark decontamination -> global exact + near dedup -> uint16 shards.

Stages (each idempotent, every output written atomically next to a JSON receipt):

  bench   benchmark files -> sorted uint64 array of word 13-gram hashes (code + math benchmarks), minus
          n-grams that are common in a background sample of ordinary code (generic idioms, not evidence of
          contamination)
  map     one raw input file -> work/map/<id>.parquet: per surviving document its ordinal, exact key,
          14 MinHash band keys (same sketch as the web pipeline: quality_core.sketch/bands), source rank,
          quality score and length. Filter and decontamination drops are only counted.
  reduce  all map outputs -> work/keep/<id>.npy (ordinals to keep). Exact duplicates and LSH band
          collisions are joined with union-find; each component keeps its best document
          (lowest source rank, then highest score, then longest).
  pack    raw input file + keep list -> shards/<source>/<id>.u16 (documents + EOS, train/data.py format)

The filter is a pure function of (row, config), so pack re-derives exactly the documents map saw.
"""
import argparse, collections, glob, gzip, hashlib, json, os, re, sys, time, uuid, zlib
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import xxhash

EOS = 50279
FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX = 50280, 50281, 50282   # unused ids (vocab 50280, embedding padded to 50304)
NGRAM = 13
P64 = np.uint64(0x9E3779B97F4A7C15)
WORD = re.compile(r'[a-z0-9]+')
QC_PATH = '/data/run01/scxi253/moe100b-parallel-quality-20261002T1701Z/source'


# ------------------------------------------------------------------ shared helpers

def atomic_json(path, obj):
    path = Path(path); tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(obj, f, sort_keys=True, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write(path, write_fn):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    write_fn(tmp); os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ngram_hashes(text):
    """Order-sensitive 64-bit hashes of every 13-word window (words = lowercase [a-z0-9]+ runs)."""
    words = WORD.findall(text.lower())
    if len(words) < NGRAM:
        return np.empty(0, np.uint64)
    w = np.fromiter((xxhash.xxh64_intdigest(x) for x in words), np.uint64, len(words))
    n = len(words) - NGRAM + 1
    h = np.zeros(n, np.uint64)
    with np.errstate(over='ignore'):
        for j in range(NGRAM):
            h = h * P64 + w[j:j + n]
    return h


def strong_windows(text, min_distinct=8):
    """Mask over ngram_hashes(text): windows with >= min_distinct distinct words that contain a letter and have
    >= 3 characters. Pure syntax ('for int i 0 i n i ...'), numbers and short identifiers never qualify, so only
    problem-statement-like text can count as benchmark evidence."""
    words = WORD.findall(text.lower())
    n = len(words) - NGRAM + 1
    if n <= 0:
        return np.zeros(0, bool)
    strong = [w if len(w) >= 3 and not w.isdigit() and any(c.isalpha() for c in w) else None for w in words]
    out = np.zeros(n, bool)
    for i in range(n):
        out[i] = len({w for w in strong[i:i + NGRAM] if w}) >= min_distinct
    return out


def in_sorted(h, sorted_arr):
    """Boolean mask: which of h occur in sorted_arr (binary search; np.isin would re-sort sorted_arr per call)."""
    if h.size == 0 or sorted_arr.size == 0:
        return np.zeros(h.size, bool)
    i = np.minimum(np.searchsorted(sorted_arr, h), sorted_arr.size - 1)
    return sorted_arr[i] == h


def read_rows(path, fmt, columns=None, start=0, end=None):
    """Yield raw rows (dicts) of one input in a fixed order. Plain .jsonl inputs may be a byte range
    [start, end): a line belongs to the range that contains its first byte."""
    if fmt == 'parquet':
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            yield from pf.read_row_group(i, columns=columns).to_pylist()
    elif path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, 'rb') as f:
            if start > 0:
                f.seek(start - 1); f.readline()
            while end is None or f.tell() < end:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    yield json.loads(line)


CHUNK = 512 << 20


def inputs(cfg):
    """[(id, source, path, start, end)] for every input of every mapped source, in a fixed order. Plain
    .jsonl files larger than CHUNK are split into byte ranges so no single task runs for hours."""
    out = []
    for s in cfg['sources']:
        if not s.get('map', True):
            continue
        for p in sorted(glob.glob(os.path.join(cfg['raw_root'], s['glob']), recursive=True)):
            if '/.cache/' in p or p.endswith(('.sha256ok', '.done', '.part')):
                continue
            size = os.path.getsize(p)
            ranges = [(0, None)]
            if s['format'] == 'jsonl' and not p.endswith('.gz') and size > CHUNK:
                ranges = [(b, min(b + CHUNK, size)) for b in range(0, size, CHUNK)]
            for b, e in ranges:
                key = p if (b, e) == (0, None) else f'{p}@{b}'
                out.append((s['name'] + '__' + hashlib.sha1(key.encode()).hexdigest()[:16], s['name'], p, b, e))
    return out


# ------------------------------------------------------------------ document filter (pure)

def alnum_frac(text):
    return sum(c.isalnum() for c in text) / max(1, len(text))


def filter_doc(row, src):
    """-> (text, score) for a kept document, or (None, reason)."""
    text = row.get(src.get('text_field', 'text'))
    if not isinstance(text, str):
        return None, 'no_text'
    if len(text) < src.get('min_chars', 64):
        return None, 'too_short'
    if len(text) > src.get('max_chars', 400_000):
        return None, 'too_long'
    if alnum_frac(text) < 0.25:
        return None, 'low_alnum'
    score = 0.0
    if src['name'] == 'stackv2-edu':
        meta = row.get('metadata') or {}
        if isinstance(meta, str):
            import ast
            meta = ast.literal_eval(meta)
        if meta.get('is_generated') or meta.get('is_vendor'):
            return None, 'generated_or_vendor'
        lang = (meta.get('language') or '').lower()
        score = float(row.get('score') or 0.0)
        floor = src['score_floor_by_lang'].get(lang, src['score_floor'])
        if score < floor:
            return None, 'edu_score_below_floor'
        if lines_too_long(text):
            return None, 'long_lines'
    return text, score


def lines_too_long(text):
    lines = text.split('\n')
    return max(map(len, lines)) > 1000 or len(text) / max(1, len(lines)) > 200


# ------------------------------------------------------------------ bench

BENCH_SKIP_FIELDS = {'private_test_cases', 'public_test_cases', 'input_output', 'private_tests', 'public_tests',
                     'generated_tests', 'incorrect_solutions', 'test_list_2', 'metadata', 'task_id', 'id', 'url'}


BOILER_LINE = re.compile(r'^\s*(import\s|from\s+\S+\s+import\s|#\s*include|using\s|package\s|require\(|const\s+\w+\s*=\s*require\(|extern crate|use\s+[\w:]+;)')


def strip_boiler_lines(text):
    """Drop import/include lines: shared by most code in a library's ecosystem, they identify no problem."""
    return '\n'.join(l for l in text.split('\n') if not BOILER_LINE.match(l))


def bench_texts(path):
    """Every string field (>= 40 chars) of every row of one benchmark file; very large fields are test blobs."""
    fmt = 'parquet' if path.endswith('.parquet') else 'jsonl'
    if path.endswith('.csv'):
        import csv
        rows = csv.DictReader(open(path, encoding='utf-8'))
    else:
        rows = read_rows(path, fmt)
    for r in rows:
        for k, v in r.items():
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if k not in BENCH_SKIP_FIELDS and isinstance(x, str) and 40 <= len(x) <= 100_000:
                    yield strip_boiler_lines(x)


def cmd_bench(cfg, a):
    files = sorted(p for p in glob.glob(os.path.join(cfg['bench_root'], 'raw', '**', '*'), recursive=True)
                   if p.endswith(('.parquet', '.jsonl', '.jsonl.gz', '.csv')) and os.path.isfile(p))
    per_repo = collections.Counter(); hashes = []; contexts = collections.defaultdict(set)
    ctx_w = cfg.get('bench_context_words', 8); max_ctx = cfg.get('bench_max_contexts', 4)
    for p in files:
        repo = Path(p).relative_to(Path(cfg['bench_root']) / 'raw').parts[0]
        n = 0
        for t in bench_texts(p):
            h = ngram_hashes(t); m = strong_windows(t)
            words = WORD.findall(t.lower())
            for i in np.flatnonzero(m).tolist():
                c = contexts[int(h[i])]
                if len(c) < max_ctx:
                    c.add(xxhash.xxh64_intdigest(' '.join(words[max(0, i - ctx_w):i]) + '|' +
                                                 ' '.join(words[i + NGRAM:i + NGRAM + ctx_w])))
            hashes.append(h[m]); n += 1
        per_repo[repo] += n
    # an n-gram seen in >= max_ctx different surroundings is boilerplate shared by many problems (logging setup,
    # ListNode definitions, "return it modulo 10^9+7"); variants of one problem (HumanEval / HumanEval+ /
    # MultiPL-E / ...) share their surroundings and stay
    shared = np.array(sorted(k for k, c in contexts.items() if len(c) >= max_ctx), np.uint64)
    allh = np.unique(np.concatenate(hashes)) if hashes else np.empty(0, np.uint64)
    # background: n-grams frequent in ordinary code are idioms, not contamination evidence
    bg = collections.Counter()
    srcs = {s['name']: s for s in cfg['sources']}
    bg_files = [(srcs[name], p) for name in cfg['background']['sources']
                for p in sorted(glob.glob(os.path.join(cfg['raw_root'], srcs[name]['glob']), recursive=True))
                if not p.endswith(('.sha256ok', '.done', '.part'))]
    docs, per_file = 0, -(-cfg['background']['docs'] // max(1, len(bg_files)))   # even share per file
    for src, p in bg_files:
        for k, r in enumerate(read_rows(p, src['format'], end=None if src['format'] == 'parquet' or p.endswith('.gz') else 256 << 20)):
            if k >= per_file:
                break
            t = r.get(src.get('text_field', 'text'))
            if isinstance(t, str):
                h = ngram_hashes(t)
                bg.update(np.unique(h[in_sorted(h, allh)]).tolist()); docs += 1
    generic = np.array(sorted(k for k, c in bg.items() if c >= cfg['background']['min_docs']), np.uint64)
    final = np.setdiff1d(np.setdiff1d(allh, generic), shared)
    out = Path(cfg['work']) / 'bench_13gram.npy'
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, lambda t: np.save(open(t, 'wb'), final))
    atomic_json(str(out) + '.json', dict(files=len(files), texts_per_repo=dict(per_repo), ngrams_all=int(allh.size),
                                         shared_context_removed=int(shared.size),
                                         background_docs=docs, generic_removed=int(generic.size),
                                         ngrams_final=int(final.size), sha256=sha256(out)))
    print(json.dumps(dict(ngrams_all=int(allh.size), shared_context_removed=int(shared.size),
                          generic_removed=int(generic.size), final=int(final.size),
                          per_repo=dict(per_repo))), flush=True)


# ------------------------------------------------------------------ map

def owner_tag():
    j, t = os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_ARRAY_TASK_ID')
    return f'{j}_{t}' if j and t else (j or 'local')


def claim(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{owner_tag()} {os.getpid()}'.encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def cmd_release(cfg, a):
    """Remove map/pack claims (without a finished output) whose Slurm job is no longer running or pending."""
    import subprocess
    q = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-h', '-r', '-o', '%F_%K %i'],
                       capture_output=True, text=True, check=True).stdout.split()
    live = set(q) | {x.split('_')[0] for x in q}
    released = 0
    for stage, done_suffix in (('map', '.parquet.json'), ('pack', None)):
        for c in glob.glob(os.path.join(cfg['work'], stage, '*.claim')):
            fid = os.path.basename(c)[:-len('.claim')]
            if stage == 'map' and os.path.exists(os.path.join(cfg['work'], 'map', fid + done_suffix)):
                continue
            if stage == 'pack' and glob.glob(os.path.join(cfg['shards'], '*', fid + '.u16.json')):
                continue
            try:
                txt = open(c).read().split()
                owner = txt[0] if txt else ''
                if owner == 'local' or owner not in live:
                    os.unlink(c); released += 1
            except FileNotFoundError:          # another task released it concurrently
                pass
    print(json.dumps(dict(stage='release', released=released)), flush=True)


def map_file(cfg, fid, sname, path, start, end, bench, max_docs=None):
    sys.path.insert(0, QC_PATH)
    import quality_core
    src = {s['name']: s for s in cfg['sources']}[sname]
    counts = collections.Counter(); cols = collections.defaultdict(list)
    for ordinal, row in enumerate(read_rows(path, src['format'], start=start, end=end)):
        if max_docs and ordinal >= max_docs:
            break
        counts['docs_in'] += 1
        text, score = filter_doc(row, src)
        if text is None:
            counts['drop/' + score] += 1; continue
        h = ngram_hashes(text)
        if np.unique(h[in_sorted(h, bench)]).size >= cfg.get('bench_min_hits', 2):
            counts['drop/benchmark_13gram'] += 1; continue
        sig, _, _ = quality_core.sketch(text)
        bands = quality_core.bands(sig) if sig else []
        cols['ordinal'].append(ordinal)
        cols['exact'].append(xxhash.xxh3_64_intdigest(' '.join(text.split())))
        for b in range(14):
            cols[f'b{b}'].append(int.from_bytes(bands[b][:8], 'little') if bands else 0)
        cols['rank'].append(src['rank']); cols['score'].append(score); cols['chars'].append(len(text))
        counts['kept'] += 1; counts['kept_chars'] += len(text)
    schema = [('ordinal', pa.uint32()), ('exact', pa.uint64())] + [(f'b{b}', pa.uint64()) for b in range(14)] + \
             [('rank', pa.uint8()), ('score', pa.float32()), ('chars', pa.uint32())]
    table = pa.table({k: pa.array(cols[k], t) for k, t in schema})
    out = Path(cfg['work']) / 'map' / (fid + '.parquet')
    atomic_write(out, lambda t: pq.write_table(table, t, compression='zstd'))
    atomic_json(str(out) + '.json', dict(id=fid, source=sname, input=path, start=start, end=end, counts=dict(counts),
                                         rows=table.num_rows,
                                         sha256=sha256(out), finished=time.strftime('%FT%T')))
    return counts


def cmd_map(cfg, a):
    bench = np.load(Path(cfg['work']) / 'bench_13gram.npy')
    todo = inputs(cfg)
    if a.only:
        todo = [t for t in todo if t[1] in a.only.split(',')]
    for fid, sname, path, start, end in todo:
        out = Path(cfg['work']) / 'map' / (fid + '.parquet.json')
        if out.exists():
            continue
        lock = Path(cfg['work']) / 'map' / (fid + '.claim')
        lock.parent.mkdir(parents=True, exist_ok=True)
        if not claim(lock):
            continue
        t = time.time()
        try:
            c = map_file(cfg, fid, sname, path, start, end, bench, a.max_docs)
        except Exception as e:
            lock.unlink(missing_ok=True)
            print(json.dumps(dict(status='MAP_FAILED_RELEASED', id=fid, error=repr(e)[:300])), flush=True)
            continue
        print(json.dumps(dict(status='MAPPED', id=fid, seconds=round(time.time() - t), **c)), flush=True)


# ------------------------------------------------------------------ reduce

def find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def union_pairs(parent, a, b):
    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(parent, x), find(parent, y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)


def collision_pairs(keys):
    """Pairs (first index of a run, other index) for runs of equal non-zero keys."""
    order = np.argsort(keys, kind='stable'); k = keys[order]
    same = np.flatnonzero((k[1:] == k[:-1]) & (k[1:] != 0)) + 1
    if same.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    head = starts[np.searchsorted(starts, same, side='right') - 1]
    return order[head], order[same]


def cmd_reduce(cfg, a):
    recs = sorted(glob.glob(os.path.join(cfg['work'], 'map', '*.parquet.json')))
    expected = {t[0] for t in inputs(cfg)}
    have = {json.load(open(r))['id'] for r in recs}
    if expected - have and not a.partial:
        raise SystemExit(f'{len(expected - have)} inputs not mapped yet (use --partial to reduce what exists)')
    ids, offsets, n = [], [0], 0
    for r in recs:
        m = json.load(open(r)); ids.append(m['id']); n += m['rows']; offsets.append(n)
    col = lambda name, dt: np.concatenate([pq.read_table(r[:-5], columns=[name])[name].to_numpy().astype(dt)
                                           for r in recs]) if recs else np.empty(0, dt)
    parent = list(range(n)); edges = collections.Counter()
    for name in ['exact'] + [f'b{b}' for b in range(14)]:
        x, y = collision_pairs(col(name, np.uint64))
        union_pairs(parent, x, y); edges[name] = int(x.size)
        print(json.dumps(dict(stage='reduce', key=name, pairs=int(x.size))), flush=True)
    rank, score, chars = col('rank', np.int64), col('score', np.float64), col('chars', np.int64)
    roots = np.fromiter((find(parent, i) for i in range(n)), np.int64, n)
    # best member per component: lowest rank, then highest score, then most characters
    order = np.lexsort((-chars, -score, rank, roots))
    first = np.r_[True, roots[order][1:] != roots[order][:-1]]
    keep = np.zeros(n, bool); keep[order[first]] = True
    stats = collections.Counter()
    for i, fid in enumerate(ids):
        lo, hi = offsets[i], offsets[i + 1]
        ords = pq.read_table(recs[i][:-5], columns=['ordinal'])['ordinal'].to_numpy()
        kept = ords[keep[lo:hi]].astype(np.uint32)
        out = Path(cfg['work']) / 'keep' / (fid + '.npy')
        atomic_write(out, lambda t: np.save(open(t, 'wb'), kept))
        src = fid.split('__')[0]
        stats[src + '/mapped'] += hi - lo; stats[src + '/kept'] += int(kept.size)
        stats[src + '/kept_chars'] += int(chars[lo:hi][keep[lo:hi]].sum())
    atomic_json(Path(cfg['work']) / 'keep' / 'REDUCE.json',
                dict(documents=n, kept=int(keep.sum()), pairs=dict(edges), per_source=dict(stats),
                     inputs=len(ids), partial=bool(expected - have), finished=time.strftime('%FT%T')))
    print(json.dumps(dict(documents=n, kept=int(keep.sum()), per_source=dict(stats))), flush=True)


# ------------------------------------------------------------------ pack

def fim_split(text, rng):
    """Two uniform character cut points -> (prefix, middle, suffix); prefix + middle + suffix == text."""
    i, j = sorted(rng.integers(0, len(text) + 1, 2).tolist())
    return text[:i], text[i:j], text[j:]


def encode_docs(tok, items):
    """items: str (left-to-right) or (prefix, middle, suffix) (FIM, PSM order). -> list of uint16 arrays ending in EOS.
    PSM: <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle EOS (DeepSeek-Coder / StarCoder layout)."""
    flat = []
    for it in items:
        flat.extend([it] if isinstance(it, str) else list(it))
    encs = iter(tok.encode_batch(flat, add_special_tokens=False))
    ids = lambda: (lambda a: a[a != EOS])(np.asarray(next(encs).ids, np.uint16))   # a stray EOS would split the doc
    out = []
    for it in items:
        if isinstance(it, str):
            body = ids()
        else:
            pre, mid, suf = ids(), ids(), ids()
            body = np.concatenate([[FIM_PREFIX], pre, [FIM_SUFFIX], suf, [FIM_MIDDLE], mid]).astype(np.uint16)
        out.append(np.append(body, np.uint16(EOS)).astype('<u2'))
    return out


def cmd_pack(cfg, a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(cfg['tokenizer'])
    for fid, sname, path, start, end in inputs(cfg):
        keepf = Path(cfg['work']) / 'keep' / (fid + '.npy')
        out = Path(cfg['shards']) / sname / (fid + '.u16')
        if not keepf.exists() or Path(str(out) + '.json').exists():
            continue
        lock = Path(cfg['work']) / 'pack' / (fid + '.claim'); lock.parent.mkdir(parents=True, exist_ok=True)
        if not claim(lock):
            continue
        t = time.time(); src = {s['name']: s for s in cfg['sources']}[sname]
        keep = set(np.load(keepf).tolist()); texts, docs, ntok = [], 0, 0
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + '.tmp-' + uuid.uuid4().hex)
        fim_rate = src.get('fim_rate', 0.0)
        with open(tmp, 'wb') as f:
            def flush():
                nonlocal ntok
                for arr in encode_docs(tok, texts):
                    f.write(arr.tobytes()); ntok += arr.size
                texts.clear()
            for ordinal, row in enumerate(read_rows(path, src['format'], start=start, end=end)):
                if ordinal in keep:
                    text, _ = filter_doc(row, src)
                    assert text is not None, f'filter not deterministic: {fid}:{ordinal}'
                    rng = np.random.default_rng([zlib.crc32(fid.encode()), ordinal])
                    texts.append(fim_split(text, rng) if fim_rate and rng.random() < fim_rate else text)
                    docs += 1
                    if len(texts) >= 2048:
                        flush()
            flush(); f.flush(); os.fsync(f.fileno())
        assert docs == len(keep), f'{fid}: packed {docs} of {len(keep)} kept documents'
        os.replace(tmp, out)
        atomic_json(str(out) + '.json', dict(id=fid, source=sname, input=path, start=start, end=end, documents=docs,
                                             tokens=ntok,
                                             sha256=sha256(out), seconds=round(time.time() - t)))
        print(json.dumps(dict(status='PACKED', id=fid, documents=docs, tokens=ntok,
                              seconds=round(time.time() - t))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['bench', 'map', 'reduce', 'pack', 'release'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--only', default='', help='comma-separated source names (map)')
    ap.add_argument('--max-docs', type=int, default=0, help='smoke tests: stop each input after N rows (map)')
    ap.add_argument('--partial', action='store_true', help='reduce over the inputs mapped so far')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    dict(bench=cmd_bench, map=cmd_map, reduce=cmd_reduce, pack=cmd_pack, release=cmd_release)[a.stage](cfg, a)


if __name__ == '__main__':
    main()

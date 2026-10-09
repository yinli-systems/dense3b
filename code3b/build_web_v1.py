"""web-v1 for the 3B run: strict-selected FineWeb-Edu, GLOBALLY deduplicated, crawl cutoff <= 2024, decontaminated
against the code/math benchmark 13-grams, with the existing web validation documents excluded from training.

web15b (v0) was packed from 25 CC snapshots each deduplicated only within itself: 83.7% of its documents are
cross-snapshot copies (up to 54 of one page), it skipped benchmark decontamination, and its validation documents
also occur in its training shards.

  select  one process: walk files round-robin across snapshots <= CC-MAIN-2024-51, keep the first occurrence of each
          normalized_text_hash (validation documents pre-seeded as seen), stop at --target-tokens.
          -> work/keep/<file_id>.npy (row indices) + SELECT.json
  pack    array workers claim files: kept rows -> benchmark 13-gram filter (same rule as build_v0) -> train_<id>.u16

  python build_web_v1.py select --out DIR [--target-tokens 15e9]
  python build_web_v1.py pack --out DIR --bench /.../bench_13gram.npy
"""
import argparse, collections, glob, hashlib, json, os, re, sys, time, uuid
import numpy as np
import pyarrow.parquet as pq

OUT_ROOT = '/data/run01/scxi253/web-strict-20261003/out'
VAL_PACK = '/data/run01/scxi253/code3b-train/web15b/PACK.json'
EOS = 50279
CC = re.compile(r'CC-MAIN-(\d{4})-(\d{2})')


def all_files():
    return sorted(f for f in glob.glob(f'{OUT_ROOT}/**/*.parquet', recursive=True) if os.path.exists(f + '.json'))


def fid(path):
    return hashlib.sha1(path.encode()).hexdigest()[:16]


def h64(b):
    return int.from_bytes(b[:8], 'little')


def val_hashes():
    """The web15b validation shard = documents of the first file in sorted order until 20M tokens."""
    pack = json.load(open(VAL_PACK))
    first = all_files()[0]
    t = pq.read_table(first, columns=['normalized_text_hash', 'n_tokens'])
    seen, tok = set(), 0
    for h, n in zip(t['normalized_text_hash'].to_pylist(), t['n_tokens'].to_pylist()):
        seen.add(h64(h)); tok += n + 1          # pack_web_tokens counted the EOS too
        if tok >= pack['val']['tokens']:
            break
    return seen, first


def cmd_select(a):
    files = [f for f in all_files() if CC.search(f).group(0) <= 'CC-MAIN-2024-51']
    by = collections.defaultdict(list)
    for f in files:
        by[CC.search(f).group(0)].append(f)
    dumps = sorted(by)
    order = []                                          # round-robin over snapshots: diverse years in any prefix
    for i in range(max(len(v) for v in by.values())):
        for d in dumps:
            if i < len(by[d]):
                order.append(by[d][i])
    seen, val_file = val_hashes()
    os.makedirs(f'{a.out}/work/keep', exist_ok=True)
    kept_tok = kept_docs = docs = dup = 0
    per_dump = collections.Counter()
    used = []
    for f in order:
        t = pq.read_table(f, columns=['normalized_text_hash', 'n_tokens'])
        keep = []
        for i, (h, n) in enumerate(zip(t['normalized_text_hash'].to_pylist(), t['n_tokens'].to_pylist())):
            docs += 1
            k = h64(h)
            if k in seen:
                dup += 1; continue
            seen.add(k); keep.append(i); kept_tok += n + 1; kept_docs += 1
        np.save(f'{a.out}/work/keep/{fid(f)}.npy', np.asarray(keep, np.int32))
        used.append(f); per_dump[CC.search(f).group(0)] += len(keep)
        if kept_tok >= a.target_tokens:
            break
    s = dict(files_used=len(used), files_available=len(files), dumps=f'{dumps[0]}..{dumps[-1]}', docs_seen=docs,
             duplicates_dropped=dup, kept_docs=kept_docs, kept_tokens=kept_tok, val_file=val_file,
             val_docs_excluded=True, per_dump_docs=dict(per_dump), files=used, finished=time.strftime('%FT%T'))
    json.dump(s, open(f'{a.out}/SELECT.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k not in ('files', 'per_dump_docs')}), flush=True)


def claim(p):
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def cmd_pack(a):
    sys.path.insert(0, '/data/run01/scxi253/code3b-data-20261003/v0/src')
    import build_v0 as B
    bench = np.load(a.bench)
    sel = json.load(open(f'{a.out}/SELECT.json'))
    os.makedirs(f'{a.out}/shards', exist_ok=True); os.makedirs(f'{a.out}/work/claims', exist_ok=True)
    for f in sel['files']:
        i = fid(f); out = f'{a.out}/shards/train_{i}.u16'
        if os.path.exists(out + '.json') or not claim(f'{a.out}/work/claims/{i}'):
            continue
        keep = set(np.load(f'{a.out}/work/keep/{i}.npy').tolist())
        t = pq.read_table(f, columns=['text', 'tokens'])
        arrs, c = [], collections.Counter()
        for r, (text, tok) in enumerate(zip(t['text'].to_pylist(), t['tokens'].to_pylist())):
            if r not in keep:
                continue
            h = B.ngram_hashes(text)
            if np.unique(h[B.in_sorted(h, bench)]).size >= 2:
                c['drop/benchmark_13gram'] += 1; continue
            ids = np.frombuffer(tok, dtype='<u2')
            assert ids[-1] == EOS
            arrs.append(ids); c['docs'] += 1; c['tokens'] += ids.size
        tmp = out + '.tmp-' + uuid.uuid4().hex
        (np.concatenate(arrs) if arrs else np.empty(0, '<u2')).astype('<u2').tofile(tmp); os.replace(tmp, out)
        json.dump(dict(source=f, kept_rows=len(keep), **c), open(out + '.json', 'w'))
        print(json.dumps(dict(id=i, **c)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['select', 'pack'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--target-tokens', type=float, default=15e9)
    ap.add_argument('--bench', default='/data/run01/scxi253/code3b-data-20261003/v0/work/bench_13gram.npy')
    a = ap.parse_args()
    dict(select=cmd_select, pack=cmd_pack)[a.stage](a)


if __name__ == '__main__':
    main()

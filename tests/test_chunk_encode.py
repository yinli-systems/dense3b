"""Exactness + speed of ChunkEncoder vs the reference tokenizer on real selected documents of every source."""
import json, sys, time, random
import numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from chunk_encode import ChunkEncoder
from score_core import verified_text, windows_for_length
plan = json.load(open(sys.argv[1])); per = int(sys.argv[2])
dec = Tokenizer.from_file(plan['decoder_path'])
tok = AutoTokenizer.from_pretrained(plan['model_path'], local_files_only=True)
print('padding in tokenizer.json:', tok._tokenizer.padding, 'truncation:', tok._tokenizer.truncation)
print('normalizer', tok._tokenizer.normalizer.__class__.__name__, 'pretok', tok._tokenizer.pre_tokenizer.__class__.__name__)
seen = {}; rnd = random.Random(7)
for part in plan['parts']:
    for t in part['tasks']:
        seen.setdefault(t['source'], []).append(t)
bad = 0; total = 0
for src, tasks in sorted(seen.items()):
    t = rnd.choice(tasks); m = t['mask']
    keep = np.unpackbits(np.frombuffer(open(m['mask_path'], 'rb').read(), dtype=np.uint8), bitorder='little')
    pf = pq.ParquetFile(t['metadata']['path']); base = 0; rg = 0
    while base + pf.metadata.row_group(rg).num_rows <= t['start']:
        base += pf.metadata.row_group(rg).num_rows; rg += 1
    rows = pf.read_row_group(rg, columns=['ordinal','offset','tokens','text_hash','token_hash']).to_pylist()
    f = open(m['token_artifact']['path'], 'rb'); texts = []
    for r in rows:
        if t['start'] <= r['ordinal'] < t['stop'] and keep[r['ordinal']]:
            f.seek(r['offset'] * 2); x = verified_text(f.read(r['tokens'] * 2), f.read(2), r, dec)
            if x is not None: texts.append(x)
        if len(texts) >= per: break
    fresh = AutoTokenizer.from_pretrained(plan['model_path'], local_files_only=True)
    ce = ChunkEncoder(fresh._tokenizer)  # built before any tok.encode call, as in the scoring workers
    t0 = time.time(); ref = [tok.encode(x, add_special_tokens=False, truncation=False, verbose=False) for x in texts]; t1 = time.time()
    W = 510; got = [ce.length_and_slices(x, lambda n: windows_for_length(n, W)[0]) for x in texts]; t2 = time.time()
    b = 0
    for a, (n, spans, sl) in zip(ref, got):
        exp = [a[lo:hi] for lo, hi in windows_for_length(len(a), W)[0]]
        b += (n != len(a)) or (sl != exp)
    b += sum(ce.encode(x) != a for x, a in zip(texts[:500], ref[:500]))
    bad += b; total += len(texts)
    print('%-26s docs %5d  mismatched %d  ref %.1fs  chunked %.1fs  speedup %.1fx  hit %.2f' % (
        src, len(texts), b, t1 - t0, t2 - t1, (t1 - t0) / max(1e-9, t2 - t1), ce.hits / max(1, ce.hits + ce.misses)))
print('TOTAL docs', total, 'mismatched', bad)

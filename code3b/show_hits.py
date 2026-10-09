"""Print the first matching 13-word window of the first N benchmark hits in one input (manual audit)."""
import json, sys
import numpy as np
import build_v0 as B

cfg = json.load(open(sys.argv[1])); src_name, path, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
bench = np.load(cfg['work'] + '/bench_13gram.npy')
src = {s['name']: s for s in cfg['sources']}[src_name]
shown = 0
for i, row in enumerate(B.read_rows(path, src['format'], end=64 << 20)):
    text, _ = B.filter_doc(row, src)
    if text is None:
        continue
    h = B.ngram_hashes(text); m = B.in_sorted(h, bench)
    if np.unique(h[m]).size >= cfg.get('bench_min_hits', 2):
        words = B.WORD.findall(text.lower()); j = int(np.argmax(m))
        print(f'[{i}] hits={int(m.sum())}/{h.size} :: ' + ' '.join(words[j:j + 13]))
        shown += 1
        if shown >= n:
            break

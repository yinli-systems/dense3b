"""Re-tokenize the moe1b pretraining shards (NeoX-50k u16) into the 65,536-entry reasoning tokenizer (u16).

Lossless: the old tokenizer is byte-level BPE, so decode(old ids) reproduces the document bytes exactly. Document
boundaries (old EOS 50279) and FIM sentinels (50280-50282) are carried over to the new special ids.

  python -m bench.reason.retokenize --list /data/run01/scxi253/moe1b/mix-m1/data.txt --task I --ntasks N
Output: $OUT/<source>/<basename>.u16 (+ .json with token counts); existing outputs are skipped.
"""
import argparse, json, os, sys, time
import numpy as np

OLD_TOK = '/data/run01/scxi253/moe7b-quality-stage2-20261002T2327Z/tokenizer.json'
NEW_TOK = os.environ.get('NEW_TOK', '/data/run01/scxi253/reason3b-data/tokenizer-v1/tokenizer.json')
OUT = os.environ.get('RETOK_OUT', '/data/run01/scxi253/reason3b-data/pt-v1')
OLD_EOS = 50279
CHUNK = 4_000_000


def source_of(p):
    return 'web-v2' if '/web-v2/' in p else p.split('/')[-2]


def convert(path, old, new, sp):
    src = source_of(path)
    dst = f'{OUT}/{src}/{os.path.basename(path)}'
    if os.path.exists(dst + '.json'):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    a = np.memmap(path, dtype=np.uint16, mode='r')
    n_old, pos, outs = len(a), 0, []
    t0 = time.time()
    while pos < n_old:
        end = min(n_old, pos + CHUNK)
        if end < n_old:                                   # cut after an EOS so documents stay whole
            e = np.flatnonzero(np.asarray(a[pos:end]) == OLD_EOS)
            if len(e):
                end = pos + int(e[-1]) + 1
        w = np.asarray(a[pos:end]).astype(np.int64)
        pos = end
        # split into runs of ordinary ids separated by special ids (>= OLD_EOS)
        sp_idx = np.flatnonzero(w >= OLD_EOS)
        bounds = np.concatenate([[-1], sp_idx, [len(w)]])
        segs, seps = [], []
        for i in range(len(bounds) - 1):
            s, e = bounds[i] + 1, bounds[i + 1]
            segs.append(w[s:e].tolist())
            seps.append(int(w[e]) if e < len(w) else None)
        texts = old.decode_batch(segs, skip_special_tokens=False)
        enc = new.encode_batch([t for t in texts], add_special_tokens=False)
        buf = []
        for e_, sep in zip(enc, seps):
            buf.extend(e_.ids)
            if sep is not None:
                buf.append(sp[sep])
        outs.append(np.asarray(buf, dtype=np.uint16))
    out = np.concatenate(outs) if outs else np.zeros(0, np.uint16)
    out.tofile(dst + '.tmp')
    os.replace(dst + '.tmp', dst)
    meta = dict(src=path, old_tokens=int(n_old), new_tokens=int(len(out)), ratio=round(len(out) / max(1, n_old), 4),
                docs=int((out == sp[OLD_EOS]).sum()), seconds=round(time.time() - t0, 1))
    json.dump(meta, open(dst + '.json', 'w'))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='append', required=True)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    old, new = Tokenizer.from_file(OLD_TOK), Tokenizer.from_file(NEW_TOK)
    sp = {OLD_EOS: new.token_to_id('<|endoftext|>'), 50280: new.token_to_id('<|fim_prefix|>'),
          50281: new.token_to_id('<|fim_middle|>'), 50282: new.token_to_id('<|fim_suffix|>')}
    assert sp[OLD_EOS] == 0 and None not in sp.values(), sp
    paths = []
    for lst in a.list:
        paths += [p.strip() for p in open(lst).read().replace('\n', ',').split(',') if p.strip()]
    paths = sorted(set(paths), key=lambda p: (-os.path.getsize(p), p))
    mine = paths[a.task::a.ntasks]
    tot_old = tot_new = 0
    t0 = time.time()
    for p in mine:
        m = convert(p, old, new, sp)
        if m:
            tot_old += m['old_tokens']; tot_new += m['new_tokens']
            print(json.dumps(dict(**m, cum_old_B=round(tot_old / 1e9, 3),
                                  old_tok_per_s=int(tot_old / (time.time() - t0)))), flush=True)
    print(json.dumps(dict(DONE=True, task=a.task, files=len(mine))), flush=True)


if __name__ == '__main__':
    main()

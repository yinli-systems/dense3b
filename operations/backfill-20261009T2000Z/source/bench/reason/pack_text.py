"""Pack raw-text parquet corpora (e.g. FineMath-3+) into u16 shards with the 64k reasoning tokenizer.

Every document is checked against the 8-gram eval index of build_reason_data.py (dropped when it contains >= 50 % of
an eval item's 8-grams); exact duplicates are dropped by text hash within a task.

  python -m bench.reason.pack_text --glob 'raw/HuggingFaceTB__finemath/finemath-3plus/*.parquet' --name finemath-3plus \
      --exclude-list /data/run01/scxi253/reason3b-data/finemath4plus_ids.txt --task I --ntasks N
"""
import argparse, glob, hashlib, json, os, time
import numpy as np

from bench.reason.build_reason_data import Decontam, ROOT

TOK = ROOT + '/tokenizer-v1/tokenizer.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True, help='relative to $ROOT')
    ap.add_argument('--name', required=True)
    ap.add_argument('--text-col', default='text')
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--shard-tokens', type=int, default=256 << 20)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    eos = tok.token_to_id('<|endoftext|>')
    assert eos == 0
    dc = Decontam()
    files = sorted(glob.glob(os.path.join(ROOT, a.glob)))[a.task::a.ntasks]
    out_dir = f'{ROOT}/pt-v1/{a.name}'
    os.makedirs(out_dir, exist_ok=True)
    seen, st, buf, nbuf, nfile, t0 = set(), dict(docs=0, kept=0, contam=0, dup=0, tokens=0), [], 0, 0, time.time()

    def flush():
        nonlocal buf, nbuf, nfile
        if not buf:
            return
        p = f'{out_dir}/{a.name}-t{a.task:03d}-{nfile:03d}.u16'
        np.concatenate(buf).astype('<u2').tofile(p + '.tmp'); os.replace(p + '.tmp', p)
        nfile += 1; buf, nbuf = [], 0

    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            texts = pf.read_row_group(g, columns=[a.text_col]).column(0).to_pylist()
            keep = []
            for t in texts:
                st['docs'] += 1
                if not t:
                    continue
                h = hashlib.blake2b(t.encode(), digest_size=8).digest()
                if h in seen:
                    st['dup'] += 1; continue
                seen.add(h)
                if dc.check(t[:20000]):
                    st['contam'] += 1; continue
                keep.append(t)
            for e in tok.encode_batch(keep, add_special_tokens=False):
                ids = np.asarray(e.ids + [eos], dtype=np.uint16)
                buf.append(ids); nbuf += ids.size; st['tokens'] += ids.size; st['kept'] += 1
                if nbuf >= a.shard_tokens:
                    flush()
        print(json.dumps(dict(file=os.path.basename(f), **st, minutes=round((time.time() - t0) / 60, 1))), flush=True)
    flush()
    json.dump(st, open(f'{out_dir}/{a.name}-t{a.task:03d}.stats.json', 'w'))


if __name__ == '__main__':
    main()

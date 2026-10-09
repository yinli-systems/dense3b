"""Select, render, tokenize and pack the normalized reasoning corpus (build_reason_data.py output) into u16 shards.

  sft  mode: chat template, loss mask (.m8) = 1 on assistant tokens only, samples never split: first-fit-decreasing
             bin packing into windows of exactly --seq tokens (the loader reads windows of seq+1 at stride seq, so
             the extra target is the next bin's first token, which is always a masked <|im_start|>). Pad = id 1.
  mid  mode: P2 reasoning mid-training text, documents concatenated with EOS (standard pretraining packing, no mask).

Partitioned by prompt hash, so the per-prompt response cap is exact inside one task:
  python -m bench.reason.build_sft --mode sft --seq 16384 --task I --ntasks N --out DIR
Template (Qwen/ChatML style, our special ids):
  <|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n<think>\\n...\\n</think>\\n\\n
  ...<|im_end|>\\n<|endoftext|>
"""
import argparse, glob, gzip, hashlib, json, os, time
from collections import Counter, defaultdict
import numpy as np

ROOT = '/data/run01/scxi253/reason3b-data'
TOK = ROOT + '/tokenizer-v1/tokenizer.json'

DOMAIN = {'math': 'math', 'code': 'code', 'science': 'science', 'stem': 'science', 'chat': 'general',
          'tool': 'tool', 'safety': 'general', 'instruction_following': 'general', 'mixed': 'mixed'}
# SFT keep-rate per (source, domain). Tuned after `build_reason_data summary`; 1.0 = keep all that passed filters.
SFT_RATE = defaultdict(lambda: 1.0, {('nemotron_v1', 'science'): 0.35, ('nemotron_v1', 'math'): 0.5,
                                     ('nemotron_v1', 'code'): 0.6, ('openmathreasoning', 'math'): 0.35})
SFT_CAP_PER_PROMPT = 4
NO_THINK_RATE = 0.25
MID_RATE = 0.35            # P2 mid-training sample rate (the /data quota leaves ~200 GB of headroom)        # of non-reasoning (empty-think) samples kept in SFT: general chat stays, but light


def frac(key):
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), 'little') / 2 ** 64


def records(task, ntasks):
    for f in sorted(glob.glob(f'{ROOT}/norm/*/part-*.jsonl.gz')):
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                r = json.loads(line)
                if int(r['phash'], 16) % ntasks == task:
                    yield r


def render(tok, ids_of, msgs):
    """-> (ids, mask) for a ChatML conversation; mask 1 on assistant content + its <|im_end|>."""
    ids, mask = [], []
    for m in msgs:
        head = tok.encode(f"{m['role']}\n", add_special_tokens=False).ids
        body = tok.encode(m['content'], add_special_tokens=False).ids
        nl = tok.encode('\n', add_special_tokens=False).ids
        a = 1 if m['role'] == 'assistant' else 0
        ids += [ids_of['<|im_start|>']] + head; mask += [0] * (1 + len(head))
        ids += body + [ids_of['<|im_end|>']]; mask += [a] * (len(body) + 1)
        ids += nl; mask += [0] * len(nl)
    ids.append(ids_of['<|endoftext|>']); mask.append(0)
    return ids, mask


def pack_ffd(samples, L):
    """samples: list of (ids, mask) with len <= L. Best-fit-decreasing into bins of exactly L tokens (pad 1).
    `open_` is kept sorted by free space, so each placement is a bisect (O(n log n) overall)."""
    import bisect
    order = sorted(range(len(samples)), key=lambda i: -len(samples[i][0]))
    bins, free, open_ = [], [], []                     # open_: sorted list of (free, bin index)
    for i in order:
        n = len(samples[i][0])
        k = bisect.bisect_left(open_, (n, -1))
        if k < len(open_):
            f, b = open_.pop(k)
            bins[b].append(i); free[b] = f - n
        else:
            b = len(bins); bins.append([i]); free.append(L - n)
        if free[b] >= 64:                              # bins with < 64 free tokens are closed
            bisect.insort(open_, (free[b], b))
    return bins, free


def write(out_prefix, chunks_ids, chunks_mask):
    ids = np.concatenate(chunks_ids).astype('<u2')
    ids.tofile(out_prefix + '.u16.tmp'); os.replace(out_prefix + '.u16.tmp', out_prefix + '.u16')
    if chunks_mask is not None:
        m = np.concatenate(chunks_mask).astype(np.uint8)
        assert m.size == ids.size
        m.tofile(out_prefix + '.m8.tmp'); os.replace(out_prefix + '.m8.tmp', out_prefix + '.m8')
    return ids.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sft', 'mid'], required=True)
    ap.add_argument('--seq', type=int, default=16384)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard-tokens', type=int, default=256 << 20)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    ids_of = {t: tok.token_to_id(t) for t in ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|pad|>')}
    assert ids_of['<|endoftext|>'] == 0 and ids_of['<|pad|>'] == 1, ids_of
    os.makedirs(a.out, exist_ok=True)
    st, t0 = Counter(), time.time()
    per_prompt = Counter()
    sel = []                                            # (ids, mask, domain)
    for r in records(a.task, a.ntasks):
        dom = DOMAIN.get(r['domain'], 'mixed')
        st['seen'] += 1
        if a.mode == 'sft':
            if frac('rate' + r['id']) >= SFT_RATE[(r['src'], dom)]:
                st['drop_rate'] += 1; continue
            if not r['think'] and frac('nt' + r['id']) >= NO_THINK_RATE and dom not in ('general', 'tool'):
                st['drop_nothink'] += 1; continue
            if r['verified'] is False:
                st['drop_unverified'] += 1; continue
            if per_prompt[r['phash']] >= SFT_CAP_PER_PROMPT:
                st['drop_cap'] += 1; continue
        if a.mode == 'mid':                             # quota-bounded P2 set: <= 1 response per prompt, ~35 % sample
            if per_prompt[r['phash']] >= 1 or frac('mid' + r['id']) >= MID_RATE:
                st['drop_mid'] += 1; continue
        ids, mask = render(tok, ids_of, r['messages'])
        if a.mode == 'sft' and len(ids) > a.seq:
            st['drop_long'] += 1; continue
        per_prompt[r['phash']] += 1
        st['kept'] += 1; st['kept_' + dom] += 1; st['tok_' + dom] += len(ids)
        sel.append((np.asarray(ids, np.uint16), np.asarray(mask, np.uint8) if a.mode == 'sft' else None))
    # shuffle deterministically, then pack / concatenate
    rng = np.random.default_rng(a.task)
    perm = rng.permutation(len(sel))
    sel = [sel[i] for i in perm]
    n_files = 0; buf_i, buf_m, buf_n = [], [], 0
    tag = f'{a.mode}-{a.seq}-t{a.task:03d}'

    def flush():
        nonlocal n_files, buf_i, buf_m, buf_n
        if buf_n:
            write(f'{a.out}/{tag}-{n_files:03d}', buf_i, buf_m if a.mode == 'sft' else None)
            n_files += 1; buf_i, buf_m, buf_n = [], [], 0

    if a.mode == 'sft':
        # pack in groups of ~20k samples so FFD stays fast and order stays mixed
        for g in range(0, len(sel), 20000):
            grp = sel[g:g + 20000]
            bins, free = pack_ffd(grp, a.seq)
            st['pad_tokens'] += int(sum(free)); st['bins'] += len(bins)
            for b, f in zip(bins, free):
                bi = [grp[i][0] for i in b] + [np.ones(f, np.uint16)]
                bm = [grp[i][1] for i in b] + [np.zeros(f, np.uint8)]
                buf_i += bi; buf_m += bm; buf_n += a.seq
                if buf_n >= a.shard_tokens:
                    flush()
    else:
        for ids, _ in sel:
            buf_i.append(ids); buf_n += len(ids)
            if buf_n >= a.shard_tokens:
                flush()
    flush()
    st['seconds'] = round(time.time() - t0)
    st['files'] = n_files
    if a.mode == 'sft' and st['bins']:
        st['pack_efficiency'] = round(1 - st['pad_tokens'] / (st['bins'] * a.seq), 4)
    json.dump(dict(st), open(f'{a.out}/{tag}.stats.json', 'w'), indent=1)
    print(json.dumps(dict(st)), flush=True)


if __name__ == '__main__':
    main()

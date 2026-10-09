"""Train the reasoning-model tokenizer: 65,536-entry byte-level BPE (fits u16 shards) with single-digit splitting.

Why not reuse the 50,280 NeoX tokenizer of the 1B MoE / 3B dense: it chunks numbers inconsistently
('12345678' -> '12345','678'; '98765' -> '9','87','65'), which is known to hurt arithmetic and long-CoT math.
Qwen2.5's 151k vocabulary splits digits but costs 233M embedding parameters at d=1536 (about 30 % extra FLOPs/token
on the logits for a 0.7B-active model) and needs u32 shards.

  sample:  decode documents from the existing u16 shards (old tokenizer is byte-level, so decode is lossless) plus
           reasoning traces from the downloaded parquet datasets, per-domain quotas, into text files
  train:   BPE on the sample (pre-tokenizer = Qwen2 regex with \\p{N} isolated, NFC, byte-level)
  report:  bytes/token per domain on a held-out sample for old NeoX, Qwen2.5 and the new tokenizer

  python -m bench.reason.train_tokenizer sample --out DIR
  python -m bench.reason.train_tokenizer train --out DIR
  python -m bench.reason.train_tokenizer report --out DIR
"""
import argparse, glob, json, os, random, re, sys, time
import numpy as np

OLD_TOK = '/data/run01/scxi253/moe7b-quality-stage2-20261002T2327Z/tokenizer.json'
QWEN_TOK = '/data/run01/scxi253/moe7b-eval/meta/tokenizer/tokenizer.json'
MIX = '/data/run01/scxi253/moe1b/mix-m1/data.txt'
RAW = '/data/run01/scxi253/reason3b-data/raw'
OLD_EOS = 50279
VOCAB = int(os.environ.get('VOCAB', 65536))

# GB of text per domain for training (held-out = 2 % extra). Code/math/reasoning are over-weighted relative to the
# pretraining mix: the vocabulary should compress what the model will spend its tokens on after mid-training.
QUOTA_GB = {'web': 1.6, 'wikipedia': 0.25, 'arxiv': 0.3, 'stackexchange': 0.2, 'code': 1.2, 'math': 0.8,
            'reasoning': 0.9, 'chat': 0.15}
SHARD_DOMAIN = {'web-v2': 'web', 'wikipedia': 'wikipedia', 'arxiv': 'arxiv', 'stackexchange': 'stackexchange',
                'stackv2-edu': 'code', 'swallow-code': 'code', 'opc-annealing': 'code', 'finemath-4plus': 'math',
                'nemotron-cc-math-4plus': 'math', 'swallow-math': 'math'}

SPECIAL = ['<|endoftext|>', '<|pad|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>',
           '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|repo_name|>', '<|file_sep|>',
           '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>', '<answer>', '</answer>']
SPECIAL += [f'<|reserved_{i}|>' for i in range(64 - len(SPECIAL))]          # 64 special ids: 0..63

# Qwen2 pre-tokenizer regex: contractions, letters with one optional leading non-letter, ONE digit at a time,
# punctuation runs, newline runs, whitespace runs (indentation stays whole).
PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
       r"\s+(?!\S)|\s+")


def shard_groups():
    groups = {}
    for p in open(MIX).read().replace('\n', ',').split(','):
        p = p.strip()
        if not p:
            continue
        src = 'web-v2' if '/web-v2/' in p else p.split('/')[-2]
        groups.setdefault(SHARD_DOMAIN[src], []).append(p)
    return groups


def docs_from_shards(paths, tok, budget_bytes, rng):
    """Random 1M-token windows from random shards, split at EOS, drop the cut first/last docs."""
    out, got = [], 0
    while got < budget_bytes:
        p = rng.choice(paths)
        a = np.memmap(p, dtype=np.uint16, mode='r')
        if len(a) < 2_000_000:
            continue
        s = rng.randrange(0, len(a) - 1_000_000)
        w = np.asarray(a[s:s + 1_000_000])
        cuts = np.flatnonzero(w == OLD_EOS)
        for x, y in zip(cuts[:-1], cuts[1:]):
            if y - x > 16:
                out.append(w[x + 1:y].tolist())
        texts = tok.decode_batch(out, skip_special_tokens=False)
        out = []
        for t in texts:
            got += len(t.encode('utf-8'))
            yield t
            if got >= budget_bytes:
                return


def reasoning_rows(budget_bytes, rng, kind):
    """Long-CoT traces (reasoning) or short chat turns, flattened as 'user\\nassistant' text."""
    import pyarrow.parquet as pq
    if kind == 'reasoning':
        pats = ['open-thoughts__OpenThoughts3-1.2M/data/*.parquet', 'open-r1__OpenR1-Math-220k/data/*.parquet',
                'nvidia__OpenCodeReasoning/*/*.parquet', 'PrimeIntellect__SYNTHETIC-1-SFT-Data/data/*.parquet',
                'nvidia__OpenMathReasoning/data/cot-*.parquet',
                'nvidia__Nemotron-Post-Training-Dataset-v1/data/math-*.parquet',
                'nvidia__Nemotron-Post-Training-Dataset-v1/data/code-*.parquet',
                'nvidia__Nemotron-Post-Training-Dataset-v1/data/stem-*.parquet']
    else:
        pats = ['allenai__tulu-3-sft-mixture/data/*.parquet']
    files = [f for pat in pats for f in glob.glob(f'{RAW}/{pat}')]
    if not files:
        print(f'WARN no files for {kind}', flush=True)
        return
    rng.shuffle(files)
    got = 0
    for f in files:
        t = pq.read_table(f)
        cols = t.column_names
        rows = t.slice(0, min(len(t), 1500)).to_pylist()
        for r in rows:
            parts = []
            for c in ('conversations', 'messages'):
                if c in r and r[c]:
                    ms = json.loads(r[c]) if isinstance(r[c], str) else r[c]
                    for m in ms:
                        parts.append(str(m.get('value') or m.get('content') or ''))
            if not parts:
                for c in ('problem', 'input', 'prompt', 'question', 'generated_solution', 'output', 'solution',
                          'response', 'llm_response'):
                    if c in cols and r.get(c):
                        parts.append(str(r[c]))
            t_ = '\n'.join(parts)
            if len(t_) < 64:
                continue
            got += len(t_.encode('utf-8'))
            yield t_
            if got >= budget_bytes:
                return


def cmd_sample(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(OLD_TOK)
    rng = random.Random(1234)
    groups = shard_groups()
    os.makedirs(a.out + '/sample', exist_ok=True)
    stats = {}
    for dom, gb in QUOTA_GB.items():
        t0 = time.time()
        budget = int(gb * 1.02e9 * float(os.environ.get('QUOTA_SCALE', '1')))
        if dom in groups:
            it = docs_from_shards(groups[dom], tok, budget, rng)
        else:
            it = reasoning_rows(budget, rng, 'reasoning' if dom == 'reasoning' else 'chat')
        n = nb = 0
        with open(f'{a.out}/sample/{dom}.train.txt', 'w') as ftr, open(f'{a.out}/sample/{dom}.heldout.txt', 'w') as fho:
            for t in it:
                t = t.replace('\x00', '')
                rec = json.dumps(t, ensure_ascii=False) + '\n'          # one JSON string per line keeps newlines
                (fho if rng.random() < 0.02 else ftr).write(rec)
                n += 1; nb += len(t)
        stats[dom] = dict(docs=n, chars=nb, seconds=round(time.time() - t0))
        print(json.dumps({dom: stats[dom]}), flush=True)
    json.dump(stats, open(f'{a.out}/sample/stats.json', 'w'), indent=1)


def text_iter(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                yield json.loads(line)


def build_tokenizer():
    from tokenizers import Tokenizer, Regex, models, normalizers, pre_tokenizers, decoders
    t = Tokenizer(models.BPE(byte_fallback=False))
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PAT), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    return t


def cmd_train(a):
    from tokenizers import trainers, pre_tokenizers, processors
    t = build_tokenizer()
    tr = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=2, special_tokens=SPECIAL, show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), max_token_length=32)
    files = sorted(glob.glob(f'{a.out}/sample/*.train.txt'))
    t0 = time.time()
    t.train_from_iterator(text_iter(files), trainer=tr)
    t.post_processor = processors.ByteLevel(trim_offsets=False)
    assert t.get_vocab_size() == VOCAB, t.get_vocab_size()
    t.save(f'{a.out}/tokenizer.json')
    print(json.dumps(dict(trained=True, vocab=t.get_vocab_size(), seconds=round(time.time() - t0))), flush=True)
    cmd_report(a)


def cmd_report(a):
    from tokenizers import Tokenizer
    toks = {'neox50k': Tokenizer.from_file(OLD_TOK), 'qwen151k': Tokenizer.from_file(QWEN_TOK),
            'new64k': Tokenizer.from_file(f'{a.out}/tokenizer.json')}
    rep = {}
    for p in sorted(glob.glob(f'{a.out}/sample/*.heldout.txt')):
        dom = os.path.basename(p).split('.')[0]
        texts = list(text_iter([p]))[:3000]
        if not texts:
            continue
        nbytes = sum(len(x.encode('utf-8')) for x in texts)
        rep[dom] = {k: round(nbytes / sum(len(e.ids) for e in t.encode_batch(texts)), 3) for k, t in toks.items()}
        print(json.dumps({dom: rep[dom]}), flush=True)
    new = toks['new64k']
    checks = ['12345678 + 98765 = 12444443', '\\frac{3}{4}x^{2} = 0.75', '    def f(x):\n        return x  # ok',
              '<|im_start|>assistant\n<think>\nLet me check.\n</think>\nThe answer is \\boxed{42}.<|im_end|>']
    rep['examples'] = {c: new.encode(c).tokens for c in checks}
    for c in checks:
        assert new.decode(new.encode(c).ids, skip_special_tokens=False) == c, c
    json.dump(rep, open(f'{a.out}/report.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(rep['examples'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['sample', 'train', 'report'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dict(sample=cmd_sample, train=cmd_train, report=cmd_report)[a.cmd](a)

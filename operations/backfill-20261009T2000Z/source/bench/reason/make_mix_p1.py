"""P1 pretraining mix for r3b_s: the retokenized moe1b mix-m1 (same sources, same shares) + FineMath-3+ (new, 64k
tokenizer, decontaminated). Writes data.txt / val.txt / mix.json into --out. Fails if any retokenized shard is missing.

  python -m bench.reason.make_mix_p1 --out /data/run01/scxi253/reason3b/mix-p1 [--finemath3-repeat 1]
"""
import argparse, glob, json, os
import numpy as np

M1 = '/data/run01/scxi253/moe1b/mix-m1'
PT = '/data/run01/scxi253/reason3b-data/pt-v1'


def new_path(p):
    src = 'web-v2' if '/web-v2/' in p else p.split('/')[-2]
    return f'{PT}/{src}/{os.path.basename(p)}', src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--finemath3-repeat', type=int, default=1)
    a = ap.parse_args()
    train = [p for p in open(f'{M1}/data.txt').read().replace('\n', ',').split(',') if p.strip()]
    val = [p for p in open(f'{M1}/val.txt').read().replace('\n', ',').split(',') if p.strip()]
    per, out_train, missing = {}, [], []
    fm = sorted(glob.glob(f'{PT}/finemath-3plus/*.u16'))
    for p in train:
        q, src = new_path(p.strip())
        if fm and src == 'finemath-4plus':
            continue                                  # FineMath-3+ is a superset of 4+: do not double-count
        if not os.path.exists(q + '.json'):
            missing.append(q); continue
        n = json.load(open(q + '.json'))['new_tokens']
        d = per.setdefault(src, dict(shards=0, tokens=0)); d['shards'] += 1; d['tokens'] += n
        out_train.append(q)
    if fm:
        n = sum(os.path.getsize(f) // 2 for f in fm)
        per['finemath-3plus'] = dict(shards=len(fm), tokens=n * a.finemath3_repeat)
        out_train += [f'{f}:{a.finemath3_repeat}' if a.finemath3_repeat > 1 else f for f in fm]
    if missing:
        raise SystemExit(f'{len(missing)} retokenized shards missing, e.g. {missing[:3]}')
    out_val = []
    for p in val:
        q, _ = new_path(p.strip())
        assert os.path.exists(q), q
        out_val.append(q)
    tot = sum(d['tokens'] for d in per.values())
    for d in per.values():
        d['tokens_B'] = round(d.pop('tokens') / 1e9, 3); d['share'] = round(d['tokens_B'] * 1e9 / tot, 4)
    os.makedirs(a.out, exist_ok=True)
    open(f'{a.out}/data.txt', 'w').write(','.join(out_train))
    open(f'{a.out}/val.txt', 'w').write(','.join(out_val))
    json.dump(dict(mix='p1', total_B=round(tot / 1e9, 2), per_source=per), open(f'{a.out}/mix.json', 'w'), indent=1)
    print(json.dumps(dict(total_B=round(tot / 1e9, 2), per_source=per), indent=1))


if __name__ == '__main__':
    main()

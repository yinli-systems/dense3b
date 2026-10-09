"""Write the v0 training mix (train/data.py spec: comma-separated paths with :R repeats) and a held-out
validation list. One shard of SwallowCode and of Stack-Edu is held out for code validation.

  python -m bench.make_mix --out /data/run01/scxi253/code3b-train/mix-v0
"""
import argparse, glob, json, os

V0 = '/data/run01/scxi253/code3b-data-20261003/v0/shards'
WEB = '/data/run01/scxi253/code3b-train/web15b'
# source -> (repeat, fraction of shards used); code ~62%, math ~25%, general web ~13% of one pass
MIX = {'swallow-code': (2, 1.0), 'stackv2-edu': (1, 1.0), 'opc-annealing': (2, 1.0),
       'nemotron-cc-math-4plus': (1, 0.5), 'swallow-math': (2, 1.0)}


def tokens(p):
    return os.path.getsize(p) // 2


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    spec, val, summary = [], [], {}
    for src, (rep, frac) in MIX.items():
        shards = sorted(glob.glob(f'{V0}/{src}/*.u16'))
        if src in ('swallow-code', 'stackv2-edu'):
            val.append(shards.pop(0))
        shards = shards[:max(1, round(len(shards) * frac))]
        spec += [f'{p}:{rep}' if rep > 1 else p for p in shards]
        summary[src] = dict(shards=len(shards), repeat=rep, unique_B=round(sum(map(tokens, shards)) / 1e9, 2),
                            seen_B=round(rep * sum(map(tokens, shards)) / 1e9, 2))
    web = sorted(glob.glob(f'{WEB}/train_*.u16'))
    spec += web
    summary['web'] = dict(shards=len(web), repeat=1, unique_B=round(sum(map(tokens, web)) / 1e9, 2),
                          seen_B=round(sum(map(tokens, web)) / 1e9, 2))
    val += sorted(glob.glob(f'{WEB}/val_*.u16'))
    total = sum(v['seen_B'] for v in summary.values())
    for v in summary.values():
        v['share'] = round(v['seen_B'] / total, 3)
    open(f'{a.out}/data.txt', 'w').write(','.join(spec))
    open(f'{a.out}/val.txt', 'w').write(','.join(val))
    json.dump(dict(per_source=summary, epoch_tokens_B=round(total, 2), val=val), open(f'{a.out}/mix.json', 'w'), indent=1)
    print(json.dumps(dict(per_source=summary, epoch_tokens_B=round(total, 2)), indent=1))


if __name__ == '__main__':
    main()

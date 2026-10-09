"""Training mix for the 1B-active MoE (general text + code + math), in train/data.py spec (paths with :R repeats).

The validation list is the 3B run's (one held-out shard each of SwallowCode and Stack-Edu, plus the web validation
shard whose documents web-v1/v2 exclude), so held-out losses are directly comparable with the 3B dense model.

  python -m bench.make_mix_moe1b --mix m0 --out /data/run01/scxi253/moe1b/mix-m0
"""
import argparse, glob, json, os

V0 = '/data/run01/scxi253/code3b-data-20261003/v0/shards'
WEBV1 = '/data/run01/scxi253/code3b-train/web-v1/shards'
WEBV2 = '/data/run01/scxi253/moe1b-data/web-v2/shards'
K1 = '/data/run01/scxi253/moe1b-data/k1/shards'
VAL_WEB = '/data/run01/scxi253/code3b-train/web15b/val_0000.u16'

# name -> list of (glob, repeat, fraction of shards used)
MIXES = {
    # start-up mix from data that is packed today: ~47% web, ~29% code, ~15% math (by tokens seen per epoch)
    'm0': {
        'web': [(f'{WEBV1}/train_*.u16', 2, 1.0)],
        'stackv2-edu': [(f'{V0}/stackv2-edu/*.u16', 1, 0.5)],
        'swallow-code': [(f'{V0}/swallow-code/*.u16', 1, 0.35)],
        'opc-annealing': [(f'{V0}/opc-annealing/*.u16', 1, 0.5)],
        'nemotron-cc-math-4plus': [(f'{V0}/nemotron-cc-math-4plus/*.u16', 1, 0.25)],
        'swallow-math': [(f'{V0}/swallow-math/*.u16', 1, 1.0)],
    },
    # general mix once web-v2 and the knowledge sources are packed: ~45% web, ~15% knowledge, ~25% code, ~15% math
    'm1': {
        'web': [(f'{WEBV2}/train_*.u16', 1, 1.0)],
        'wikipedia': [(f'{K1}/wikipedia/*.u16', 1, 1.0)],
        'arxiv': [(f'{K1}/arxiv/*.u16', 1, 0.5)],
        'stackexchange': [(f'{K1}/stackexchange/*.u16', 1, 1.0)],
        'stackv2-edu': [(f'{V0}/stackv2-edu/*.u16', 1, 0.33)],
        'swallow-code': [(f'{V0}/swallow-code/*.u16', 1, 0.3)],
        'opc-annealing': [(f'{V0}/opc-annealing/*.u16', 1, 0.42)],
        'finemath-4plus': [(f'{K1}/finemath-4plus/*.u16', 1, 0.5)],
        'nemotron-cc-math-4plus': [(f'{V0}/nemotron-cc-math-4plus/*.u16', 1, 0.13)],
        'swallow-math': [(f'{V0}/swallow-math/*.u16', 1, 0.3)],
    },
}
HELD_OUT = {'swallow-code', 'stackv2-edu'}       # first shard of each is validation (as in the 3B mix)


def tokens(p):
    return os.path.getsize(p) // 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', default='m0')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    spec, val, summary = [], [], {}
    for src, parts in MIXES[a.mix].items():
        uniq = seen = n = 0
        for pat, rep, frac in parts:
            shards = sorted(glob.glob(pat))
            if src in HELD_OUT:
                val.append(shards.pop(0))
            shards = shards[:max(1, round(len(shards) * frac))]
            spec += [f'{p}:{rep}' if rep > 1 else p for p in shards]
            t = sum(map(tokens, shards))
            uniq += t; seen += rep * t; n += len(shards)
        summary[src] = dict(shards=n, unique_B=round(uniq / 1e9, 2), seen_B=round(seen / 1e9, 2))
    val.append(VAL_WEB)
    total = sum(v['seen_B'] for v in summary.values())
    for v in summary.values():
        v['share'] = round(v['seen_B'] / total, 3)
    open(f'{a.out}/data.txt', 'w').write(','.join(spec))
    open(f'{a.out}/val.txt', 'w').write(','.join(val))
    json.dump(dict(mix=a.mix, per_source=summary, epoch_tokens_B=round(total, 2), val=val),
              open(f'{a.out}/mix.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1), 'epoch_B', round(total, 2))


if __name__ == '__main__':
    main()

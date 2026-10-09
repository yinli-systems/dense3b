"""Dense counterfactual: are MoE expert partitions the natural clusters of the dense twin's representation space?

For every pair of probe outputs at the same step (MoE twin, Dense twin; same probe tokens by construction), and every
layer: NMI and ARI between the MoE top-1 expert and the Dense k-means cluster of the FFN input, plus two references:
  * moe_geometry: NMI(MoE's own k-means, MoE route)  -- how much routing is "just geometry" inside the MoE
  * chance: NMI against a shuffled assignment
  python -m bench.routing.compare MOE_PROBE_DIR DENSE_PROBE_DIR --out compare.json
"""
import argparse, glob, json, os
import numpy as np
from bench.routing.probe import nmi


def ari(a, b):
    a, b = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    n = len(a); c = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(c, (a, b), 1)
    comb = lambda x: x * (x - 1) / 2
    s, sa, sb = comb(c).sum(), comb(c.sum(1)).sum(), comb(c.sum(0)).sum()
    exp = sa * sb / comb(n)
    return float((s - exp) / (0.5 * (sa + sb) - exp)) if (sa + sb) / 2 != exp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('moe_dir'); ap.add_argument('dense_dir'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    step = lambda p: json.load(open(p))['step']
    md = {step(p): p for p in glob.glob(os.path.join(a.moe_dir, '*.json'))}
    dd = {step(p): p for p in glob.glob(os.path.join(a.dense_dir, '*.json'))}
    rng = np.random.default_rng(0); out = []
    for s in sorted(set(md) & set(dd)):
        zm, zd = np.load(md[s].replace('.json', '.npz')), np.load(dd[s].replace('.json', '.npz'))
        row = dict(step=s, layers={})
        for key in zm.files:
            if not key.startswith('top1_l'):
                continue
            li = key[len('top1_l'):]
            route, dk, mk = zm[key], zd[f'km_l{li}'], zm[f'km_l{li}']
            row['layers'][li] = dict(dense_nmi=nmi(dk, route), dense_ari=ari(dk, route), moe_geometry_nmi=nmi(mk, route),
                                     chance_nmi=nmi(rng.permutation(dk), route))
        out.append(row)
    json.dump(out, open(a.out, 'w'), indent=1)
    for r in out:
        v = list(r['layers'].values())
        print(r['step'], {k: round(float(np.mean([x[k] for x in v])), 4) for k in v[0]})


if __name__ == '__main__':
    main()

"""Compare a v2 canary parquet against the v1 parquet for the same task, row by row."""
import json, sys, glob
import numpy as np, pyarrow.parquet as pq
root, sid, rng = sys.argv[1], sys.argv[2], sys.argv[3]
V1 = '/data/run01/scxi253/moe100b-neural-score-20261002T1550Z'
v1 = json.load(open(f'{V1}/scores/{sid}/{rng}/COMPLETE.json'))['artifact']['path']
v2 = sorted(glob.glob(f'{root}/canary/{sid}/{rng}/attempt-*/scores.parquet'))[-1]
a = pq.read_table(v1).to_pydict(); b = pq.read_table(v2).to_pydict()
n = len(b['ordinal']); assert a['ordinal'][:n] == b['ordinal'], 'ordinal mismatch'
exact = ['source_text_tokens', 'text_hash', 'status', 'classifier_tokens', 'covered_classifier_tokens',
         'head_start', 'middle_start', 'tail_start']
res = dict(rows=n, exact_mismatches={k: sum(x != y for x, y in zip(a[k][:n], b[k])) for k in exact})
for k in ['head_score', 'middle_score', 'tail_score', 'mean_score', 'min_score']:
    x = np.array([v if v is not None else np.nan for v in a[k][:n]], float)
    y = np.array([v if v is not None else np.nan for v in b[k]], float)
    d = np.abs(x - y)[~np.isnan(x)]
    res[k] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()), p999=float(np.quantile(d, .999)))
xs = np.array([v for v in a['mean_score'][:n] if v is not None]); ys = np.array([v for v in b['mean_score'] if v is not None])
res['threshold_flips_at_2.5_3.0'] = [int(((xs >= t) != (ys >= t)).sum()) for t in (2.5, 3.0)]
print(json.dumps(res, indent=1))

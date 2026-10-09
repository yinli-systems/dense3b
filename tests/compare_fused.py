"""Compare fused canary chunks against the completed v1 near and benchmark chunks, row by row."""
import collections, glob, json, sys
import pyarrow.parquet as pq

root = sys.argv[1]
V1 = '/data/run01/scxi253/moe100b-parallel-quality-20261002T1701Z/output'
res = collections.Counter(); dclm = []
for receipt in sorted(glob.glob(f'{root}/canary/*/*/COMPLETE.json')):
    r = json.load(open(receipt)); i = r['identity']; rng = '%09d-%09d' % (i['start'], i['stop'])
    f = pq.read_table(r['artifact']['path']).to_pydict()
    near = pq.read_table(json.load(open(f"{V1}/near/{i['source_id']}/{rng}/COMPLETE.json"))['artifact']['path']).to_pydict()
    bench = pq.read_table(json.load(open(f"{V1}/benchmark/{i['source_id']}/{rng}/COMPLETE.json"))['artifact']['path']).to_pydict()
    assert f['ordinal'] == near['ordinal'] == bench['ordinal'], 'ordinal mismatch ' + receipt
    n = len(f['ordinal']); res['tasks'] += 1; res['rows'] += n
    for k in range(n):
        res['signature_mismatch'] += f['signature'][k] != near['signature'][k]
        res['bands_mismatch'] += (f['bands'][k] or []) != (near['bands'][k] or [])
        res['units_mismatch'] += f['units'][k] != near['units'][k]
        if f['status'][k] == 'SCANNED':
            res['near_status_mismatch'] += f['near_status'][k] != near['status'][k]
            res['benchmark_status_mismatch'] += f['benchmark_status'][k] != bench['status'][k]
            res['evidence_mismatch'] += f['evidence_json'][k] != bench['evidence_json'][k]
            dclm.append(f['dclm_hq_prob'][k])
        else:
            res['unscanned_status_mismatch'] += not (near['status'][k] == bench['status'][k] == f['status'][k])
dclm.sort()
q = lambda p: round(dclm[int(p * (len(dclm) - 1))], 4) if dclm else None
print(json.dumps(dict(res, dclm_scored=len(dclm), dclm_quantiles={p: q(p) for p in (0.1, 0.5, 0.9, 0.99)}), indent=1))

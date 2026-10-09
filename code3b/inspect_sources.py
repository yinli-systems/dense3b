"""Print file count, size, columns and one example row for every downloaded 3B code source."""
import glob, gzip, json, os, sys
import pyarrow.parquet as pq

SRCS = {
    'swallow-code': 'tokyotech-llm__swallow-code/train/*.jsonl',
    'swallow-math': 'tokyotech-llm__swallow-math/**/*.jsonl',
    'opc-annealing': 'OpenCoder-LLM__opc-annealing-corpus/**/*.jsonl',
    'stackv2-edu': 'common-pile__stackv2_edu_filtered/**/*.gz',
    'nemotron-cc-math': 'nvidia__Nemotron-CC-Math-v1/**/*.parquet',
    'opc-fw-code': 'OpenCoder-LLM__opc-fineweb-code-corpus/**/*.parquet',
    'opc-fw-math': 'OpenCoder-LLM__opc-fineweb-math-corpus/**/*.parquet',
    'starcoderdata': 'bigcode__starcoderdata/**/*.parquet',
}


def first_row(f):
    if f.endswith('.parquet'):
        pf = pq.ParquetFile(f)
        return pf.schema_arrow.names, pf.metadata.num_rows, pf.read_row_group(0).slice(0, 1).to_pylist()[0]
    op = gzip.open if f.endswith('.gz') else open
    with op(f, 'rt') as fh:
        r = json.loads(fh.readline())
    return list(r), '?', r


def main(root):
    os.chdir(root)
    for k, pat in SRCS.items():
        fs = sorted(f for f in glob.glob(pat, recursive=True) if '/.cache/' not in f)
        if not fs:
            print('==', k, 'NO FILES'); continue
        dirs = sorted({os.path.dirname(f).split('/', 1)[-1] for f in fs})
        cols, n, r = first_row(fs[0])
        ex = {c: (v[:80] + '...' if isinstance(v, str) and len(v) > 80 else str(v)[:80]) for c, v in r.items()}
        print(f'== {k}: files={len(fs)} GB={sum(map(os.path.getsize, fs)) / 1e9:.1f} rows_first={n} dirs={dirs[:10]}')
        print('   cols:', cols)
        print('   ex:', json.dumps(ex, ensure_ascii=False)[:700])


if __name__ == '__main__':
    main(sys.argv[1])

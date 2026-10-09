import time, subprocess, sys, os, tempfile
import bench.eval_suite as S
hp = S.load_jsonl('humanevalplus.jsonl')
def ck(r):
    prog = r['prompt'] + r['canonical_solution'] + '\n\n' + r['test'] + f'\n\ncheck({r["entry_point"]})\n'
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'p.py'); open(p, 'w').write(prog); t = time.time()
        try:
            rr = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=120, cwd=d)
            return r['task_id'], rr.returncode == 0, round(time.time() - t, 1), rr.stderr.strip().split('\n')[-1][:120]
        except subprocess.TimeoutExpired:
            return r['task_id'], False, 120, 'TIMEOUT'
res = S.pmap(ck, hp)
print('he+ with evalplus canonical:', sum(x[1] for x in res), '/', len(res))
for x in res:
    if not x[1] or x[2] > 20: print(x)

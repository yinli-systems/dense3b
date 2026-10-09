"""Sanity check of eval_suite scoring with reference solutions (should be ~100%)."""
import json, sys
import bench.eval_suite as S

he = S.load_jsonl('humaneval.jsonl'); hp = {r['task_id']: r for r in S.load_jsonl('humanevalplus.jsonl')}
def he_ck(r):
    prog = r['prompt'] + r['canonical_solution']
    return (S.execute(prog + '\n\n' + r['test'] + f'\n\ncheck({r["entry_point"]})\n', 15)[0],
            S.execute(prog + '\n\n' + hp[r['task_id']]['test'] + f'\n\ncheck({r["entry_point"]})\n', 60)[0])
res = S.pmap(he_ck, he)
print('humaneval oracle', sum(a for a, _ in res), sum(b for _, b in res), '/', len(he),
      [r['task_id'] for r, (a, b) in zip(he, res) if not (a and b)][:10])
mb = S.load_jsonl('mbppplus.jsonl')
def mb_ck(r):
    imports = '\n'.join(eval(r['test_imports']) if isinstance(r['test_imports'], str) else r['test_imports'])
    tl = r['test_list'] if isinstance(r['test_list'], list) else eval(r['test_list'])
    return (S.execute(imports + '\n' + r['code'] + '\n\n' + '\n'.join(tl) + '\n', 15)[0],
            S.execute(imports + '\n' + r['code'] + '\n\n' + r['test'] + '\n', 60)[0])
res = S.pmap(mb_ck, mb)
print('mbpp oracle', sum(a for a, _ in res), sum(b for _, b in res), '/', len(mb),
      [r['task_id'] for r, (a, b) in zip(mb, res) if not (a and b)][:10])
cr = S.load_jsonl('cruxeval.jsonl')
o = S.pmap(lambda r: S.execute(f'{r["code"]}\nassert f({r["input"]}) == {r["output"]}\n', 5)[0], cr)
i = S.pmap(lambda r: S.execute(f'{r["code"]}\nassert {r["output"]} == f({r["input"]})\n', 5)[0], cr)
print('cruxeval oracle O', sum(o), 'I', sum(i), '/', len(cr))
ex = S._fim_examples()
f = S.pmap(lambda e: S.execute(e[1]['prompt'] + e[2] + e[3] + e[4] + '\n\n' + e[1]['test'] + f'\n\ncheck({e[1]["entry_point"]})\n', 15)[0], ex)
print('fim oracle', sum(f), '/', len(ex))
rows = S.load_jsonl('lcb/test6.jsonl')
print('lcb v6 tests decodable', sum(1 for r in rows if S._lcb_tests(r)), '/', len(rows),
      'functional', sum(1 for r in rows if S._lcb_tests(r)[0]['testtype'] == 'functional'))

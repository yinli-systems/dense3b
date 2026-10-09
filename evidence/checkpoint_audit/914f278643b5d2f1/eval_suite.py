"""Base-model capability suite for dense3b checkpoints (one GPU, greedy decoding, executed tests).

Tasks (all data verified on disk and decontaminated out of the training corpus with 13-gram matching):
  humaneval   HumanEval (164) + HumanEval+ (EvalPlus tests) from one set of completions
  mbpp        MBPP (sanitised prompts from MBPP+, 378) + MBPP+ tests
  cruxeval    CRUXEval-O and CRUXEval-I (800 each), 2-shot
  lcb         LiveCodeBench code generation; headline = release v6 (contests 2025-01-04..2025-04-06), after the
              corpus crawl cutoff; also v1-v5 for reference
  fim         HumanEval-based infilling: single-line and multi-line spans of the canonical solution (328),
              executed, plus termination metrics (stops with EOS, suffix duplication, over-generation)
  gsm8k       GSM8K test (1319), 4-shot chain of thought, answer extraction
  heldout     per-source held-out loss (SwallowCode, Stack-Edu, web)

Protocols are our own (documented in each function) and are NOT identical to every official harness; numbers are
for tracking this run across checkpoints. Absolute comparisons with published models need the official harnesses.

  python -m bench.eval_suite CKPT_DIR --out OUT.json [--tasks humaneval,mbpp,...] [--batch 16]
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import pickle
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
import zlib

import torch

from model.config import get_config
from model.generate import greedy
from model.model import Transformer
from train.data import PackedTokens, shard_paths

TOK = '/data/run01/scxi253/moe7b-quality-stage2-20261002T2327Z/tokenizer.json'
J = '/data/run01/scxi253/code3b-data-20261003/bench/jsonl'
EOS, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, BAN = 50279, 50280, 50281, 50282, 50280
VAL = {'swallow-code': '/data/run01/scxi253/code3b-data-20261003/v0/shards/swallow-code/swallow-code__01a89e93621f3962.u16',
       'stackv2-edu': '/data/run01/scxi253/code3b-data-20261003/v0/shards/stackv2-edu/stackv2-edu__006e6aa034e0c6a2.u16',
       'web': '/data/run01/scxi253/code3b-train/web15b/val_0000.u16'}
CODE_STOP = ['\ndef ', '\nclass ', '\nif __name__', '\nprint(', '\n#', '\nassert']


# ------------------------------------------------------------------ infrastructure

class Gen:
    def __init__(self, model, tok, batch):
        self.m, self.tok, self.batch = model, tok, batch

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def run(self, prompts, max_new, stops=(), raw_ids=False):
        """prompts: list of str (or id lists if raw_ids). Returns (texts, finished_with_eos)."""
        ids = prompts if raw_ids else [self.enc(p) for p in prompts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        texts, eos_flags = [None] * len(ids), [None] * len(ids)
        for k in range(0, len(order), self.batch):
            idx = order[k:k + self.batch]
            def stop_fn(b, gen, idx=idx):
                if not stops:
                    return False
                t = self.tok.decode(gen)
                return any(s in t for s in stops)
            outs = greedy(self.m, [ids[i][-3500:] for i in idx], max_new, EOS, stop_fn=stop_fn, ban_from=BAN)
            for i, o in zip(idx, outs):
                t = self.tok.decode(o)
                cut = min((t.find(s) for s in stops if s in t), default=-1)
                texts[i] = t[:cut] if cut >= 0 else t
                eos_flags[i] = len(o) < max_new and cut < 0
        return texts, eos_flags


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))


def execute(code, timeout=10, stdin=None):
    """Run code in a fresh interpreter in a temp dir. -> (ok, stdout)."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'prog.py')
        open(p, 'w').write(code)
        try:
            r = subprocess.run([sys.executable, p], input=stdin, capture_output=True, text=True, timeout=timeout,
                               cwd=d, preexec_fn=_limits, start_new_session=True, env=dict(PATH=os.environ.get('PATH', ''), OMP_NUM_THREADS='1'))
            return r.returncode == 0, r.stdout
        except subprocess.TimeoutExpired:
            return False, ''


def pmap(fn, items, workers=6):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


LIMIT = 0          # smoke tests: first N rows of every file


def load_jsonl(name):
    rows = []
    for l in open(os.path.join(J, name)):
        rows.append(json.loads(l))
        if LIMIT and len(rows) >= LIMIT:
            break
    return rows


# ------------------------------------------------------------------ tasks

def task_humaneval(g):
    """Completion of the prompt; stop at the next top-level statement. HumanEval tests and EvalPlus tests."""
    base = load_jsonl('humaneval.jsonl')
    plus = {r['task_id']: r for r in load_jsonl('humanevalplus.jsonl')}
    comps, _ = g.run([r['prompt'] for r in base], 512, CODE_STOP)
    def check(args):
        r, c = args
        prog = r['prompt'] + c
        ok = execute(prog + '\n\n' + r['test'] + f'\n\ncheck({r["entry_point"]})\n', 15)[0]
        if r['task_id'] == 'HumanEval/32':     # EvalPlus input format for this task needs harness special-casing
            return ok, ok
        okp = ok and execute(prog + '\n\n' + plus[r['task_id']]['test'] + f'\n\ncheck({r["entry_point"]})\n', 60)[0]
        return ok, okp
    res = pmap(check, list(zip(base, comps)))
    n = len(base)
    return dict(humaneval=round(sum(a for a, _ in res) / n, 4), humaneval_plus=round(sum(b for _, b in res) / n, 4), n=n,
                details=[dict(task=r['task_id'], base=a, plus=b, completion=c) for r, c, (a, b) in zip(base, comps, res)])


def task_mbpp(g):
    """EvalPlus base-model prompt: docstring with the task and one example assert, then the model writes the code.
    MBPP = the original 3 asserts; MBPP+ = EvalPlus tests."""
    rows = load_jsonl('mbppplus.jsonl')
    prompts = []
    for r in rows:
        tl = r['test_list'] if isinstance(r['test_list'], list) else eval(r['test_list'])
        prompts.append(f'"""\n{r["prompt"]}\n{tl[0]}\n"""\n')
    comps, _ = g.run(prompts, 512, ['\nassert', '\n"""', '\nprint(', '\nif __name__', '\n\n\n\n'])
    def check(args):
        r, c = args
        imports = '\n'.join(eval(r['test_imports']) if isinstance(r['test_imports'], str) else r['test_imports'])
        tl = r['test_list'] if isinstance(r['test_list'], list) else eval(r['test_list'])
        ok = execute(imports + '\n' + c + '\n\n' + '\n'.join(tl) + '\n', 15)[0]
        okp = ok and execute(imports + '\n' + c + '\n\n' + r['test'] + '\n', 60)[0]
        return ok, okp
    res = pmap(check, list(zip(rows, comps)))
    n = len(rows)
    return dict(mbpp=round(sum(a for a, _ in res) / n, 4), mbpp_plus=round(sum(b for _, b in res) / n, 4), n=n,
                details=[dict(task=r['task_id'], base=a, plus=b, completion=c) for r, c, (a, b) in zip(rows, comps, res)])


CRUX_SHOTS = [
    ('def f(s):\n    return s[::-1] + s', '"ab"', '"baab"'),
    ('def f(xs):\n    total = 0\n    for x in xs:\n        if x % 2 == 0:\n            total += x\n    return total', '[1, 2, 3, 4]', '6'),
]


def task_cruxeval(g):
    """2-shot. Output prediction: complete `assert f(<input>) == `. Input prediction: complete `assert f(` such
    that the given output results. Scored by executing the completed assertion."""
    rows = load_jsonl('cruxeval.jsonl')
    shot_o = ''.join(f'{c}\nassert f({i}) == {o}\n\n' for c, i, o in CRUX_SHOTS)
    shot_i = ''.join(f'{c}\nassert {o} == f({i})\n\n' for c, i, o in CRUX_SHOTS)
    po = [shot_o + f'{r["code"]}\nassert f({r["input"]}) == ' for r in rows]
    pi = [shot_i + f'{r["code"]}\nassert {r["output"]} == f(' for r in rows]
    out_o, _ = g.run(po, 96, ['\n'])
    out_i, _ = g.run(pi, 96, ['\n'])
    def ck_o(a):
        r, pred = a
        return execute(f'{r["code"]}\nassert f({r["input"]}) == {pred.strip()}\n', 5)[0]
    def ck_i(a):
        r, pred = a
        pred = pred.strip()
        return execute(f'{r["code"]}\nassert {r["output"]} == f({pred}\n', 5)[0] if pred.endswith(')') else False
    so, si = pmap(ck_o, list(zip(rows, out_o))), pmap(ck_i, list(zip(rows, out_i)))
    return dict(cruxeval_o=round(sum(so) / len(rows), 4), cruxeval_i=round(sum(si) / len(rows), 4), n=len(rows))


def _lcb_tests(r):
    pub = json.loads(r['public_test_cases'])
    priv = r['private_test_cases']
    try:
        priv = json.loads(priv)
    except Exception:
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv.encode()))))
    return pub + priv


def _lcb_check(r, code):
    tests = _lcb_tests(r)
    if not code.strip():
        return False
    if tests and tests[0]['testtype'] == 'functional':
        fn = json.loads(r['metadata'])['func_name']
        runner = code + '\n\nimport json, sys\n' \
            f'_tests = {json.dumps([(t["input"], t["output"]) for t in tests])}\n' \
            'for _inp, _out in _tests:\n' \
            '    _args = [json.loads(x) for x in _inp.split("\\n") if x.strip()]\n' \
            f'    _got = Solution().{fn}(*_args)\n' \
            '    if _got != json.loads(_out):\n        sys.exit(1)\n'
        return execute(runner, 30)[0]
    for t in tests:
        ok, out = execute(code, 8, stdin=t['input'])
        if not ok or out.split() != t['output'].split():
            return False
    return True


def task_lcb(g):
    """Zero-shot. Prompt: problem statement, format instruction (stdin/stdout or starter code), then an opened
    ```python block; generation stops at the closing fence. All public + private tests must pass."""
    rows = []
    for v in range(1, 7):
        name = 'test.jsonl' if v == 1 else f'test{v}.jsonl'
        for r in load_jsonl(os.path.join('lcb', name)):
            r['release'] = v
            rows.append(r)
    prompts = []
    for r in rows:
        if r['starter_code']:
            fmt = f'You will use the following starter code to write the solution and enclose your code within delimiters.\n```python\n{r["starter_code"]}\n```'
        else:
            fmt = 'Read the inputs from stdin, solve the problem and write the answer to stdout.'
        prompts.append(f'### Question:\n{r["question_content"]}\n\n### Format: {fmt}\n\n### Answer:\n```python\n')
    comps, _ = g.run(prompts, 768, ['```'])
    res = pmap(lambda a: _lcb_check(*a), list(zip(rows, comps)), workers=6)
    by = {}
    for r, ok in zip(rows, res):
        by.setdefault(r['release'], []).append(ok)
    out = {f'lcb_v{v}': round(sum(x) / len(x), 4) for v, x in sorted(by.items())}
    out['lcb_v6_post_cutoff'] = out.get('lcb_v6')
    out['lcb_all'] = round(sum(res) / len(res), 4)
    out['n'] = {f'v{v}': len(x) for v, x in sorted(by.items())}
    return out


def _fim_examples():
    rows = load_jsonl('humaneval.jsonl')
    ex = []
    for k, r in enumerate(rows):
        body = r['canonical_solution'].split('\n')
        nz = [i for i, l in enumerate(body) if l.strip()]
        rng = random.Random(1000 + k)
        if not nz:
            continue
        i = rng.choice(nz)                                         # single line
        ex.append(('single', r, '\n'.join(body[:i]) + '\n', body[i] + '\n', '\n'.join(body[i + 1:])))
        if len(nz) >= 3:                                           # multi-line block of 2-4 lines
            a = rng.randrange(len(nz) - 1)
            b = min(len(nz) - 1, a + rng.randint(1, 3))
            i0, i1 = nz[a], nz[b]
            ex.append(('multi', r, '\n'.join(body[:i0]) + '\n', '\n'.join(body[i0:i1 + 1]) + '\n', '\n'.join(body[i1 + 1:])))
    return ex


def task_fim(g):
    """PSM infilling of HumanEval canonical solutions. pass = program with the generated middle passes the tests.
    Termination: the model ends the middle with EOS within budget; suffix_dup = it re-emits the suffix's first line."""
    ex = _fim_examples()
    prompts, budgets = [], []
    for kind, r, pre, mid, suf in ex:
        prompts.append([FIM_PREFIX] + g.enc(r['prompt'] + pre) + [FIM_SUFFIX] + g.enc(suf) + [FIM_MIDDLE])
    gens, eos = g.run(prompts, 192, (), raw_ids=True)
    def check(a):
        (kind, r, pre, mid, suf), gen = a
        prog = r['prompt'] + pre + gen + suf
        return execute(prog + '\n\n' + r['test'] + f'\n\ncheck({r["entry_point"]})\n', 15)[0]
    passed = pmap(check, list(zip(ex, gens)))
    stats = {}
    for (kind, r, pre, mid, suf), gen, ok, e in zip(ex, gens, passed, eos):
        s = stats.setdefault(kind, dict(n=0, pass_=0, exact=0, eos=0, suffix_dup=0, overgen_chars=0))
        first_suf = next((l for l in suf.split('\n') if l.strip()), None)
        s['n'] += 1; s['pass_'] += ok; s['exact'] += gen.strip() == mid.strip(); s['eos'] += bool(e)
        s['suffix_dup'] += bool(first_suf and first_suf.strip() in gen)
        s['overgen_chars'] += max(0, len(gen) - len(mid))
    out = {}
    for kind, s in stats.items():
        n = s['n']
        out[kind] = dict(n=n, pass_rate=round(s['pass_'] / n, 4), exact=round(s['exact'] / n, 4),
                         terminated=round(s['eos'] / n, 4), suffix_dup=round(s['suffix_dup'] / n, 4),
                         mean_overgen_chars=round(s['overgen_chars'] / n, 1))
    return out


GSM_SHOTS = [
    ('Tom has 3 boxes with 12 apples each. He gives away 10 apples. How many apples does he have left?',
     'Tom starts with 3 * 12 = 36 apples. After giving away 10 he has 36 - 10 = 26 apples. The answer is 26.'),
    ('A train travels 60 miles per hour for 2.5 hours. How many miles does it travel?',
     'Distance is speed times time: 60 * 2.5 = 150 miles. The answer is 150.'),
    ('Sara buys 4 notebooks at $3 each and a pen for $2. She pays with a $20 bill. How much change does she get?',
     'The notebooks cost 4 * 3 = 12 dollars. With the pen the total is 12 + 2 = 14 dollars. Her change is 20 - 14 = 6 dollars. The answer is 6.'),
    ('A class has 30 students. Two fifths of them walk to school and the rest take the bus. How many take the bus?',
     'Two fifths of 30 is 30 * 2 / 5 = 12 students who walk. So 30 - 12 = 18 take the bus. The answer is 18.'),
]


def task_gsm8k(g):
    """4-shot chain of thought (our own exemplars); answer = number after 'The answer is', else the last number."""
    rows = load_jsonl('gsm8k_test.jsonl')
    shots = ''.join(f'Question: {q}\nAnswer: {a}\n\n' for q, a in GSM_SHOTS)
    outs, _ = g.run([shots + f'Question: {r["question"]}\nAnswer:' for r in rows], 320, ['\nQuestion:', '\n\n'])
    ok = 0
    for r, o in zip(rows, outs):
        gold = r['answer'].split('####')[-1].strip().replace(',', '')
        m = re.search(r'The answer is\s*\$?(-?[\d,]*\.?\d+)', o)
        nums = [m.group(1)] if m else re.findall(r'-?\d[\d,]*\.?\d*', o)
        pred = nums[-1].replace(',', '').rstrip('.') if nums else None
        try:
            ok += pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            pass
    return dict(gsm8k=round(ok / len(rows), 4), n=len(rows))


@torch.no_grad()
def task_heldout(g, n_windows=64):
    out = {}
    for name, path in VAL.items():
        ds = PackedTokens(shard_paths(path), 4096, seed=12345)
        tot = cnt = 0.0
        for i in range(n_windows):
            w = torch.from_numpy(ds.window(i)).cuda()[None]
            x, y = w[:, :-1], w[:, 1:].clone()
            y[y == 1] = -100
            n = int((y != -100).sum())
            tot += g.m(x, y)['ce'].item() * n; cnt += n
        out[name] = round(tot / cnt, 4)
    return out


TASKS = dict(heldout=task_heldout, humaneval=task_humaneval, mbpp=task_mbpp, cruxeval=task_cruxeval, fim=task_fim,
             gsm8k=task_gsm8k, lcb=task_lcb)


def load(ckpt):
    meta = json.load(open(os.path.join(ckpt, 'meta.json')))
    cfg = get_config(meta['args']['preset'], **dict(json.loads(meta['args']['override']), doc_mask=False))
    m = Transformer(cfg)
    sd = torch.load(os.path.join(ckpt, 'model.pt'), map_location='cpu', weights_only=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k not in ('cos', 'sin')] and not unexpected, (missing, unexpected)
    m = m.cuda().bfloat16().eval()
    m.cos, m.sin = m.cos.float(), m.sin.float()
    return m, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N rows per data file')
    a = ap.parse_args(argv)
    global LIMIT
    LIMIT = a.limit
    from tokenizers import Tokenizer
    m, meta = load(a.ckpt)
    g = Gen(m, Tokenizer.from_file(TOK), a.batch)
    res = dict(ckpt=a.ckpt, step=meta['step'], tokens_B=round((meta['step'] + 1) * 256 * 4096 / 1e9, 2), tasks={})
    details = {}
    for t in a.tasks.split(','):
        t0 = time.time()
        r = TASKS[t](g)
        if isinstance(r, dict) and 'details' in r:
            details[t] = r.pop('details')
        r['seconds'] = round(time.time() - t0)
        res['tasks'][t] = r
        print(json.dumps({t: r}), flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)
    json.dump(details, open(a.out.replace('.json', '.details.json'), 'w'))
    print('DONE', a.out, flush=True)


if __name__ == '__main__':
    main()

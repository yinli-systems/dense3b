"""Normalize, license-filter, quality-filter and decontaminate the post-training / reasoning mid-training corpora.

  python -m bench.reason.build_reason_data evalindex              # 8-gram index of every eval set we report
  python -m bench.reason.build_reason_data normalize --task I --ntasks N   # one Slurm array task
  python -m bench.reason.build_reason_data summary

Output: $ROOT/norm/<source>/part-<task>.jsonl.gz, one record per (prompt, response):
  {"id", "src", "domain", "gen", "lic", "messages": [{"role", "content"}...], "think": bool,
   "answer": str|null, "verified": bool|null}
The final assistant turn of a reasoning sample is "<think>\\n...\\n</think>\\n\\n<final answer>"; a non-thinking sample
has an empty think block (one model serves both modes, as in Qwen3).

License policy (user requirement: permissive only). Kept: Apache-2.0 / MIT / CC-BY-4.0 / ODC-BY data whose responses
come from permissively licensed teachers (DeepSeek-R1 / R1-0528 (MIT), QwQ-32B / Qwen2.5 / Qwen3 (Apache-2.0)).
Dropped: responses from Llama-family generators (Llama licence attaches naming/attribution terms to derived models),
prompts sourced from lmsys-chat-1m (its own licence agreement), tulu-3 subsets whose responses come from OpenAI models,
and anything tagged non-commercial.
"""
import argparse, glob, gzip, hashlib, json, os, re, sys, time, zlib
from collections import Counter, defaultdict

ROOT = '/data/run01/scxi253/reason3b-data'
RAW = ROOT + '/raw'
J = '/data/run01/scxi253/code3b-data-20261003/bench/jsonl'
MC = '/data/run01/scxi253/moe1b-data/mc/raw'
N = 8                       # n-gram length for decontamination (Tulu 3 uses 8; prompts are short)
CONTAM_FRAC = 0.5           # contaminated if >= 50 % of an eval item's 8-grams occur in the training prompt
MIN_NGRAMS = 3              # eval items need >= 3 distinct 8-grams (>= 10 words) to be indexed
MAX_CHARS = 140_000         # ~ 32k tokens

# ----------------------------------------------------------------------------- readers

def read_rows(path, columns=None):
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            for r in pf.read_row_group(i, columns=columns).to_pylist():
                yield r
    elif path.endswith('.jsonl') or path.endswith('.jsonl.gz'):
        op = gzip.open if path.endswith('.gz') else open
        with op(path, 'rt') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.endswith('.json'):
        d = json.load(open(path))
        yield from (d if isinstance(d, list) else d.get('data', []))
    elif path.endswith('.csv'):
        import csv
        csv.field_size_limit(1 << 30)
        with open(path, newline='') as f:
            yield from csv.DictReader(f)


def words(t):
    return re.findall(r'\w+', t.lower())


def ngram_hashes(t):
    w = words(t)
    if len(w) < N:
        return {h64(' '.join(w))} if w else set()
    return {h64(' '.join(w[i:i + N])) for i in range(len(w) - N + 1)}


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')

# ----------------------------------------------------------------------------- eval sets

TEXT_KEYS = ('problem', 'question', 'Question', 'prompt', 'question_content', 'ctx', 'sentence', 'question_stem',
             'input', 'code')


def eval_items():
    """(bench, text) for every benchmark we report or might report. Text = the part a model sees."""
    srcs = {
        'humaneval': [f'{J}/humaneval.jsonl'], 'mbpp': [f'{J}/mbppplus.jsonl'], 'cruxeval': [f'{J}/cruxeval.jsonl'],
        'lcb': glob.glob(f'{J}/lcb/test*.jsonl'), 'gsm8k': [f'{J}/gsm8k_test.jsonl'],
        'math500': glob.glob(f'{RAW}/HuggingFaceH4__MATH-500/*.jsonl'),
        'aime24': glob.glob(f'{RAW}/HuggingFaceH4__aime_2024/**/*.parquet', recursive=True),
        'aime25': glob.glob(f'{RAW}/math-ai__aime25/**/*.jsonl', recursive=True),
        'amc23': glob.glob(f'{RAW}/math-ai__amc23/**/*.jsonl', recursive=True) + glob.glob(f'{RAW}/math-ai__amc23/*.parquet'),
        'mmlu_pro': glob.glob(f'{RAW}/TIGER-Lab__MMLU-Pro/data/test-*.parquet'),
        'gpqa': glob.glob(f'{RAW}/Idavidrein__gpqa/*.csv'),
        'mmlu': glob.glob(f'{MC}/cais__mmlu/all/test-*.jsonl'),
        'arc': glob.glob(f'{MC}/allenai__ai2_arc/*/test-*.jsonl'),
        'hellaswag': glob.glob(f'{MC}/Rowan__hellaswag/data/validation-*.jsonl'),
        'winogrande': glob.glob(f'{MC}/allenai__winogrande/*/validation-*.jsonl'),
        'boolq': glob.glob(f'{MC}/google__boolq/data/validation-*.jsonl'),
        'obqa': glob.glob(f'{MC}/allenai__openbookqa/main/test-*.jsonl'),
        'sciq': glob.glob(f'{MC}/allenai__sciq/data/test-*.jsonl'),
    }
    seen = set()
    for bench, files in srcs.items():
        n = 0
        for f in files:
            for r in read_rows(f):
                t = next((str(r[k]) for k in TEXT_KEYS if k in r and r[k]), None)
                if bench == 'gpqa':
                    t = r.get('Question')
                if bench == 'humaneval' or bench == 'mbpp':
                    t = r.get('prompt') or r.get('text') or t
                if bench == 'cruxeval':
                    t = r.get('code')
                if not t or len(t) < 20:
                    continue
                key = (bench, t[:200])
                if key in seen:
                    continue
                seen.add(key); n += 1
                yield bench, t
        print(json.dumps(dict(eval_bench=bench, items=n, files=len(files))), flush=True)


def cmd_evalindex(a):
    items, index, short = [], defaultdict(list), {}
    for bench, t in eval_items():
        hs = ngram_hashes(t)
        if len(hs) < MIN_NGRAMS:                         # < 10 words: only an exact (word-normalized) match counts
            if len(words(t)) >= 3:
                short[' '.join(words(t))] = bench
            continue
        k = len(items)
        items.append(dict(bench=bench, n=len(hs), head=t[:120]))
        for h in hs:
            index[h].append(k)
    os.makedirs(f'{ROOT}/decontam', exist_ok=True)
    import pickle
    pickle.dump(dict(items=items, index=dict(index), short=short), open(f'{ROOT}/decontam/eval_index.pkl', 'wb'))
    print(json.dumps(dict(items=len(items), ngrams=len(index), short_exact=len(short),
                          per_bench=Counter(i['bench'] for i in items))), flush=True)


class Decontam:
    def __init__(self):
        import pickle
        d = pickle.load(open(f'{ROOT}/decontam/eval_index.pkl', 'rb'))
        self.items, self.index, self.short = d['items'], d['index'], d.get('short', {})

    def check(self, text):
        if self.short:
            w = ' '.join(words(text))
            if w in self.short:
                return self.short[w]
        hits = Counter()
        for h in ngram_hashes(text):
            for k in self.index.get(h, ()):
                hits[k] += 1
        best = None
        for k, c in hits.items():
            if c >= max(MIN_NGRAMS, CONTAM_FRAC * self.items[k]['n']) and (best is None or c > hits[best]):
                best = k
        return None if best is None else self.items[best]['bench']

# ----------------------------------------------------------------------------- answer checking (light math-verify)

def last_boxed(s):
    i = s.rfind('\\boxed')
    if i < 0:
        i = s.rfind('\\fbox')
        if i < 0:
            return None
    j = s.find('{', i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == '{':
            depth += 1
        elif s[k] == '}':
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def norm_ans(x):
    if x is None:
        return None
    x = str(x).strip().strip('$').strip()
    x = re.sub(r'\\text\{([^}]*)\}', r'\1', x)
    for a, b in (('\\left', ''), ('\\right', ''), ('\\dfrac', '\\frac'), ('\\tfrac', '\\frac'), ('\\!', ''),
                 ('\\,', ''), ('\\;', ''), ('\\ ', ''), ('^\\circ', ''), ('^{\\circ}', ''), ('\\%', ''), ('%', ''),
                 ('$', ''), (' ', ''), ('dollars', ''), ('units', '')):
        x = x.replace(a, b)
    x = x.rstrip('.').lower()
    x = re.sub(r'^(x|y|n|k|a|m|answer)=', '', x)
    m = re.fullmatch(r'\\frac\{?(-?\d+)\}?\{?(\d+)\}?', x)
    if m:
        x = f'{m.group(1)}/{m.group(2)}'
    try:
        if '/' in x and re.fullmatch(r'-?\d+/\d+', x):
            p, q = x.split('/')
            return repr(round(int(p) / int(q), 9))
        return repr(round(float(x.replace(',', '')), 9))
    except ValueError:
        return x


def answers_match(resp_final, expected):
    b = last_boxed(resp_final)
    return b is not None and norm_ans(b) == norm_ans(expected)

# ----------------------------------------------------------------------------- quality filters

CJK = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]')


def quality_reason(prompt, resp, think):
    if not resp or len(resp) < 2:
        return 'empty'
    if len(prompt) + len(resp) > MAX_CHARS:
        return 'too_long'
    if think and ('</think>' not in resp or resp.count('<think>') != 1):
        return 'think_incomplete'
    if think and len(resp.split('</think>', 1)[1].strip()) < 1:
        return 'no_final'
    pc, rc = len(CJK.findall(prompt)), len(CJK.findall(resp))
    if pc < 0.01 * max(1, len(prompt)) and rc > 0.005 * len(resp):
        return 'lang_mix'
    if len(resp) > 6000 and len(zlib.compress(resp.encode(), 1)) < 0.10 * len(resp.encode()):
        return 'looping'
    return None


def wrap_think(reasoning, final):
    return f'<think>\n{reasoning.strip()}\n</think>\n\n{final.strip()}'


def split_think(content):
    """content with an inline <think>...</think> -> (True, normalized); without -> (False, empty-think wrapped)."""
    c = content.strip()
    if c.startswith('<think>') and '</think>' in c:
        r, f = c[len('<think>'):].split('</think>', 1)
        if not r.strip():
            return False, wrap_think('', f)
        return True, wrap_think(r, f)
    if '<think>' in c:
        return True, c                                   # malformed: quality_reason rejects it
    return False, wrap_think('', c)

# ----------------------------------------------------------------------------- source adapters
# Each yields dicts: prompt messages (list), response (str), think (bool), gen, lic, domain, answer, verified, sid.

def msgs_from(conv):
    out = []
    for m in conv:
        role = m.get('role') or {'human': 'user', 'gpt': 'assistant', 'system': 'system'}.get(m.get('from'), m.get('from'))
        out.append(dict(role=role, content=str(m.get('content') if 'content' in m else m.get('value', ''))))
    return out


def ad_openthoughts3(path):
    for r in read_rows(path):
        m = msgs_from(r['conversations'])
        if len(m) < 2 or m[-1]['role'] != 'assistant':
            continue
        think, resp = split_think(m[-1]['content'])
        yield dict(prompt=m[:-1], response=resp, think=think, gen='QwQ-32B', lic='apache-2.0',
                   domain=r.get('domain') or 'mixed', answer=None, verified=None, sid=r.get('source'))


def ad_openr1_math(path):
    for r in read_rows(path):
        gens, ok = r.get('generations') or [], r.get('correctness_math_verify') or []
        complete = r.get('is_reasoning_complete') or [True] * len(gens)
        for k, g in enumerate(gens):
            if k < len(ok) and ok[k] and (k >= len(complete) or complete[k]):
                think, resp = split_think(g)
                yield dict(prompt=[dict(role='user', content=r['problem'])], response=resp, think=think,
                           gen='DeepSeek-R1', lic='apache-2.0', domain='math', answer=r.get('answer'), verified=True,
                           sid=r.get('source'))


def ad_openmathreasoning(path):
    for r in read_rows(path):
        gen = r.get('generation_model') or ''
        if 'llama' in gen.lower():
            continue
        think, resp = split_think(r['generated_solution'])
        yield dict(prompt=[dict(role='user', content=r['problem'])], response=resp, think=think, gen=gen,
                   lic='cc-by-4.0', domain='math', answer=r.get('expected_answer'), verified=None,
                   sid=r.get('problem_source'))


def ad_opencodereasoning(path):
    for r in read_rows(path):
        if not r.get('input') or r['input'] == '-':
            continue                                     # split_1 rows point back to external datasets; skip
        think, resp = split_think(r['output'])
        yield dict(prompt=[dict(role='user', content=r['input'])], response=resp, think=think, gen='DeepSeek-R1',
                   lic='cc-by-4.0', domain='code', answer=None, verified=None, sid=r.get('source'))


def ad_nemotron_v1(path):
    for r in read_rows(path):
        gen = r.get('generator') or ''
        if 'llama' in gen.lower():
            continue
        meta = r.get('metadata') or ''
        if 'lmsys' in str(meta).lower():
            continue
        m = r['messages']
        m = json.loads(m) if isinstance(m, str) else m
        m = [dict(role=x['role'], content=x.get('content') or '', reasoning=x.get('reasoning_content')) for x in m]
        if len(m) < 2 or m[-1]['role'] != 'assistant' or not m[-2]['content'] and m[-2]['role'] == 'user':
            continue
        last = m[-1]
        if last.get('reasoning'):
            think, resp = True, wrap_think(last['reasoning'], last['content'])
        else:
            think, resp = split_think(last['content'])
        prompt = [dict(role=x['role'], content=x['content']) for x in m[:-1]]
        yield dict(prompt=prompt, response=resp, think=think, gen=gen, lic='cc-by-4.0',
                   domain=r.get('category') or 'mixed', answer=None, verified=None, sid='nemotron-v1')


def ad_llama_nemotron(path):
    cat = path.split('/')[-2]
    for r in read_rows(path):
        gen = str(r.get('generator') or '')
        if any(x in gen.lower() for x in ('llama', 'nemotron')) or 'nc' in str(r.get('license', '')).lower():
            continue
        prompt = msgs_from(r['input']) if isinstance(r.get('input'), list) else [dict(role='user', content=str(r.get('input')))]
        if r.get('system_prompt') and r['system_prompt'] not in ('detailed thinking on', 'detailed thinking off'):
            prompt = [dict(role='system', content=r['system_prompt'])] + prompt
        think, resp = split_think(str(r.get('output') or ''))
        yield dict(prompt=prompt, response=resp, think=think, gen=gen, lic='cc-by-4.0', domain=cat, answer=None,
                   verified=None, sid='llama-nemotron')


def ad_synthetic1(path):
    for r in read_rows(path):
        m = r['messages']
        m = json.loads(m) if isinstance(m, str) else m
        m = msgs_from(m)
        if len(m) < 2 or m[-1]['role'] != 'assistant':
            continue
        if r.get('score') is not None and float(r['score']) < 1:
            continue
        think, resp = split_think(m[-1]['content'])
        yield dict(prompt=m[:-1], response=resp, think=think, gen='DeepSeek-R1', lic='apache-2.0',
                   domain=r.get('task_type') or 'mixed', answer=None, verified=r.get('score') is not None, sid=None)


def ad_codeforces_cots(path):
    for r in read_rows(path):
        m = r.get('messages')
        if not m:
            continue
        m = msgs_from(json.loads(m) if isinstance(m, str) else m)
        if len(m) < 2 or m[-1]['role'] != 'assistant':
            continue
        if r.get('finish_reason') not in (None, 'stop'):
            continue
        think, resp = split_think(m[-1]['content'])
        yield dict(prompt=m[:-1], response=resp, think=think, gen='DeepSeek-R1', lic='cc-by-4.0', domain='code',
                   answer=None, verified=None, sid='codeforces')


TULU_KEEP = ('oasst1', 'flan', 'sciriff', 'table_gpt', 'aya', 'coconot', 'hard_coded', 'evol_codealpaca_heval_decontaminated_off')


def ad_tulu3(path):
    for r in read_rows(path):
        src = r.get('source') or ''
        if not any(k in src.lower() for k in TULU_KEEP[:-1]):
            continue
        m = msgs_from(r['messages'])
        if len(m) < 2 or m[-1]['role'] != 'assistant':
            continue
        yield dict(prompt=m[:-1], response=wrap_think('', m[-1]['content']), think=False, gen=src, lic='odc-by',
                   domain='chat', answer=None, verified=None, sid=src)


SOURCES = {   # name -> (glob under RAW, adapter)
    'openthoughts3': ('open-thoughts__OpenThoughts3-1.2M/data/*.parquet', ad_openthoughts3),
    'openr1_math': ('open-r1__OpenR1-Math-220k/data/*.parquet', ad_openr1_math),
    'openmathreasoning': ('nvidia__OpenMathReasoning/data/cot-*.parquet', ad_openmathreasoning),
    'opencodereasoning': ('nvidia__OpenCodeReasoning/split_0/*.parquet', ad_opencodereasoning),
    'nemotron_v1': ('nvidia__Nemotron-Post-Training-Dataset-v1/data/*.parquet', ad_nemotron_v1),
    'llama_nemotron': ('nvidia__Llama-Nemotron-Post-Training-Dataset/*/*/*.jsonl', ad_llama_nemotron),
    'synthetic1': ('PrimeIntellect__SYNTHETIC-1-SFT-Data/data/*.parquet', ad_synthetic1),
    'codeforces_cots': ('open-r1__codeforces-cots/*decontaminated/*.parquet', ad_codeforces_cots),
    'tulu3': ('allenai__tulu-3-sft-mixture/data/*.parquet', ad_tulu3),
}


def work_items():
    items = []
    for name, (pat, _) in SOURCES.items():
        for f in sorted(glob.glob(f'{RAW}/{pat}')):
            items.append((os.path.getsize(f), name, f))
    items.sort(key=lambda x: (-x[0], x[2]))
    return items


def cmd_normalize(a):
    items = work_items()
    mine = [it for i, it in enumerate(items) if i % a.ntasks == a.task]
    dc = Decontam()
    for _, name, f in mine:
        tag = hashlib.md5(f.encode()).hexdigest()[:10]
        out = f'{ROOT}/norm/{name}/part-{tag}.jsonl.gz'
        if os.path.exists(out + '.done'):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        st, t0 = Counter(), time.time()
        with gzip.open(out + '.tmp', 'wt', compresslevel=3) as fo:
            for x in SOURCES[name][1](f):
                st['seen'] += 1
                ptxt = '\n'.join(m['content'] for m in x['prompt'] if m['role'] != 'system')
                why = quality_reason(ptxt, x['response'], x['think'])
                if why is None and x['domain'] == 'math' and x['answer'] and x['verified'] is None:
                    final = x['response'].split('</think>', 1)[-1]
                    x['verified'] = answers_match(final, x['answer'])
                    if not x['verified'] and last_boxed(final) is not None and norm_ans(x['answer']) is not None \
                            and re.fullmatch(r"-?[\d.]+'?", norm_ans(x['answer']) or ''):
                        why = 'wrong_answer'                     # numeric answers: confident mismatch -> drop
                if why is None:
                    bench = dc.check(ptxt)
                    if bench:
                        why = 'contam_' + bench
                if why:
                    st[why] += 1
                    continue
                st['kept'] += 1
                rid = hashlib.blake2b((ptxt + '\x00' + x['response']).encode(), digest_size=10).hexdigest()
                rec = dict(id=rid, src=name, sid=x['sid'], domain=x['domain'], gen=x['gen'], lic=x['lic'],
                           messages=x['prompt'] + [dict(role='assistant', content=x['response'])], think=x['think'],
                           answer=x['answer'], verified=x['verified'],
                           phash=hashlib.blake2b(' '.join(words(ptxt)).encode(), digest_size=8).hexdigest())
                fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
        os.replace(out + '.tmp', out)
        json.dump(dict(file=f, seconds=round(time.time() - t0), **st), open(out + '.done', 'w'))
        print(json.dumps(dict(src=name, file=os.path.basename(f), seconds=round(time.time() - t0), **st)), flush=True)


def cmd_summary(a):
    tot = defaultdict(Counter)
    for d in glob.glob(f'{ROOT}/norm/*/*.done'):
        s = json.load(open(d))
        name = d.split('/')[-2]
        for k, v in s.items():
            if isinstance(v, int) and k != 'seconds':
                tot[name][k] += v
    out = {k: dict(v) for k, v in sorted(tot.items())}
    json.dump(out, open(f'{ROOT}/norm/summary.json', 'w'), indent=1)
    for k, v in out.items():
        print(k, json.dumps(v))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['evalindex', 'normalize', 'summary', 'list'])
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    a = ap.parse_args()
    if a.cmd == 'list':
        for s, n, f in work_items():
            print(n, round(s / 1e9, 2), f)
    else:
        dict(evalindex=cmd_evalindex, normalize=cmd_normalize, summary=cmd_summary)[a.cmd](a)

"""Evaluate a train_dense checkpoint on one GPU: per-source held-out loss, sample generations (code, FIM, math),
HumanEval pass@1 (greedy, executed), GSM8K accuracy (greedy, 8-shot-free answer extraction on a subset).

  python -m bench.eval_ckpt CKPT_DIR --out results.json [--humaneval-n 164] [--gsm8k-n 200]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from model.config import get_config
from model.model import Transformer
from train.data import PackedTokens, shard_paths

TOK = '/data/run01/scxi253/moe7b-quality-stage2-20261002T2327Z/tokenizer.json'
BENCH = '/data/run01/scxi253/code3b-data-20261003/bench/jsonl'   # converted from the verified parquet files
EOS, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX = 50279, 50280, 50281, 50282
VAL = {'swallow-code': '/data/run01/scxi253/code3b-data-20261003/v0/shards/swallow-code/swallow-code__01a89e93621f3962.u16',
       'stackv2-edu': '/data/run01/scxi253/code3b-data-20261003/v0/shards/stackv2-edu/stackv2-edu__006e6aa034e0c6a2.u16',
       'web': '/data/run01/scxi253/code3b-train/web15b/val_0000.u16'}
HE_STOP = ['\ndef ', '\nclass ', '\nif __name__', '\nprint(', '\n#', '\nassert']


def load(ckpt):
    meta = json.load(open(os.path.join(ckpt, 'meta.json')))
    ov = json.loads(meta['args']['override'])
    cfg = get_config(meta['args']['preset'], **dict(ov, doc_mask=False))   # single documents: plain causal SDPA
    m = Transformer(cfg)
    sd = torch.load(os.path.join(ckpt, 'model.pt'), map_location='cpu', weights_only=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if k not in ('cos', 'sin')] and not unexpected, (missing, unexpected)
    m = m.cuda().bfloat16().eval()
    m.cos, m.sin = m.cos.float(), m.sin.float()
    return m, meta


@torch.no_grad()
def heldout(m, n_windows=64):
    out = {}
    for name, path in VAL.items():
        ds = PackedTokens(shard_paths(path), 4096, seed=12345)
        tot, cnt = 0.0, 0
        for i in range(n_windows):
            w = torch.from_numpy(ds.window(i)).cuda()[None]
            x, y = w[:, :-1], w[:, 1:].clone()
            y[y == 1] = -100
            n = int((y != -100).sum())
            tot += m(x, y)['ce'].item() * n; cnt += n
        out[name] = round(tot / cnt, 4)
    return out


@torch.no_grad()
def generate(m, tok, ids, max_new=256, stop=()):
    """Greedy decoding (no KV cache; prompts are short). Stops at EOS or any stop string in the completion."""
    ids = list(ids)
    start = len(ids)
    for k in range(max_new):
        logits = m(torch.tensor([ids[-4096:]], device='cuda'))[0, -1]
        logits[50280:] = -float('inf')                 # FIM sentinels / padding ids are never outputs
        nxt = int(logits.argmax())
        if nxt == EOS:
            break
        ids.append(nxt)
        if stop and k % 8 == 7:
            text = tok.decode(ids[start:])
            cut = min((text.find(s) for s in stop if s in text), default=-1)
            if cut >= 0:
                return text[:cut]
    text = tok.decode(ids[start:])
    if stop:
        cut = min((text.find(s) for s in stop if s in text), default=-1)
        text = text[:cut] if cut >= 0 else text
    return text


def run_python(code, timeout=10):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code); p = f.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, timeout=timeout, text=True)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(p)


def humaneval(m, tok, n):
    rows = [json.loads(l) for l in open(f'{BENCH}/humaneval.jsonl')][:n]
    passed, samples = 0, []
    for r in rows:
        comp = generate(m, tok, tok.encode(r['prompt'], add_special_tokens=False).ids, 384, HE_STOP)
        prog = r['prompt'] + comp + '\n\n' + r['test'] + f'\n\ncheck({r["entry_point"]})\n'
        ok = run_python(prog)
        passed += ok
        if len(samples) < 3:
            samples.append(dict(task=r['task_id'], passed=ok, completion=comp[:600]))
    return dict(pass_at_1=round(passed / len(rows), 4), passed=passed, n=len(rows), samples=samples)


def gsm8k(m, tok, n):
    rows = [json.loads(l) for l in open(f'{BENCH}/gsm8k_test.jsonl')][:n]
    correct, samples = 0, []
    for r in rows:
        prompt = f"Question: {r['question']}\nAnswer: Let's think step by step."
        out = generate(m, tok, tok.encode(prompt, add_special_tokens=False).ids, 256, ['\nQuestion:', '\n\n\n'])
        gold = r['answer'].split('####')[-1].strip().replace(',', '')
        nums = re.findall(r'-?\d[\d,]*\.?\d*', out)
        pred = nums[-1].replace(',', '').rstrip('.') if nums else None
        ok = pred is not None and pred == gold
        correct += ok
        if len(samples) < 2:
            samples.append(dict(gold=gold, pred=pred, output=out[:500]))
    return dict(accuracy=round(correct / len(rows), 4), correct=correct, n=len(rows), samples=samples)


def samples(m, tok):
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    out = {}
    out['python_function'] = generate(m, tok, enc('def is_prime(n: int) -> bool:\n    """Return True if n is a prime number."""\n'), 160, HE_STOP)
    out['binary_search'] = generate(m, tok, enc('def binary_search(arr, target):\n'), 200, HE_STOP)
    pre = 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n    if n < 2:\n        return n\n'
    suf = '\n\nprint(fibonacci(10))\n'
    out['fim_middle'] = generate(m, tok, [FIM_PREFIX] + enc(pre) + [FIM_SUFFIX] + enc(suf) + [FIM_MIDDLE], 120)
    out['math'] = generate(m, tok, enc('Question: A store sells pencils for $0.25 each. How much do 12 pencils cost?\n'
                                       "Answer: Let's think step by step."), 120, ['\nQuestion:'])
    out['prose'] = generate(m, tok, enc('The main difference between a list and a tuple in Python is'), 80, ['\n\n'])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--humaneval-n', type=int, default=164)
    ap.add_argument('--gsm8k-n', type=int, default=200)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    t0 = time.time()
    m, meta = load(a.ckpt)
    res = dict(ckpt=a.ckpt, step=meta['step'], tokens_B=round((meta['step'] + 1) * 256 * 4096 / 1e9, 2))
    res['heldout_loss'] = heldout(m); print(json.dumps(res), flush=True)
    res['samples'] = samples(m, tok); print(json.dumps(res['samples'], ensure_ascii=False, indent=1), flush=True)
    res['humaneval'] = humaneval(m, tok, a.humaneval_n); print(json.dumps(res['humaneval'])[:300], flush=True)
    res['gsm8k'] = gsm8k(m, tok, a.gsm8k_n); print(json.dumps(res['gsm8k'])[:300], flush=True)
    res['seconds'] = round(time.time() - t0)
    json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print('DONE', a.out, flush=True)


if __name__ == '__main__':
    main()

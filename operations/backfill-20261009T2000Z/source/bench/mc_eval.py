"""General-text multiple-choice benchmarks by log-likelihood ranking (lm-eval-harness conventions), for both the
3B dense and the 1B-active MoE checkpoints:

  arc_easy, arc_challenge, openbookqa, sciq   question -> " {choice}"          acc, acc_norm (per-byte)
  hellaswag                                   ctx -> " {ending}"                acc, acc_norm
  winogrande                                  partial scoring of the suffix    acc
  boolq                                       passage/question -> " yes"/" no"  acc
  mmlu_mc5                                    5-shot, letters " A".." D"        acc   (standard MMLU)
  mmlu_cf                                     0-shot cloze, " {choice}"         acc, acc_norm (OLMo-style MMLU-CF)

  python -m bench.mc_eval CKPT --out res.json [--tasks ...] [--limit N]
"""
import argparse, json, math, os, re, time
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from bench.eval_suite import load, TOK

R = '/data/run01/scxi253/moe1b-data/mc/raw'
EOS = 50279


def rows(path):
    """Benchmarks are pre-converted parquet -> jsonl (the training env has no pyarrow)."""
    with open(path.replace('.parquet', '.jsonl')) as f:
        return [json.loads(l) for l in f]


def hs_pre(t):
    t = t.strip().replace(' [title]', '. ')
    t = re.sub(r'\[.*?\]', '', t)
    return t.replace('  ', ' ')


def build(task, limit=0):
    """-> list of (context, [continuations], gold, kind) ; kind 'cont' (shared context) or 'wino' (per-option ctx)."""
    out = []
    if task in ('arc_easy', 'arc_challenge'):
        sub = 'ARC-Easy' if task == 'arc_easy' else 'ARC-Challenge'
        for r in rows(f'{R}/allenai__ai2_arc/{sub}/test-00000-of-00001.parquet'):
            labels = list(r['choices']['label'])
            if r['answerKey'] not in labels:
                continue
            out.append((f"Question: {r['question']}\nAnswer:", [' ' + c for c in r['choices']['text']],
                        labels.index(r['answerKey']), 'cont'))
    elif task == 'openbookqa':
        for r in rows(f'{R}/allenai__openbookqa/main/test-00000-of-00001.parquet'):
            labels = list(r['choices']['label'])
            out.append((r['question_stem'], [' ' + c for c in r['choices']['text']], labels.index(r['answerKey']), 'cont'))
    elif task == 'sciq':
        for r in rows(f'{R}/allenai__sciq/data/test-00000-of-00001.parquet'):
            ch = [r['distractor1'], r['distractor2'], r['distractor3'], r['correct_answer']]
            out.append((f"{r['support'].strip()}\nQuestion: {r['question']}\nAnswer:".lstrip(), [' ' + c for c in ch], 3, 'cont'))
    elif task == 'hellaswag':
        for r in rows(f'{R}/Rowan__hellaswag/data/validation-00000-of-00001.parquet'):
            ctx = r['ctx_a'] + ' ' + r['ctx_b'].capitalize()
            out.append((hs_pre(r['activity_label'] + ': ' + ctx), [' ' + hs_pre(e) for e in r['endings']], int(r['label']), 'cont'))
    elif task == 'winogrande':
        for r in rows(f'{R}/allenai__winogrande/winogrande_xl/validation-00000-of-00001.parquet'):
            s = r['sentence']; i = s.index('_')
            ctxs = [s[:i] + r['option1'], s[:i] + r['option2']]
            cont = ' ' + s[i + 1:].strip()
            out.append((ctxs, [cont, cont], int(r['answer']) - 1, 'wino'))
    elif task == 'boolq':
        for r in rows(f'{R}/google__boolq/data/validation-00000-of-00001.parquet'):
            out.append((f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", [' no', ' yes'], int(bool(r['answer'])), 'cont'))
    elif task in ('mmlu_mc5', 'mmlu_cf'):
        dev = {}
        for r in rows(f'{R}/cais__mmlu/all/dev-00000-of-00001.parquet'):
            dev.setdefault(r['subject'], []).append(r)
        L = 'ABCD'
        def q_mc(r, ans=True):
            s = r['question'].strip() + '\n' + ''.join(f'{L[j]}. {c}\n' for j, c in enumerate(r['choices'])) + 'Answer:'
            return s + (f' {L[r["answer"]]}\n\n' if ans else '')
        for r in rows(f'{R}/cais__mmlu/all/test-00000-of-00001.parquet'):
            subj = r['subject'].replace('_', ' ')
            if task == 'mmlu_mc5':
                head = f'The following are multiple choice questions (with answers) about {subj}.\n\n'
                ctx = head + ''.join(q_mc(d) for d in dev[r['subject']][:5]) + q_mc(r, ans=False)
                out.append((ctx, [' A', ' B', ' C', ' D'], r['answer'], 'cont'))
            else:
                out.append((f"Question: {r['question'].strip()}\nAnswer:", [' ' + c for c in r['choices']], r['answer'], 'cont'))
    if limit:
        out = out[:limit]
    return out


@torch.no_grad()
def score(model, tok, reqs, max_len=2048, batch_tokens=32768):
    """reqs: list of (ctx_ids, cont_ids) -> list of (sum logprob, n tokens)."""
    order = sorted(range(len(reqs)), key=lambda i: -(len(reqs[i][0]) + len(reqs[i][1])))
    res = [None] * len(reqs)
    i = 0
    while i < len(order):
        L = min(max_len, len(reqs[order[i]][0]) + len(reqs[order[i]][1]))
        bs = max(1, batch_tokens // L)
        idx = order[i:i + bs]; i += bs
        seqs, spans = [], []
        for j in idx:
            c, x = reqs[j]
            s = (c + x)[-max_len:]
            n = min(len(x), len(s) - 1)
            seqs.append(s); spans.append(n)
        T = max(map(len, seqs))
        toks = torch.full((len(seqs), T), EOS, dtype=torch.long)
        for b, s in enumerate(seqs):
            toks[b, :len(s)] = torch.tensor(s)
        toks = toks.cuda()
        x = model.embed(toks)
        for blk in model.layers:
            x, _ = blk(x, model.cos, model.sin, None)
        h = model.head.norm(x)
        for b, s in enumerate(seqs):
            n = spans[b]; Ls = len(s)
            lp = F.log_softmax(F.linear(h[b, Ls - 1 - n:Ls - 1], model.head.weight.to(h.dtype)).float(), -1)
            tgt = torch.tensor(s[Ls - n:], device=lp.device)
            res[idx[b]] = (lp.gather(1, tgt[:, None]).sum().item(), n)
    return res


def run_task(model, tok, task, limit):
    items = build(task, limit)
    enc = lambda s: tok.encode(s).ids
    reqs, meta = [], []
    for k, (ctx, conts, gold, kind) in enumerate(items):
        for j, c in enumerate(conts):
            cx = ctx[j] if kind == 'wino' else ctx
            reqs.append((enc(cx), enc(c))); meta.append((k, j, len(c.encode('utf-8'))))
    sc = score(model, tok, reqs)
    ll = {}
    for (k, j, nbytes), (lp, n) in zip(meta, sc):
        ll.setdefault(k, {})[j] = (lp, lp / max(1, nbytes))
    acc = np.mean([max(ll[k], key=lambda j: ll[k][j][0]) == items[k][2] for k in ll])
    accn = np.mean([max(ll[k], key=lambda j: ll[k][j][1]) == items[k][2] for k in ll])
    return dict(acc=round(float(acc), 4), acc_norm=round(float(accn), 4), n=len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', default='arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa,sciq,mmlu_cf,mmlu_mc5')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    model, meta = load(a.ckpt)
    tok = Tokenizer.from_file(meta.get('tokenizer', TOK))
    res = dict(checkpoint=a.ckpt, step=meta['step'], preset=meta['args']['preset'], tasks={})
    for t in a.tasks.split(','):
        t0 = time.time()
        res['tasks'][t] = dict(run_task(model, tok, t, a.limit), seconds=round(time.time() - t0))
        print(json.dumps({t: res['tasks'][t]}), flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)
    print('DONE', a.out)


if __name__ == '__main__':
    main()

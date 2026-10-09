"""GPU check of model/engine.py: (1) prefill+decode logits == full forward logits, (2) greedy Engine == greedy()
token-for-token (with refill: more prompts than slots), (3) decode throughput of r3b_s at 32 rows x 4k context."""
import json, time, torch
from dataclasses import replace
from model.config import get_config
from model.model import Transformer, init_weights
from model.engine import Engine
from model.generate import greedy


def build(c):
    m = Transformer(c); init_weights(m, 0)
    m = m.cuda().to(torch.bfloat16); m.cos, m.sin = m.cos.float(), m.sin.float(); m.eval()
    with torch.no_grad():                                   # sharper logits than init so argmax is not a coin flip
        m.head.weight.mul_(8)
    return m


@torch.no_grad()
def full_logits(m, ids):
    t = torch.tensor(ids, device='cuda')[None]
    x = m.embed(t)
    for blk in m.layers:
        x, _ = blk(x, m.cos, m.sin, None)
    return torch.nn.functional.linear(m.head.norm(x[0]), m.head.weight.to(x.dtype)).float()


def main():
    torch.manual_seed(0)
    c = replace(get_config('r3b_s'), n_layers=6, max_seq_len=8192)
    m = build(c)
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(64, 60000, (n,), generator=g).tolist() for n in (37, 5, 120, 64, 9, 300, 17)]
    eng = Engine(m, max_rows=3, max_len=1024)
    # (1) logits equivalence along a greedy continuation of prompt 0
    ids = list(prompts[0]); lg = eng.prefill(0, ids); errs = []
    for t in range(24):
        ref = full_logits(m, ids)[-1]
        errs.append(float((lg - ref).abs().max() / ref.abs().max()))
        nxt = int(lg.argmax()); ids.append(nxt)
        lg = eng.decode(torch.tensor([0], device='cuda'), torch.tensor([nxt], device='cuda'))[0]
    # (2) greedy equivalence incl. slot refill
    a = eng.generate(prompts, max_new=32, temperature=0)
    b = greedy(m, prompts, 32, eos=-1)
    agree = [sum(x == y for x, y in zip(r['ids'], o)) / 32 for r, o in zip(a, b)]
    prefix = [next((i for i, (x, y) in enumerate(zip(r['ids'], o)) if x != y), 32) for r, o in zip(a, b)]
    # (3) throughput, full r3b_s
    del m, eng; torch.cuda.empty_cache()
    m = build(replace(get_config('r3b_s'), max_seq_len=17408))
    res = {}
    for rows, ctx in ((32, 4096), (64, 2048), (16, 16384)):
        eng = Engine(m, max_rows=rows, max_len=ctx + 600)
        ps = [torch.randint(64, 60000, (ctx,), generator=g).tolist() for _ in range(rows)]
        for s, p in enumerate(ps):
            eng.prefill(s, p)
        rr = torch.arange(rows, device='cuda'); tok = torch.zeros(rows, dtype=torch.long, device='cuda')
        for _ in range(5):
            eng.decode(rr, tok)
        torch.cuda.synchronize(); t0 = time.time(); n = 64
        for _ in range(n):
            tok = eng.decode(rr, tok).argmax(-1)
        torch.cuda.synchronize(); dt = (time.time() - t0) / n
        res[f'rows{rows}_ctx{ctx}'] = dict(ms_per_step=round(dt * 1e3, 2), tok_per_s=round(rows / dt),
                                           mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        del eng; torch.cuda.empty_cache()
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), logit_rel_err_max=max(errs), greedy_agree=agree,
                          first_divergence=prefix, decode=res), indent=1))


if __name__ == '__main__':
    main()

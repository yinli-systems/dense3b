"""Convert the RunPod 7B MoE (OLMo-core export, w1/w3 split, fp32 safetensors) into this codebase's checkpoint
layout (model.pt in bf16 + meta.json), so the same evaluation suites run on all three models.

  python -m bench.convert_olmo_moe7b SRC.safetensors OUT_DIR --step 69000 --tokenizer TOKENIZER.json

Mapping (E=64 experts, I=1024 expert width, H=2048):
  embeddings.weight                      -> embed.weight (tied with head.weight)
  blocks.i.attention.w_{q,k,v,out}       -> layers.i.attn.w{q,k,v,o}.weight
  blocks.i.attention.{q,k}_norm          -> layers.i.attn.{q,k}_norm.weight   (full-width RMSNorm)
  blocks.i.{attention,feed_forward}_norm -> layers.i.{attn,ffn}_norm.weight
  ...feed_forward_moe.router.weight      -> layers.i.ffn.router.weight (E, H)
  ...experts.mlp.w1/w3 (E*I, H)          -> layers.i.ffn.w13 (E, H, 2I): [:, :, :I] = w1[e].T, [:, :, I:] = w3[e].T
  ...experts.mlp.w2 (E*I, H)             -> layers.i.ffn.w2 (E, I, H)
  ...shared_mlp.w{1,2,3}.weight          -> layers.i.ffn.shared.w{1,2,3}.weight
  lm_head.norm.weight                    -> head.norm.weight
"""
import argparse, json, os, struct
import numpy as np
import torch

from model.config import get_config
from model.model import Transformer


_DT = {'F32': np.float32, 'F16': np.float16, 'I64': np.int64, 'I32': np.int32}


class _Lazy(dict):
    """key -> loader; pop() materializes one tensor at a time in BF16, so peak RAM is ~ the BF16 model (the fp32
    export is 29 GB; materializing it all plus an fp32 model OOM-killed the 60 GB job)."""
    def pop(self, k, *default):
        if k not in self and default:
            return default[0]
        return dict.pop(self, k)()


def read_safetensors(path):
    """Minimal reader (the training env has no safetensors package): 8-byte header size, JSON header, raw data."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode='r', offset=8 + n)
    out = _Lazy()
    for k, v in header.items():
        if k == '__metadata__':
            continue
        def load(v=v):
            a, b = v['data_offsets']
            if v['dtype'] == 'BF16':
                return torch.from_numpy(np.array(mm[a:b]).view(np.int16).reshape(v['shape'])).view(torch.bfloat16)
            t = torch.from_numpy(np.array(mm[a:b]).view(_DT[v['dtype']]).reshape(v['shape']))
            return t.to(torch.bfloat16) if t.is_floating_point() else t
        out[k] = load
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--tokenizer', required=True)
    a = ap.parse_args()
    c = get_config('moe7b_olmo')
    E, I, H = c.n_experts, c.d_expert, c.d_model
    src = read_safetensors(a.src)
    sd = {'embed.weight': src.pop('embeddings.weight'), 'head.norm.weight': src.pop('lm_head.norm.weight')}
    w_out = src.pop('lm_head.w_out.weight', None)
    if w_out is not None:
        assert torch.equal(w_out, sd['embed.weight']), 'lm_head is not tied to the embeddings'
        del w_out
    for i in range(c.n_layers):
        b, o = f'blocks.{i}.', f'layers.{i}.'
        for s_, d_ in (('w_q', 'wq'), ('w_k', 'wk'), ('w_v', 'wv'), ('w_out', 'wo')):
            sd[f'{o}attn.{d_}.weight'] = src.pop(f'{b}attention.{s_}.weight')
        sd[f'{o}attn.q_norm.weight'] = src.pop(f'{b}attention.q_norm.weight')
        sd[f'{o}attn.k_norm.weight'] = src.pop(f'{b}attention.k_norm.weight')
        sd[f'{o}attn_norm.weight'] = src.pop(f'{b}attention_norm.weight')
        sd[f'{o}ffn_norm.weight'] = src.pop(f'{b}feed_forward_norm.weight')
        m = f'{b}feed_forward_moe.'
        sd[f'{o}ffn.router.weight'] = src.pop(f'{m}router.weight').view(E, H)
        w1 = src.pop(f'{m}experts.mlp.w1').view(E, I, H)
        w3 = src.pop(f'{m}experts.mlp.w3').view(E, I, H)
        sd[f'{o}ffn.w13'] = torch.cat([w1.transpose(1, 2), w3.transpose(1, 2)], dim=2).contiguous()
        sd[f'{o}ffn.w2'] = src.pop(f'{m}experts.mlp.w2').view(E, I, H).contiguous()
        for w in ('w1', 'w2', 'w3'):
            sd[f'{o}ffn.shared.{w}.weight'] = src.pop(f'{m}shared_mlp.{w}.weight')
    assert not src, f'unconverted keys: {list(src)[:10]}'
    with torch.device('meta'):
        ref = Transformer(c).state_dict()               # key/shape check without allocating an fp32 model
    sd['head.weight'] = sd['embed.weight']              # tied
    for k, v in ref.items():
        if k.endswith('router.bias') and k not in sd:
            sd[k] = torch.zeros(v.shape, dtype=torch.bfloat16)
    missing = [k for k in ref if k not in sd and k not in ('cos', 'sin')]
    unexpected = [k for k in sd if k not in ref]
    assert not missing and not unexpected, (missing, unexpected)
    bad = [k for k in sd if tuple(sd[k].shape) != tuple(ref[k].shape)]
    assert not bad, [(k, tuple(sd[k].shape), tuple(ref[k].shape)) for k in bad]
    os.makedirs(a.out, exist_ok=True)
    full = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
    torch.save(full, os.path.join(a.out, 'model.pt'))
    json.dump(dict(step=a.step, args=dict(preset='moe7b_olmo', override='{}'), config=c.to_dict(),
                   tokenizer=a.tokenizer, eos_id=c.eos_id, ban_from=151665,
                   source=dict(path=a.src, format='olmo-core export (w1/w3 split)')),
              open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'COMPLETE'), 'w').close()
    print('CONVERTED', a.out, sum(v.numel() for v in full.values()))


if __name__ == '__main__':
    main()

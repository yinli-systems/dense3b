"""Model configs and exact parameter accounting. Pure Python (no torch) so tools/budget.py can import it."""
from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class MoEConfig:
    vocab_size: int = 50304          # 50,280 padded to a multiple of 128 (ids >= 50280 never occur)
    d_model: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4               # GQA
    head_dim: int = 128
    qk_norm: bool = True
    attn_gate: bool = True            # elementwise sigmoid gate on the SDPA output (Qwen gated attention)
    n_dense_layers: int = 1           # first layers use a dense SwiGLU instead of MoE
    d_ff_dense: int = 4096
    n_experts: int = 128
    top_k: int = 8
    d_expert: int = 512
    d_shared: int = 2048              # total width of the always-on shared expert (0 = none)
    route_groups: int = 8             # expert groups (= EP ranks) for group-limited routing; 0 = off
    route_topk_groups: int = 4        # M: each token may reach at most M groups
    routed_scale: float = 1.0
    z_loss_coef: float = 1e-3
    seq_aux_coef: float = 1e-4        # tiny sequence-wise balance loss (DeepSeek-V3); bias does the real work
    bias_update_rate: float = 1e-3
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = False
    init_std: float = 0.02
    max_seq_len: int = 4096
    fuse_proj: bool = False           # one QKV(G) GEMM and one W1|W3 GEMM (Muon still orthogonalises each part)
    doc_mask: bool = False            # attention never crosses an EOS inside a packed window (flex_attention)
    eos_id: int = 50279
    qk_norm_full: bool = False        # RMSNorm over all heads' q (k) dims at once (OLMo-core default) instead of per head
    router_kind: str = 'sigmoid_bias' # 'softmax': OLMo-core softmax over experts, top-k scores as weights (no renorm)
    shared_combine: str = 'sum'       # 'average': out = shared/(k+1) + routed*k/(k+1) (OLMo-core MoE with shared mlp)

    def to_dict(self):
        return asdict(self)

    @property
    def n_moe_layers(self):
        return self.n_layers - self.n_dense_layers


PRESETS = {
    # proposed design
    'moe7b': MoEConfig(),
    # team baseline, OLMoE-1B-7B-like (arXiv 2409.02060): 64 experts x 1024, top-8, MHA, no shared expert
    'olmoe': MoEConfig(n_kv_heads=16, attn_gate=False, n_dense_layers=0, n_experts=64, d_expert=1024,
                       d_shared=0, route_groups=0, rope_theta=10_000.0),
    # 3B dense code model: Llama-3.2-3B shape (3072 x 28, GQA 24/8, SwiGLU 8192) with this tokenizer and tied
    # embeddings = 3.24B parameters. Every layer dense (n_dense_layers == n_layers), so no expert/router code runs.
    'dense3b': MoEConfig(d_model=3072, n_layers=28, n_heads=24, n_kv_heads=8, head_dim=128, n_dense_layers=28,
                         d_ff_dense=8192, tie_embeddings=True, fuse_proj=True, doc_mask=True),
    # 1B-class MoE for scattered 4090s (every rank holds all experts; measured 2026-10-07 on one 4090, model/moe_local.py):
    # 2.49B total / 0.61B active, 64 fine-grained experts x 384 (granularity 8) top-8 + one shared expert (2 experts'
    # width), FFN activation ratio 15%. 23.6k tok/s/GPU fwd+bwd at 2 x 4096 tokens (63% MFU of 165 TF).
    'moe1b': MoEConfig(d_model=1536, n_layers=20, n_heads=12, n_kv_heads=4, head_dim=128, n_dense_layers=1,
                       d_ff_dense=4096, n_experts=64, top_k=8, d_expert=384, d_shared=768, route_groups=0,
                       tie_embeddings=True, fuse_proj=True, doc_mask=True,
                       bias_update_rate=5e-3, seq_aux_coef=3e-4),   # early routing drifts fast: balance faster
    # Reasoning MoE "S" (docs/REASONING_MOE_PLAN.md): the 1B MoE's proven d1536 / 64x384 top-8 expert shape, 4 layers
    # deeper (depth helps multi-step reasoning at fixed active size), and the 65,536-entry digit-split BPE
    # (bench/reason/train_tokenizer.py; id 0 = <|endoftext|>). 3.02B total / 0.74B active.
    'r3b_s': MoEConfig(vocab_size=65536, d_model=1536, n_layers=24, n_heads=12, n_kv_heads=4, head_dim=128,
                       n_dense_layers=1, d_ff_dense=4096, n_experts=64, top_k=8, d_expert=384, d_shared=768,
                       route_groups=0, tie_embeddings=True, fuse_proj=True, doc_mask=True, eos_id=0,
                       bias_update_rate=5e-3, seq_aux_coef=3e-4),
    # Routing-causality study (docs/ROUTING_STUDY.md): three twins on the moe1b data recipe, identical attention /
    # width / depth / tokenizer / token stream; only the FFN differs.
    #   moe1b_r   1.055B total / 0.427B active: 32 experts x 256 (G=12) top-4 + shared 768, layer 0 dense
    #   dense1b_l 1.053B (= MoE total), d_ff 8704        dense1b_s 0.416B (= MoE active), d_ff 1792
    'moe1b_r': MoEConfig(d_model=1536, n_layers=20, n_heads=12, n_kv_heads=4, head_dim=128, n_dense_layers=1,
                         d_ff_dense=4096, n_experts=32, top_k=4, d_expert=256, d_shared=768, route_groups=0,
                         tie_embeddings=True, fuse_proj=True, doc_mask=True, bias_update_rate=5e-3, seq_aux_coef=3e-4),
    'dense1b_l': MoEConfig(d_model=1536, n_layers=20, n_heads=12, n_kv_heads=4, head_dim=128, n_dense_layers=20,
                           d_ff_dense=8704, route_groups=0, tie_embeddings=True, fuse_proj=True, doc_mask=True),
    'dense1b_s': MoEConfig(d_model=1536, n_layers=20, n_heads=12, n_kv_heads=4, head_dim=128, n_dense_layers=20,
                           d_ff_dense=1792, route_groups=0, tie_embeddings=True, fuse_proj=True, doc_mask=True),
    # The RunPod 7B MoE (OLMo-core llama_like_moe, Qwen2.5 tokenizer) for evaluation in this codebase:
    # 7.02B total / 1.39B active, 16 x d2048, GQA 16/4, full-width QK RMSNorm, 64 experts x1024 top-8 softmax
    # router + shared 1024 averaged with the routed output, tied embeddings, norm eps 1e-5.
    'moe7b_olmo': MoEConfig(vocab_size=151680, d_model=2048, n_layers=16, n_heads=16, n_kv_heads=4, head_dim=128,
                            qk_norm=True, qk_norm_full=True, attn_gate=False, n_dense_layers=0, n_experts=64, top_k=8,
                            d_expert=1024, d_shared=1024, route_groups=0, rope_theta=500_000.0, norm_eps=1e-5,
                            tie_embeddings=True, fuse_proj=False, doc_mask=False, eos_id=151643, max_seq_len=4096,
                            router_kind='softmax', shared_combine='average'),
    # CPU-sized configs for tests and smoke runs
    'tiny': MoEConfig(vocab_size=256, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, head_dim=16,
                      d_ff_dense=128, n_experts=8, top_k=2, d_expert=32, d_shared=32, route_groups=4,
                      route_topk_groups=2, max_seq_len=64),
}


def get_config(name, **overrides):
    return replace(PRESETS[name], **overrides)


def param_counts(c: MoEConfig):
    """Exact parameter counts of model/model.py for config c (tested against the module)."""
    d, hd = c.d_model, c.head_dim
    attn = d * c.n_heads * hd * 2 + d * c.n_kv_heads * hd * 2       # wq, wo, wk, wv
    attn += d * c.n_heads * hd if c.attn_gate else 0                 # gate projection
    attn += 2 * hd if c.qk_norm else 0
    norms = 2 * d
    expert = 3 * d * c.d_expert
    shared = 3 * d * c.d_shared
    router = c.n_experts * d
    dense_ffn = 3 * d * c.d_ff_dense
    L, Lm = c.n_layers, c.n_moe_layers
    embed = c.vocab_size * d
    head = 0 if c.tie_embeddings else c.vocab_size * d
    out = dict(
        embed=embed, head=head, final_norm=d,
        attention=L * attn, norms=L * norms,
        dense_ffn=c.n_dense_layers * dense_ffn,
        routed_experts=Lm * c.n_experts * expert,
        shared_experts=Lm * shared, router=Lm * router,
    )
    out['total'] = sum(out.values())
    active_routed = Lm * c.top_k * expert
    out['active_routed'] = active_routed
    out['active'] = out['total'] - out['routed_experts'] + active_routed     # OLMoE convention: incl. embeddings
    out['active_nonembed'] = out['active'] - embed - head
    # parameters that do matmul work per token (input embedding is a lookup; head always multiplies)
    out['active_matmul'] = out['active_nonembed'] - out['norms'] - out['final_norm'] + (head or embed)
    return out

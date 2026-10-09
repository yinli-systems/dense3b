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

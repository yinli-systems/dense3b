# MoE-7B-A1B: model and training design for 6 × 8 consumer GPUs

Status: proposal + reference implementation (`model/`, `train/`, `tools/budget.py`, `tests/moe/`).
Every number in the budget sections comes from `python tools/budget.py`. Efficiency assumptions are
labelled, and the Day-1 measurements in §11 replace them.

Citations: links marked † were recalled from the papers but not re-fetched in this session (the sandbox
proxy blocks arxiv.org). Unmarked links were confirmed via search during this session.

## 0. Summary

* **Architecture:** 16 layers, d_model 2048, GQA 16q/4kv with QK-norm and a sigmoid output gate.
  Layer 0 is dense. Layers 1–15 are MoE with **128 fine-grained experts (width 512), top-8, plus one
  shared expert (width 2048)**. Routing uses sigmoid scores, aux-loss-free bias balancing and a
  group-limited constraint (each token reaches at most 4 of the 8 EP ranks). Parameters:
  **6.70 B total / 1.04 B active** (0.83 B active non-embedding).
* **Why it should beat the OLMoE-1B-7B-like baseline on this hardware:**
  * **19 % fewer FLOPs per token**: 6.40 G vs 7.88 G.
  * **53 % fewer all-to-all bytes per token**: 0.87 MB vs 1.85 MB, from per-rank token dedup plus the
    group limit.
  * **Granularity G = 8**, inside the 8–12 optimum of the Ling MoE scaling law. OLMoE has G = 4.
  * **Lower activation ratio**: 12.8 % vs 16.0 % of non-embedding parameters. The same scaling law says
    efficiency leverage rises as the activation ratio falls.
  * **Muon instead of AdamW**: about 2× compute efficiency in the Moonlight scaling study.
* **Parallelism:** EP = FSDP-shard = the 8 GPUs of a node, replicated across the 6 nodes (HSDP).
  Expert gradients go once per step over RoCE in BF16. Global batch is 960 × 4096 = 3.93 M tokens.
* **Throughput:** the budget gives **211–262 B tokens in 15 days on 4090** and **261–319 B on 5090**.
  The target is **160 B tokens (40,700 steps)**, which leaves at least 25 % slack for failures and for
  the gap between assumed and measured efficiency. Extra throughput goes to a longer WSD stable phase,
  not a bigger model.
* **The binding constraint is communication.** Compute alone would allow more than 600 B tokens. The
  all-to-all over host-memory SHM and the per-step inter-node gradient all-reduce together take 45–51 %
  of a step. Every choice below is made with that in mind.

## 1. What the hardware allows (summary of §8)

| Quantity | RTX 4090 | RTX 5090 | Source |
|---|---|---|---|
| Dense BF16 tensor TFLOPS (FP32 accumulate) | 165.2 | 209.5 | [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), [RTX 5090 spec summary](https://www.spheron.network/blog/nvidia-rtx-5090-specs/) |
| Dense FP8 TFLOPS (FP32 accumulate) | 330.3 | 419 | same |
| Intra-node GPU↔GPU (NCCL SHM, DRAM-bound) | 15–18 GB/s | 22–27 GB/s | given constraint (budget uses the low end) |
| Inter-node per node | 2 × 25 GbE = 6.25 GB/s raw; 4.38 GB/s assumed effective (70 %, shared with JuiceFS) | same | given constraint |

GeForce parts run FP32-accumulate MMA at half the FP16-accumulate rate. 165/209 TF is therefore the
correct BF16 training ceiling, not the 330/419 TF headline numbers.

## 2. Architecture

### 2.1 Config (`model/config.py`, preset `moe7b`)

| Item | Value | Notes |
|---|---|---|
| Layers | 16 | layer 0 dense SwiGLU (d_ff 4096); layers 1–15 MoE |
| d_model | 2048 | |
| Attention | GQA, 16 query heads × 128, 4 KV heads | QK-RMSNorm per head; elementwise sigmoid gate on the SDPA output; no biases |
| Position | RoPE, θ = 500,000 | keeps later context extension cheap |
| Norm | pre-norm RMSNorm, ε = 1e-6 | final RMSNorm before the head |
| Routed experts | 128 per MoE layer, SwiGLU, d_expert = 512 | granularity G = 2·d_model/d_expert = 8 |
| Top-k | 8 routed (6.25 % of routed experts) | |
| Shared experts | 1, SwiGLU width 2048 (= 4 routed experts' worth) | always on, computed locally |
| Router | linear 2048→128 in FP32, **sigmoid** scores, top-k on score + bias, gates = scores normalised over the selected k | DeepSeek-V3 style |
| Group-limited routing | 8 groups (= EP ranks); a token may use at most M = 4 groups (top-4 by sum of top-2 scores per group) | caps all-to-all fan-out at 4 |
| Balancing | aux-loss-free bias, u = 1e-3, sign update on **global-batch** loads every optimizer step; sequence-wise aux α = 1e-4; router z-loss 1e-3 | |
| Dispatch | dropless (no capacity factor, nothing dropped); token dedup per destination rank | |
| Embeddings | untied, vocab 50,280 padded to 50,304 | |
| Init | truncated normal, std 0.02; output projections (wo, w2) std 0.02/√(2L); every expert seeded by (layer, global id) | identical weights for any EP layout |
| Loss | chunked FP32 cross-entropy (logits recomputed in backward, never stored) | |

### 2.2 Exact parameter count (tested in `tests/moe/test_model.py`)

| Component | Parameters |
|---|---|
| Embedding / LM head (untied) | 103,022,592 / 103,022,592 |
| Attention (16 layers, incl. gate and QK-norm) | 234,885,120 |
| Dense FFN (layer 0) | 25,165,824 |
| Routed experts (15 × 128 × 3·2048·512) | 6,039,797,760 |
| Shared experts (15 × 3·2048·2048) | 188,743,680 |
| Routers | 3,932,160 |
| Norms | 67,584 |
| **Total** | **6,698,637,312** |
| **Active per token** (OLMoE convention, incl. embedding and head) | **1,036,326,912** |
| Active, non-embedding | 830,281,728 |
| Training FLOPs per token (6·N_active-matmul + causal attention at 4k) | 6.40 G (5.60 G matmul + 0.81 G attention) |

The `olmoe` preset reproduces OLMoE-1B-7B's published 6.9 B total / 1.3 B active (6,919,100,416 /
1,281,955,840), which validates the counter.

### 2.3 Why each choice (evidence)

**Fine granularity, low activation ratio**
* Krajewski et al. fit scaling laws with granularity as a variable: MoE beats dense consistently, and the
  gap widens with scale. ([2402.07871](https://arxiv.org/abs/2402.07871))
* The Ling team trained 300+ models. Efficiency leverage is driven mainly by the **activation ratio**
  (lower is better, power law). Granularity G = 2·d_model/d_expert has a U-shaped optimum at
  **G ≈ 8–12**. ([2507.17702](https://arxiv.org/abs/2507.17702))
* OLMoE's 64 × 1024 experts give G = 4. Our 128 × 512 gives G = 8 at a lower activation ratio.
* We stop at G = 8 instead of 12 deliberately. Smaller experts make thinner per-expert GEMMs. With
  4,096 tokens × 8 slots spread over 16 local experts, each expert already sees only ≈2,048 rows per
  micro-batch.

**Shared expert (kept, with a Day-1 ablation)**
* Evidence for: DeepSeekMoE ([2401.06066](https://arxiv.org/abs/2401.06066)†) and Ling
  ([2507.17702](https://arxiv.org/abs/2507.17702)) report gains from a small shared fraction.
* Evidence against: OLMoE found no benefit ([2409.02060](https://arxiv.org/abs/2409.02060)). Qwen3-MoE
  also dropped shared experts ([2505.09388](https://arxiv.org/abs/2505.09388)†).
* The deciding argument here is systems. The shared expert is the only FFN work that needs no
  all-to-all, so it is the natural compute to overlap with dispatch (ScMoE-style shortcut overlap,
  [2404.05019](https://arxiv.org/abs/2404.05019)†).
* If the ablation shows no loss gain, set `d_shared=0` and spend the 189 M parameters on routed experts.

**Token-choice, dropless**
* OLMoE: token choice clearly beats expert choice, and dropless beats capacity-limited routing.
  ([2409.02060](https://arxiv.org/abs/2409.02060))
* MegaBlocks shows dropless MoE can be implemented with block-sparse/grouped GEMMs without the quality
  loss of token dropping. ([2211.15841](https://arxiv.org/abs/2211.15841)†)

**Sigmoid router with aux-loss-free bias balancing**
* Wang et al. add a per-expert bias to the scores used for **selection only**, updated as
  b ← b + u·sign(mean − load). This balances load without an interfering gradient and gives better
  perplexity than aux-loss balancing. ([2408.15664](https://arxiv.org/abs/2408.15664)†)
* DeepSeek-V3 adopts it with u = 0.001, sigmoid gating and a tiny sequence-wise loss (α = 1e-4).
  ([2412.19437](https://arxiv.org/abs/2412.19437)†)
* We compute loads over the **global batch**. Qwen shows micro-batch balancing forces per-sequence
  uniformity and hurts specialisation, while global-batch balancing improves perplexity and
  specialisation. ([2501.11873](https://arxiv.org/abs/2501.11873))
* The router z-loss (1e-3) follows ST-MoE ([2202.08906](https://arxiv.org/abs/2202.08906)†) and OLMoE.
  With sigmoid gating it acts only as a logit-magnitude regulariser.

**Group-limited routing (M = 4 of 8 ranks)**
* This is DeepSeek-V3's node-limited routing (M = 4 nodes) moved down to the GPU level.
* Without it, dedup still sends 5.34 rows per token per layer (expected distinct ranks for uniform
  top-8 of 128 over 8 ranks). With it, the bound is 4 rows. A real router measures 3.98 of 4
  (`tests/moe/test_budget.py`).
* That is **2× less all-to-all than OLMoE without dedup**. The quality cost is unmeasured at our scale,
  so it is a Day-1 ablation.

**First layer dense**
* DeepSeekMoE/V3 keep the first layer(s) dense because routing in layer 0 balances poorly.
  ([2401.06066](https://arxiv.org/abs/2401.06066)†, [2412.19437](https://arxiv.org/abs/2412.19437)†)
* It also removes one of 16 all-to-all rounds.

**Attention: GQA + QK-norm + output gate; no MLA**
* GQA with 4 KV heads is near-MHA quality with fewer parameters
  ([2305.13245](https://arxiv.org/abs/2305.13245)†). It saves 50 M parameters versus OLMoE's MHA.
* QK-norm (OLMoE, OLMo 2 [2501.00656](https://arxiv.org/abs/2501.00656)†) removes attention-logit
  growth, the main pre-training instability, and makes Muon's QK-clip unnecessary.
* Qwen's gated attention study is the deciding paper here: 30 variants on 15 B MoE / 1.7 B dense
  models over 3.5 T tokens. A head-specific sigmoid gate after SDPA consistently improves quality,
  stability and tolerated LR, and removes attention sinks.
  ([2505.06708](https://arxiv.org/abs/2505.06708),
  [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/120216))
* Cost of the gate: +4.2 M parameters per layer.
* MLA ([2405.04434](https://arxiv.org/abs/2405.04434)†) mainly shrinks the KV cache for inference. At
  4k training context it adds FLOPs and kernels without a training-quality argument, so we skip it.

**Multi-token prediction: off**
* Meta's MTP paper: the gains grow with model size and **do not appear for small models**.
  ([2404.19737](https://arxiv.org/abs/2404.19737))
* At ~1 B active, the extra head's FLOPs and memory are better spent on tokens. This is an optional
  Day-1 ablation.

**From scratch, not upcycled**
* OLMoE found that from-scratch training catches up with upcycling.
  ([2409.02060](https://arxiv.org/abs/2409.02060))
* No architecture-compatible dense checkpoint with this tokenizer exists anyway. Pythia uses the NeoX
  vocabulary, but it has parallel attention, LayerNorm and GeLU.

## 3. Optimizer and schedule

| Item | Value | Evidence / calculation |
|---|---|---|
| Optimizer | **Muon** for all matrices inside blocks: attention, dense/shared FFN, and every expert matrix (w1 and w3 orthogonalised separately). AdamW (β = 0.9/0.95, ε = 1e-8) for embedding, head, router and norms | Moonlight: ≈2× compute efficiency over AdamW in compute-optimal scaling ([2502.16982](https://arxiv.org/abs/2502.16982)); also validated on a 16B-A3B MoE. Essential AI: Muon extends the compute/time Pareto frontier and keeps data efficiency at large batch ([2505.02222](https://arxiv.org/abs/2505.02222)†) |
| Muon details | Nesterov momentum 0.95, 5 Newton–Schulz steps (3.4445, −4.7750, 2.0315), update × 0.2·√max(A, B), decoupled WD | Moonlight's two scaling fixes: weight decay, and per-matrix update-RMS matching so one LR serves Muon and AdamW |
| Peak LR | **7e-4** | DeepSeek LLM fit η_opt = 0.3118·C^−0.125 with C = 8.1e20 → **7.6e-4** ([2401.02954](https://arxiv.org/abs/2401.02954)†); `tools/budget.py` reproduces it |
| Batch | **960 × 4096 = 3.93 M tokens** | The same fit gives B_opt = 0.2920·C^0.3271 = **2.0 M**. We use ≈2× because each step pays a fixed inter-node all-reduce (§6.4). Near the optimum the loss is flat in B, and the critical batch size grows with data size rather than model size ([2410.21676](https://arxiv.org/abs/2410.21676)). 960 divides 48 GPUs |
| Schedule | WSD: 1,000-step linear warmup (3.9 B tokens), constant, then a **(1 − √) cooldown to 0 over the last 20 %** (8,140 steps ≈ 32 B tokens) | Matches cosine at equal compute and allows extending the run without restarting the schedule ([2405.18392](https://arxiv.org/abs/2405.18392)†, MiniCPM [2404.06395](https://arxiv.org/abs/2404.06395)†). The cooldown is where the highest-quality data mix goes (§9) |
| Weight decay | 0.1 on matrices; none on embeddings, norms, biases | OLMo 2 removes WD on embeddings |
| Grad clip | 1.0 global norm (exact across dense FSDP shards and EP shards) | |
| Steps | 40,700 = 160 B tokens | |

AdamW stays available (`--optimizer adamw`) as the Day-1 control and fallback.

Muon on this layout costs almost nothing in communication:
* Expert matrices are whole on their EP rank, so their Newton–Schulz runs locally.
* Only the FSDP-sharded dense matrices (≈0.45 B parameters) are all-gathered inside the node, once per
  optimizer step.
* Muon also saves 4 B/parameter of optimizer state versus AdamW (≈3 GB per GPU here), which is what
  makes micro-batch 1 fit on a 24 GB card.

## 4. Precision plan

| Tensor / op | Precision |
|---|---|
| Matmuls (attention projections, experts, shared, head) | BF16 inputs, FP32 accumulate |
| Master weights, gradients accumulated locally, Muon/AdamW state | FP32 |
| Router: weight, logits, sigmoid, top-k, bias | FP32 (FSDP mixed-precision policy override) |
| Norms, softmax inside SDPA, cross-entropy | FP32 |
| Dense gradient reduction (intra-node reduce-scatter + inter-node all-reduce) | FP32 |
| Expert gradient inter-node all-reduce | **BF16** (`--grad-sync-dtype`). It halves the dominant inter-node bytes; averaging 6 values in BF16 adds at most 2⁻⁸ relative error, at the level of the BF16 compute noise already present. The FP32 path is tested to match single-process training exactly |
| All-to-all payload | BF16. FP8 e4m3 row-scaled for the forward dispatch only is available (`--fp8-dispatch`), as in DeepSeek-V3 |

**FP8 compute is not enabled at the start.**
* The budget shows compute is only about half of step time (49–54 %). FP8 doubles peak on both cards (330 / 419 TF),
  so at best it removes ~25 % of a step while adding numerical risk.
* The risks are documented: SwiGLU outlier amplification late in training (Smooth-SwiGLU,
  [2409.12517](https://arxiv.org/abs/2409.12517)†), and the need for fine-grained 1×128 / 128×128
  scaling (DeepSeek-V3).
* The block-scaled MMA fast paths in vendor libraries target datacenter parts, so on sm_89 / sm_120 we
  would have to write and validate our own FP8 grouped kernel.
* `model/fp8.py` provides row-wise quantise/dequantise hooks and a policy object. Embeddings, head,
  router, norms, attention softmax, master weights and optimizer state stay at ≥ BF16 under any policy.
* Turn FP8 on only if the Day-1 profile shows the run is compute-bound.

## 5. Implementation map

| File | Contents |
|---|---|
| `model/config.py` | presets `moe7b`, `olmoe`, `tiny`; exact parameter accounting |
| `model/layers.py` | RMSNorm, RoPE, GQA + QK-norm + gate (SDPA/flash), SwiGLU |
| `model/moe.py` | Router (sigmoid, group-limited, bias, z-loss, seq-aux, global-load bias update); `dispatch_combine` (dropless, per-rank dedup, differentiable variable-split all-to-all, optional FP8 dispatch, selective expert recompute); MoE layer |
| `model/grouped_gemm.py` | sync-free Triton grouped GEMM (forward/dX: M-grouped; dW: K-grouped). The tile→expert map is computed on device and the grid is a host-side upper bound. Pure-PyTorch reference fallback |
| `model/model.py` | Transformer, chunked FP32 CE, init |
| `train/parallel.py` | 2-D mesh (replicate × shard), FSDP2/HSDP with experts ignored, EP group = shard group, expert grad sync (post-accumulate hooks, async), exact global clip, global-batch router-bias update |
| `train/optim.py` | Muon + AdamW (DTensor-aware), WSD |
| `train/data.py` | deterministic packed uint16 loader (stateless, resumable from the step number), `path:R` repetition |
| `train/ckpt.py` | per-rank checkpoints on node-local disk, background write, upload hook, atomic rename |
| `train/train.py` | training loop with logging of loss, aux, grad norm, tok/s, router MaxVio / load entropy, a2a rows per (token, expert) pair, dropped tokens |
| `tools/budget.py` | every number in §§2, 6, 7, 8 |

The all-to-all needs one host sync per MoE layer to exchange split sizes. NCCL's variable-split
all-to-all requires them, and torchtitan and Megatron pay the same cost. The grouped GEMM itself never
syncs.

## 6. Parallelism plan for this hardware

### 6.1 Layout

```
mesh = (replicate = 6 nodes) × (shard = 8 GPUs)          [5090 alternative: (12 sockets) × (4 GPUs)]
node n:  GPU0 … GPU7
  experts   : EP over the 8 GPUs → 16 experts per GPU per layer; replicated on all 6 nodes
  dense     : FSDP2 shard over the 8 GPUs, replicated across nodes (HSDP)
  data      : every GPU is its own data-parallel rank (48 ranks × 20 sequences per step)
intra-node traffic (SHM): all-to-all (dispatch/combine), FSDP all-gather/reduce-scatter of dense params
inter-node traffic (RoCE): once per optimizer step: expert grads (BF16) + dense grad shards (FP32)
```

No token ever crosses the network. With 2 × 25 GbE per node, cross-node all-to-all would cost about
15 GB/s per node at the needed token rate, more than 3× the link. Cross-node EP is therefore ruled out.

### 6.2 All-to-all bytes per token (bytes one GPU sends per token per training step, all 15 MoE layers)

There are four all-to-alls per layer: forward dispatch, forward combine, and the two backward
transposes. Only the 7/8 remote share is counted. A 64 B FP32 metadata row (gate weights and local ids)
travels with each row.

| Scheme | Rows per token per layer | MB per token per step |
|---|---|---|
| OLMoE as planned (64e top-8, one row per expert) | 8.00 | 1.849 |
| OLMoE + dedup | 5.43 | 1.256 |
| Ours, no dedup | 8.00 | 1.734 |
| Ours, dedup only | 5.34 | ≈1.16 |
| **Ours: dedup + group-limited M = 4** | **≤ 4.00 (3.98 measured)** | **0.867** |
| … + FP8 dispatch | 4.00 | ≈0.76 |
| 5090 socket-local EP = 4, M = 2 of 4 | 2.00 | 0.372 |

How dedup works (`model/moe.py`):
* Each token is sent once to every rank that hosts at least one of its experts.
* The receiving rank expands the row to its local experts, runs the grouped GEMM, and sums the
  gate-weighted outputs per (token, rank).
* It sends back one row, so combine benefits from dedup as much as dispatch.
* Gate-weight gradients flow back through the metadata all-to-all.
* Exactness against a naive per-expert loop, single-process and 2-rank gloo, forward and all gradients,
  is tested in `tests/moe/test_dispatch.py`.

### 6.3 Overlap strategy

1. **Shared expert ‖ dispatch.** The shared-expert FFN (33 % of active FFN FLOPs in MoE layers) needs no communication. Run
   it on a second CUDA stream while the dispatch all-to-all is in flight, and the same in backward.
   Implementation note: `AllToAll` is a plain autograd Function, so this is a stream-scheduling change
   only. It is the first Day-2 optimisation, and the budget's "expected" scenario assumes 30 % of
   all-to-all time hidden.
2. **No recompute of communication.** `--recompute-experts` re-runs only the local expert GEMMs in
   backward. It never re-runs an all-to-all.
3. **Expert gradient all-reduce ‖ last backward.** Post-accumulate-grad hooks launch an async BF16
   all-reduce per expert tensor during the last micro-batch's backward. Layer 15's experts start
   syncing while layers 14…1 are still back-propagating.
4. **HSDP with `set_requires_all_reduce(last)`.** Intra-node reduce-scatter runs every micro-batch.
   The inter-node all-reduce runs once per step.
5. **Checkpoint uploads are throttled.** The RoCE bond is shared with JuiceFS.

### 6.4 Synchronous HSDP vs DiLoCo

Per step every node pushes 2·5/6 of one replica: experts 6.04 B × 2 B plus dense 0.66 B × 4 B. That is
**24.5 GB per node per step, 5.6 s at 4.38 GB/s**, and only ≈0.3–0.5 s of it overlaps the last backward.

This is the largest single overhead (24–36 % of a step). We still choose synchronous training:
* **Quality.** DiLoCo with one replica matches or beats data parallel. Its scaling-law paper shows
  M ≥ 2 replicas only catch up with DP above several billion parameters
  ([2503.09799](https://arxiv.org/abs/2503.09799), [DiLoCo 2311.08105](https://arxiv.org/abs/2311.08105)†).
  At ~1 B active with 6 replicas we would expect a measurable loss penalty.
* **Throughput.** Even with the sync fully exposed, the conservative 4090 budget reaches 211 B tokens,
  well above the 160 B target.
* **Robustness.** DiLoCo's outer step adds hyperparameters (H, outer LR/momentum) that have no MoE
  evidence at this scale.

When to switch to DiLoCo:
* The Day-1 measured inter-node bandwidth is below ~2 GB/s per node (JuiceFS contention, or LACP
  hashing everything onto one 25 G leg).
* The configuration: one replica per node (the existing HSDP group), H ≈ 100 inner steps, outer
  Nesterov (lr 0.7, momentum 0.9), outer state in pinned host memory. That cuts inter-node bytes by
  ~100×.
* Cheaper levers come first: raise the global batch to 1,920 sequences (halves the relative sync cost),
  and set NCCL to spread flows over both bond legs (`NCCL_IB_QPS_PER_CONNECTION` /
  `NCCL_NSOCKS_PERTHREAD`).

### 6.5 Socket-local EP on 5090 (2 NUMA nodes)

* With EP = 4 per socket, the all-to-all never crosses the inter-socket link and moves 0.372 MB/token
  instead of 0.867.
* The cost is memory: 32 experts per GPU and 29.2 GB/GPU at micro-batch 1 (fits 32 GB, tight). Expert
  replicas also double, adding one intra-node cross-socket all-reduce per step.
* Config: `--ep 4` with `--override '{"route_groups": 4, "route_topk_groups": 2}'`.
* 4090 nodes have 8 NUMA domains, one per GPU, so there is no socket-local option there. EP = 8 is used.

## 7. Memory per GPU (`tools/budget.py`)

States are FP32 master + FP32 grad + optimizer state (Muon momentum for matrices, Adam moments for
embedding/head/router), sharded by EP/FSDP over 8 GPUs. Activations are BF16 with selective expert
recompute. "Other" is the largest unsharded unit + prefetch, a logits chunk, and 2 GB of NCCL/allocator
slack.

| GPU | Micro-batch | States GB | Activations GB | Other GB | **Total GB** | Fits | Without expert recompute |
|---|---|---|---|---|---|---|---|
| 4090 (24 GB) | 1 × 4096 | 10.2 | 5.7 | 3.2 | **19.1** | yes | 24.6 (does not fit) |
| 5090 (32 GB) | 2 × 4096 | 10.2 | 11.4 | 3.2 | **24.8** | yes | 35.9 (does not fit) |

## 8. Time budget per optimizer step and tokens in 15 days

Per GPU per step: 20 sequences = 81,920 tokens, 4090 at micro-batch 1 and 5090 at micro-batch 2.

Assumptions:
* **Conservative:** GEMM-weighted compute efficiency 30 % of BF16 peak, no all-to-all overlap.
* **Expected:** 40 % and 30 % of the all-to-all hidden.
* SHM bandwidth at the low end of its range (15 / 22 GB/s).
* Grad sync overlaps only the last micro-batch's backward.
* 90 % availability.
* Recompute FLOPs are included in time but not in MFU.

**Proposed MoE-7B-A1B**

| GPU | Scenario | Compute s | All-to-all s | Grad sync s (exposed) | Step s | tok/s/GPU | MFU | Tokens in 15 d |
|---|---|---|---|---|---|---|---|---|
| 4090 | conservative | 11.83 | 4.73 | 5.61 (5.21) | 21.78 | 3,761 | 15 % | **211 B** |
| 4090 | expected | 8.88 | 3.31 | 5.61 (5.31) | 17.50 | 4,681 | 18 % | **262 B** |
| 5090 | conservative | 9.33 | 3.23 | 5.61 (4.98) | 17.54 | 4,670 | 14 % | **261 B** |
| 5090 | expected | 7.00 | 2.26 | 5.61 (5.14) | 14.40 | 5,690 | 17 % | **319 B** |

**OLMoE-1B-7B-like baseline, same cluster, same recompute**

| GPU | Scenario | Step s | tok/s/GPU | Tokens in 15 d |
|---|---|---|---|---|
| 4090 | conservative (as planned, no dedup) | 30.90 | 2,651 | 148 B |
| 4090 | conservative + dedup | 27.66 | 2,962 | 166 B |
| 5090 | expected (as planned) | 19.11 | 4,286 | 240 B |

What the tables mean:
* **160 B tokens** takes **11.4 days** on 4090 in the conservative case (9.2 expected) and 7.5 days on
  5090 in the expected case. All figures include 90 % availability.
* As planned, the baseline would need 16.2 days on 4090 (conservative) for the same 160 B tokens, which
  exceeds the 15-day window.
* MFU is low (14–18 %) because 45–51 % of each step is communication. The measures in §6.3 target that
  share, not FLOPs.

## 9. Data: repetition and upsampling

* Muennighoff et al.: up to ~4 epochs of repeated data is almost as good as unique data; returns
  decline quickly beyond that ([2305.16264](https://arxiv.org/abs/2305.16264)†).
* Policy:
  * Stable phase: one pass over the deduplicated corpus.
  * If the measured throughput leaves slack beyond 160 B, extend the stable phase (WSD allows it) by
    **upsampling the top-quality buckets** (both classifiers agree, top percentile; FineMath ≥ 4;
    Stack-Edu ≥ 3) up to **≤ 4 total exposures**, rather than lowering the quality threshold.
  * Cooldown (last 20 %): weight these same buckets heavily. This is the standard annealing practice
    and is cheap to A/B from a stable-phase checkpoint.
* Mechanism: `--data 'math/*.u16:3,web/*.u16'` repeats a shard set R times inside the deterministic
  epoch permutation.

## 10. Risks (ranked) and mitigations

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| 1 | SHM all-to-all slower than 15 GB/s at our message sizes (8 GPUs contend for the same DRAM; both directions) | step time ↑ (a2a is 16–22 % of the step) | Measure Day 1. Levers in order: shared-expert overlap; FP8 dispatch (−12 %); socket-local EP4 on 5090; M = 3; top-k 6 |
| 2 | Inter-node bandwidth below 4.4 GB/s (JuiceFS contention, LACP putting one flow on one leg) | exposed grad sync ↑ | BF16 expert sync (default); multi-QP/socket NCCL; storage QoS and throttled uploads; global batch 1,920; DiLoCo fallback (§6.4) |
| 3 | Triton grouped GEMM untested on real sm_89/sm_120 here (CPU interpreter only) | slow or wrong expert GEMMs | GPU test in `tests/moe/test_grouped_gemm.py` (skips on CPU); Day-1 autotune of BM/BN/BK and num_warps/stages; the reference loop and `MOE_GMM=ref` stay as a correctness fallback |
| 4 | Muon + fine-grained MoE + group-limited routing at this scale (few public data points) | instability or worse loss | QK-norm, z-loss, clip 1.0; Day-1 Muon vs AdamW A/B on a ~2 B-token proxy; switch with `--optimizer adamw` |
| 5 | Load imbalance across EP ranks (the slowest rank sets the step time) | throughput ↓ | aux-free bias on global loads every step; MaxVio and entropy logged per layer; raise u or α if MaxVio stays above 0.2 after warmup |
| 6 | 4090 memory estimate off by > 3 GB | OOM | expert recompute (on), micro-batch 1, chunked CE, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; next lever is attention recompute |
| 7 | Host sync per MoE layer (split sizes), GPU idle bubbles | throughput ↓ | measure in the profile; overlap with the shared expert; a fixed-capacity padded-split mode is a known follow-up |
| 8 | Design choices not ablated on this corpus (shared expert, M = 4, gate) | sub-optimal loss | Day-1/2 ablations (§11) before the main run is locked |
| 9 | Node or GPU failures over 15 days | lost time | per-rank async checkpoints to local NVMe every 200 steps (≈1 h), upload hook; stateless data loader resumes from the step |
| 10 | Resharding a checkpoint to a different (world, EP) layout is not implemented | inflexible restarts | expert init and keys are layout-independent; an offline reshard tool is a follow-up |

## 11. Day-1 measurements and experiments (in order)

Items 1–4 are scripted in `bench/` (`bench/README.md`: two sbatch jobs; results feed
`tools/budget.py --measured`).


1. **NCCL SHM all-to-all** (`nccl-tests alltoall_perf`, `NCCL_P2P_DISABLE=1`). Message sizes 8–64 MB,
   EP = 8 and EP = 4 socket-local (5090). Fill `GPUS[*]['shm_gbs']` in `tools/budget.py`.
2. **Inter-node all-reduce** with JuiceFS under normal load. Check that both bond legs carry traffic.
   Fill `CLUSTER['nic_eff']`.
3. **Grouped GEMM**: run `pytest tests/moe -k gpu`, then autotune `BLOCK` per GPU and compare against
   the per-expert cuBLAS loop.
4. **One-node full-model step**: peak memory (validates §7) and a step breakdown with the torch
   profiler (validates §8). Update `SCENARIOS` and re-run `tools/budget.py`.
5. **Quality ablations** on the full model, 48 GPUs, ~2 B tokens each, same data order:
   * Muon vs AdamW
   * shared expert on/off
   * M = 4 vs unrestricted
   * attention gate on/off
   * LR ∈ {5e-4, 7e-4, 1e-3}
   * optional: MTP, document masking.
6. **FP8 dispatch** loss check over 1 B tokens (enable only if no loss difference beyond seed noise).

## 12. Tests (CPU, `pytest`)

`pytest` (configured by `pytest.ini` to `tests/moe`): **44 passed, 1 skipped** (the GPU-only Triton
test).

* Parameter counts: module = analytic = doc; the OLMoE preset reproduces 6.9 B / 1.3 B.
* Router: normalised gates, distinct top-k, group limit, bias update direction, a skewed router balanced
  by the bias alone, counting only in training mode.
* Dispatch: dropless equals a naive per-expert loop (with and without dedup); exact permutation inverse;
  gradients equal; selective recompute gives identical values and gradients; FP8 dispatch error < 8 %.
* **2-process gloo EP:** forward and all gradients equal the single-process computation.
* Grouped GEMM: reference equals per-expert matmul; autograd equals dense; dW reference; **Triton
  kernels in the Triton CPU interpreter** equal the reference, including empty experts.
* Training: Newton–Schulz orthogonality and update RMS; parameter-group routing; WSD shape; the tiny
  model overfits one batch; checkpoint round-trip including router bias and identical next step;
  incomplete checkpoints ignored; loader determinism / disjointness / permutation / repeat.
* **Single-process smoke run** of `train.train` with resume.
* **4-process torchrun (2 replicas × FSDP2+EP 2):** loss and grad norm match single-process training to
  1e-4 with an FP32 expert sync, and within BF16 noise with the default BF16 sync.
* Day-1 kit (`tests/moe/test_bench.py`): comm benchmark on 4 gloo ranks, grouped-GEMM autotuner dry run in
  the Triton interpreter (and the written tile config reloads through `MOE_GMM_CONFIG`), synthetic shards,
  `--measured` budget, timing fields.
* Budget: the expected-distinct-ranks formula matches Monte Carlo; group-limited rows measured with the
  real router; doc numbers.

## 13. Comparison with the OLMoE-1B-7B-like baseline

| | OLMoE-1B-7B-like (planned) | **Proposed** | Why it matters on this cluster |
|---|---|---|---|
| Total / active params | 6.92 B / 1.28 B | 6.70 B / 1.04 B | similar capacity; 19 % fewer FLOPs per token (6.40 vs 7.88 G) |
| Experts | 64 × 1024, top-8, G = 4 | 128 × 512, top-8, G = 8, + 1 shared (2048) | G inside the 8–12 optimum; lower activation ratio → higher efficiency leverage (Ling) |
| Routing / balance | softmax, aux LB loss 0.01, z-loss | sigmoid, aux-free bias on global loads, α = 1e-4, z-loss | better perplexity than aux-loss balancing (2408.15664, 2501.11873) |
| All-to-all per token | 1.85 MB | 0.87 MB | −53 % of the SHM traffic, the main intra-node cost |
| Attention | MHA 16 heads + QK-norm | GQA 16/4 + QK-norm + output gate | gate: better loss and stability at higher LR (2505.06708) |
| Optimizer | AdamW | Muon (+AdamW for embed/head/router) | ≈2× compute efficiency (2502.16982); −3 GB/GPU of state |
| Schedule | cosine | WSD, 1−√ cooldown | the token count can be extended once throughput is measured |
| Tokens in 15 d (4090, conservative) | 148 B | 211 B | +43 % tokens on the same hardware |

## References

* OLMoE — https://arxiv.org/abs/2409.02060 (ablations summarised via https://github.com/allenai/OLMoE)
* DeepSeekMoE — https://arxiv.org/abs/2401.06066 †
* DeepSeek-V2 (MLA) — https://arxiv.org/abs/2405.04434 †
* DeepSeek-V3 (aux-free balancing at scale, node-limited routing, FP8, MTP) — https://arxiv.org/abs/2412.19437 †
* Auxiliary-loss-free load balancing — https://arxiv.org/abs/2408.15664 †
* Scaling laws for fine-grained MoE — https://arxiv.org/abs/2402.07871
* Towards greater leverage: scaling laws for efficient MoE (Ling) — https://arxiv.org/abs/2507.17702
* Global-batch load balancing (Qwen) — https://arxiv.org/abs/2501.11873
* Qwen3 technical report — https://arxiv.org/abs/2505.09388 †
* ST-MoE (router z-loss) — https://arxiv.org/abs/2202.08906 †
* MegaBlocks (dropless) — https://arxiv.org/abs/2211.15841 †
* Gated attention — https://arxiv.org/abs/2505.06708
* GQA — https://arxiv.org/abs/2305.13245 †
* OLMo 2 — https://arxiv.org/abs/2501.00656 †
* Moonlight / Muon is scalable — https://arxiv.org/abs/2502.16982
* Practical efficiency of Muon for pretraining — https://arxiv.org/abs/2505.02222 †
* DeepSeek LLM (LR / batch scaling fits) — https://arxiv.org/abs/2401.02954 †
* How does critical batch size scale — https://arxiv.org/abs/2410.21676
* WSD / cooldown scaling — https://arxiv.org/abs/2405.18392 †, MiniCPM — https://arxiv.org/abs/2404.06395 †
* Multi-token prediction — https://arxiv.org/abs/2404.19737
* DiLoCo — https://arxiv.org/abs/2311.08105 †; DiLoCo scaling laws — https://arxiv.org/abs/2503.09799
* Shortcut-connected expert parallelism — https://arxiv.org/abs/2404.05019 †
* Scaling FP8 training to trillion-token LLMs — https://arxiv.org/abs/2409.12517 †
* Scaling data-constrained LMs — https://arxiv.org/abs/2305.16264 †
* NVIDIA Ada architecture whitepaper — https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
* RTX 5090 (GB202) tensor throughput — https://www.spheron.network/blog/nvidia-rtx-5090-specs/

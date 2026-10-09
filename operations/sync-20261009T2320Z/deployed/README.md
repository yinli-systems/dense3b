# moe7b-curated-corpus

Data-quality tooling for the 7B MoE pretraining corpus. Goal: **the highest-quality unique tokens —
quality first, quantity second.** Everything here produces *evidence* (scores, signatures, matches);
nothing deletes source data or admits data to training automatically.

Compute runs on the ParaCloud Slurm cluster (`gpu_4090`, max 6 CPUs / GPU). This repo holds code,
tests and decisions only — no data, no credentials.

## Layout

| Path | What |
|---|---|
| `neural_scoring/score_fast.py` | v2 FineWeb-Edu classifier scorer (head/middle/tail windows per doc). CPU work in worker processes; reuses hash-verified v1 receipts. |
| `neural_scoring/chunk_encode.py` | Exact, cached BERT WordPiece encoding (splits on BERT whitespace, caches chunks). |
| `neural_scoring/score_core.py` | Shared mechanics from v1 (windows, numeric gate, text verification). Unchanged. |
| `fused_scan/scan_fused.py` | One decode per doc → v1 MinHash near signature + v1 benchmark evidence (v1 code imported, sha-pinned) + DCLM fastText score. Task-parallel. |
| `fused_scan/reduce_fused.py` | Global LSH candidate groups (v1 grouping code), benchmark aggregation, per-source DCLM histograms. |
| `tests/test_chunk_encode.py` | Exactness of `ChunkEncoder` vs reference tokenizer on real docs from every source. |
| `tests/compare_canary.py` | Row-by-row comparison of a v2 canary against the v1 parquet for the same task. |
| `docs/MODEL_DESIGN.md` | Model + training design for the 7B-total / 1B-active MoE on 6×8 consumer GPUs. |
| `model/`, `train/`, `tools/budget.py` | MoE reference implementation, training loop, FLOPs/comm/memory budget. Tests: `tests/moe/` (`pytest`). |

## Verified so far (2026-10-02)

* `ChunkEncoder`: 0 id mismatches on 24,000 real documents (4 sources) + eviction/edge-case set
  (`\x0b`, `\x1f`, ` `, NBSP, accents, CJK, >512-token runs).
* v2 scorer canary (30,000 docs, olmoe_science chunk 0): all structural fields identical to v1;
  scores differ only by bf16 rounding (mean |Δ| 0.0012, max 0.013; project gate is 0.01 / 0.05).
* Throughput: ~580–814 windows/s per GPU vs 152 for v1 (≈4–5×).
* Bug caught by the canary and fixed: `tokenizer.json` ships padding/truncation; `encode_batch`
  padded chunks to the batch's longest. `ChunkEncoder` now uses a private copy with both disabled.

Root causes found: v1 GPU utilisation ~16% — single-threaded pipeline bottlenecked in BertNormalizer
/ pre-tokenisation; near/benchmark jobs were single-process while holding an idle GPU each.

## Selection policy (quality first)

Agreement-based, not recall-based (cf. Ultra-FineWeb intersection; Nemotron-CC uses max for recall):

1. **Whole-document quality** — the *minimum* of head/middle/tail FineWeb-Edu window scores must pass
   (18.6% of head≥3 tokens in olmoe_science have a window <2).
2. **Two independent classifiers agree** — FineWeb-Edu (educational value) and DCLM fastText
   OH-2.5+ELI5 (informativeness), each in the source's top percentile band.
3. **Domain specialists** — FineMath classifier (≥4) for math; Stack-Edu per-language (≥3) for code.
4. **Zero-tolerance hygiene** — any hygiene flag, any benchmark match, any text-roundtrip failure → out.
5. **Dedup keeps the best** — one document per near-dup cluster, highest combined score.
6. **Guard style collapse** — percentiles per source/topic, not one global threshold.

## Pinned external models

| Model | Revision | Use |
|---|---|---|
| HuggingFaceFW/fineweb-edu-classifier | 284663cb… | neural scoring |
| mlfoundations/fasttext-oh-eli5 | cd8b714a… | DCLM score |
| HuggingFaceTB/finemath-classifier | bd0b0e33… | math |
| HuggingFaceTB/stack-edu-classifier-{python,java,c,cpp,csharp,go,javascript} | per language | code |

## References

DCLM (arXiv 2406.11794) · FineWeb (2406.17557) · Nemotron-CC (2412.02595) · Ultra-FineWeb (2505.05427) ·
SmolLM2 / FineMath / Stack-Edu (2502.02737) · Olmo 3 (2512.13961) · Datasets, Documents, and Repetitions
(2503.07879) · Removing Noise, not Finding Gold (2510.00866) · WebOrganizer (2502.10341)

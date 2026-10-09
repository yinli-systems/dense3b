# Evidence formats (inputs to the final selection)

All evidence is keyed by **(source_id, ordinal)**. `source_id` is a 24-hex file id; `ordinal` is the
0-based document index inside that source file. Every document belongs to one `source`
(`fineweb_edu`, `dclm_baseline`, `finemath`, `olmoe_science_reference`, `stack_v3_permissive`).
Every artifact is referenced by a JSON receipt carrying `artifact.path`, `artifact.bytes`,
`artifact.sha256` — readers must verify the sha before trusting a payload.

## 1. Candidate set (frozen selection)

* Plan: `config/plan.json` of the neural-score workspace. `parts[*].tasks[*]` has `source`,
  `source_id`, `start`, `stop` (ordinal range), `documents`, and `mask`.
* `mask.mask_path` → `<source_id>.keep.bits`: little-endian bit array, bit `o` = 1 ⇔ document `o`
  is in the candidate set. `mask.documents` = bit count; `mask.selected_documents` = popcount.
* Non-code selection sha `1b3ed259…`; code candidates come from
  `selection/code-reviewed-v2/<source_id>.keep.bits` (same encoding).

## 2. Neural scores (FineWeb-Edu classifier) — `neural_scoring/score_fast.py`

Receipt: `scores/<source_id>/<start:09d>-<stop:09d>/COMPLETE.json` (v2 root, or reused v1 root —
the part receipt `reports/part-NNN.COMPLETE.json` lists every chunk receipt in `receipts`).
Parquet columns:

| column | type | notes |
|---|---|---|
| ordinal | uint64 | |
| source_text_tokens | uint32 | tokens in the training tokenizer |
| text_hash | binary(32) | sha256 of the UTF-8 text |
| metadata_status / metadata_flags | uint8 / uint16 | from metadata pass |
| status | string | `SCORED_NOT_FACT_CHECKED` or `HOLD_RAW_RECONSTRUCTION_REQUIRED` (scores null) |
| classifier_tokens, covered_classifier_tokens | uint32 | BERT tokens total / covered by windows |
| head_start, middle_start, tail_start | uint32 | window offsets (510-token windows) |
| head_score, middle_score, tail_score | float32 | regression output, ~0–5 scale |
| mean_score, min_score, score_range | float32 | derived |

Numerical noise floor between runs: ~0.0156 (one bf16 ulp near 2–4). Thresholds should not be
interpreted more finely than ±0.02.

## 3. Hygiene (full-content scan) — one file per source

`full-content/<source_id>/COMPLETE.json` → `content.parquet`:
`ordinal uint64, tokens uint32, flags uint16, normalized_text_hash binary(32),
repeated_line_fraction float32, characters uint32`.

`flags` bitmask: 1 text_roundtrip_review · 2 replacement_or_nul_review · 4 secret_marker_review ·
8 repeated_lines_review · 16 email_like_review · 32 reserved_ids_review.
`normalized_text_hash` equal across documents ⇒ exact duplicate after normalisation.

## 4. Fused evidence — `fused_scan/scan_fused.py`

`output/<source_id>/<start:09d>-<stop:09d>/COMPLETE.json` → `evidence.parquet`:

| column | notes |
|---|---|
| ordinal, source_text_tokens, text_hash, metadata_status, metadata_flags | as above |
| status | `SCANNED` or `UNSCANNED_OVERSIZE_OR_RAW_RECONSTRUCTION_REVIEW` (>262,144 tokens or text hash mismatch) |
| near_status | `NEAR_SIGNATURE_ONLY` / `SHORT_DOCUMENT_NO_NEAR_SIGNATURE` |
| signature | binary(896) — 112 × uint64 LE MinHash, 5-unit shingles |
| bands | list<binary(16)> — 14 bands × 8 hashes |
| units, shingles | uint32 |
| benchmark_status | `no_registered_lexical_match` / `reference_match_review` / `match_budget_review` |
| evidence_json | JSON with matched references (`kind` incl. `long_reference_match_review`, `short_reference_match_review`, `legacy_unattributed_13word_match_review`) |
| dclm_hq_prob | float32, P(`__label__hq`) from DCLM OH-2.5+ELI5 fastText on text with `\n`→space |

Global reduction (`global/`): `candidate_members.tsv.gz` lines
`<band:02d>:<hex>\t<source_id>\t<ordinal>\t<tokens>`, grouped by key (only buckets with >1 member);
`dclm_histograms.json`: per source, 1000-bin `doc_hist` and `token_share_at_or_above_bin`.

## 5. Code specialists (Stack-Edu) — one file per source per language

`code-specialist/<lang>/<source_id>.json` (receipt, `context_tokens`=1024) →
`<source_id>.parquet`: `ordinal, tokens, text_hash, head_score, middle_score, tail_score,
mean_score, score_range` (0–5 scale). Languages: python, java (done); c, cpp, csharp, go,
javascript (pending).

## 6. FineMath classifier (pending)

Will mirror §5 with a single `score` per document (first 512 tokens, 0–5 scale; FineMath-4+ ⇔ ≥ 3.5
after rounding convention to be confirmed against the model card).

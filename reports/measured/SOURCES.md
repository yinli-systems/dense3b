# Results provenance and reporting conventions

Release: **dense3b-measured-20261018** · Published **2026-10-18**.

The model author confirmed every value as a measured result and supplied final corrections on 2026-10-18. This publication records those measurements; it does not execute model evaluation.

The 6.82B row is linked to the archived step 6499 evaluation. Its token label is author-reported and rounded. The 20B through 100B rows are author-confirmed measurement reports; their per-example outputs, checkpoint hashes and evaluator manifests were not supplied with this release.

## Measurement sources

| Source | Coverage | Evidence available in this repository |
|---|---|---|
| Archived step 6499 evaluation | Early 6.82B-labelled row | [Checkpoint report](../checkpoints/dense3b-step_00006499.md), versioned summaries and sample recount records |
| Author-confirmed measurements, 18 October 2026 | 20B, 40B, 60B, 80B and 100B | Published scores and author corrections; raw outputs and later evaluator manifests are not attached |

## Precision and protocol

- Percentages retain the precision supplied by the author; display precision is not uncertainty.
- MBPP at 100B is reported as 52%, rounded to an integer percentage. No exact decimal or passed count is assigned.
- MMLU at 100B is approximately 52%. The 6.82B anchor uses MMLU-MC5; later reports are labelled MMLU without assigning an unprovided evaluator protocol. MMLU-CF stays separate.
- Later per-run sample counts, prompt/shot settings, decoding settings, extraction rules and evaluator hashes remain unreported in this release.
- Integer counts are retained only where the archived evaluation provides them. Arithmetic compatibility does not create a new per-example record.
- The initial data mixture remains available in the original manifest. No revised mixture or switch point is supplied in this release.

## Later-run systems measurement

The author reported **45,380 tokens/s** mean global training throughput during later training. The averaging interval and supporting logs were not supplied. This is **not** the same statistic as the **41,780 tokens/s** median from the 9 October archived snapshot (last at most 100 observations).

## Final author corrections

| Benchmark | Stage | Confirmed score (%) |
|---|---|---:|
| humaneval_plus | 40B | 39.0 |
| mbpp | 20B | 38.4 |
| mbpp | 40B | 42.9 |
| mbpp | 60B | 46.3 |
| cruxeval_i | 60B | 30.750 |
| mbpp_plus | 100B | 51.3228 |

## Integrity

Canonical source: [`results/measured/2026-10-18.json`](../../results/measured/2026-10-18.json).

Source SHA-256: `cde60b0457ea030a7219dd5d3f692197b92abfd878e7ad4259fd5c7d0fd6589c`.

The page builder checks schema, ranges, stage coverage, exact corrections, Plus/base ordering and archived-count compatibility. These checks do not independently verify the author's later measurements. No model weights, synthetic per-example records, optimizer states, credentials or datasets are added by this release.

The historical evaluation and training archive remains at its original paths. Original capture receipts and artifact bytes are unchanged. Archive rebuilds validate sanctioned public redactions against the [existing audit](../../provenance/PUBLIC_NETWORK_REDACTION_AUDIT.json) and [two explicitly verified byte-identical aliases](../../provenance/PUBLIC_NETWORK_REDACTION_ALIASES.json). Unlisted changes still fail hash validation.

The [original publication validation record](validation.json) predates this throughput update. The updated reporting code contains metric validation and a regression test; those Python tests have not been rerun for this patch.

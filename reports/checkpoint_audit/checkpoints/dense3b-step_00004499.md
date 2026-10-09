# dense3b — checkpoint 4499

[All checkpoints](../CHECKPOINTS.md)

**Storage:** `historical_result_only`. **Evaluation coverage:** `primary_task_counts_present`.

**Tokens:** 4.718592 B; basis: `constant_observed_schedule_reconstruction_not_metadata_verified`. Schedule reconstructions are explicitly not metadata-verified; they describe logical stream progress and exclude replayed work after rollbacks.

**Evidence types:** evaluation_summary.

**Missing primary task families:** none at the recorded full sample counts.

**Recorded weight hashes:** `d823c750632ab45d5d5a0bde4327fee812c5ac6be21531a5d06921cf678cce74`. These are hashes stated in evaluation manifests, not new checksums from this audit.

## Checkpoint locations

| Path | Model bytes | Optimizer shard files | COMPLETE marker |
|---|---:|---:|---|
| Not present in the live inventory | — | — | — |

Historical metadata, results and resume traces are retained even when the model weights were pruned. A resume trace only establishes that the trainer reported a resumed start; it is weaker than a saved metadata file.

## Training observations near this checkpoint

Each row keeps its original job and actual observed step. An observation up to 10 steps earlier is not relabeled as a checkpoint measurement.

| Job | Observed step | Train loss | Validation loss | Source |
|---|---:|---:|---:|---|
| 1667439 | 4499 | 1.077359 | 0.802570 | [source](../../../evidence/raw/results/train-1667439/metrics.jsonl#L4500) |

## Every evaluation version

A later file is not automatically better. Protocol fingerprints, checkpoint hashes, sample counts, smoke runs and re-runs stay separate. Percent signs are used only for fractions; loss and router statistics keep their native units.

### Evaluation `7b1c28bffad71131`

Protocol: `unversioned-mc` (unversioned); original token summary: `None` B.

[Original result](../../../evidence/imported/moe1b/mc-dense3b-step4499.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| arc_challenge | acc | 27.0500% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | acc_norm | 29.5200% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | seconds | 15 | seconds | 1172 | full_recorded_count | — | summary_only |
| arc_easy | acc | 59.7600% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | acc_norm | 52.6900% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | seconds | 25 | seconds | 2376 | full_recorded_count | — | summary_only |
| boolq | acc | 52.4800% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | acc_norm | 61.9900% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | seconds | 60 | seconds | 3270 | full_recorded_count | — | summary_only |
| hellaswag | acc | 30.0300% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | acc_norm | 33.6800% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | seconds | 215 | seconds | 10042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc | 27.4500% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc_norm | 28.9500% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | seconds | 269 | seconds | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc | 24.3300% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc_norm | 24.3300% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | seconds | 2342 | seconds | 14042 | full_recorded_count | — | summary_only |
| openbookqa | acc | 19.2000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | acc_norm | 31.0000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | seconds | 5 | seconds | 500 | full_recorded_count | — | summary_only |
| sciq | acc | 82.1000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | acc_norm | 75.9000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | seconds | 31 | seconds | 1000 | full_recorded_count | — | summary_only |
| winogrande | acc | 51.4600% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | acc_norm | 51.4600% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | seconds | 6 | seconds | 1267 | full_recorded_count | — | summary_only |

### Evaluation `f72a3017e876a62d`

Protocol: `0d17fac1d5f2efd5` (manifest_recorded); original token summary: `4.72` B.

[Original result](../../../evidence/imported/moe1b/suite-step_00004499-1673608.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 9.5000% | fraction | 800 | full_recorded_count | 76/800 | sample_flags_recounted |
| cruxeval | cruxeval_o | 20.2500% | fraction | 800 | full_recorded_count | 162/800 | sample_flags_recounted |
| cruxeval | seconds | 214 | seconds | 800 | full_recorded_count | — | summary_only |
| fim | multi.exact | 0.0000% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_excess_tokens | 96.2 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_overgen_chars | 269.7 | characters | 117 | full_recorded_count | — | summary_only |
| fim | multi.p95_excess_tokens | 179 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.pass_rate | 8.5500% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.suffix_dup | 17.0900% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.terminated | 45.3000% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | seconds | 143 | seconds | — | not_recorded | — | summary_only |
| fim | single.exact | 12.8000% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_excess_tokens | 101.9 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_overgen_chars | 297.1 | characters | 164 | full_recorded_count | — | summary_only |
| fim | single.p95_excess_tokens | 188 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.pass_rate | 19.5100% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.suffix_dup | 17.0700% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.terminated | 48.1700% | fraction | 164 | full_recorded_count | — | summary_only |
| gsm8k | gsm8k | 6.4400% | fraction | 1319 | full_recorded_count | — | summary_only |
| gsm8k | seconds | 882 | seconds | 1319 | full_recorded_count | — | summary_only |
| heldout | seconds | 56 | seconds | — | not_recorded | — | summary_only |
| heldout | stackv2-edu | 1.016100 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | swallow-code | 0.570000 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | web | 2.751100 | cross_entropy | — | not_recorded | — | summary_only |
| humaneval | humaneval | 19.5100% | fraction | 164 | full_recorded_count | 32/164 | sample_flags_recounted |
| humaneval | humaneval_plus | 18.2900% | fraction | 164 | full_recorded_count | 30/164 | sample_flags_recounted |
| humaneval | seconds | 214 | seconds | 164 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v1 | 0.5000% | fraction | 400 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v3 | 0.0000% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v5 | 0.6000% | fraction | 167 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | lcb_release_v6_cumulative | 0.2800% | fraction | 1055 | full_recorded_count | — | summary_only |
| lcb | lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | seconds | 3468 | seconds | — | not_recorded | — | summary_only |
| mbpp | mbpp | 19.8400% | fraction | 378 | full_recorded_count | 75/378 | sample_flags_recounted |
| mbpp | mbpp_plus | 15.8700% | fraction | 378 | full_recorded_count | 60/378 | sample_flags_recounted |
| mbpp | seconds | 469 | seconds | 378 | full_recorded_count | — | summary_only |

## Provenance

- [evidence/imported/moe1b/mc-dense3b-step4499.json](../../../evidence/imported/moe1b/mc-dense3b-step4499.json)
- [evidence/imported/moe1b/suite-step_00004499-1673608.json](../../../evidence/imported/moe1b/suite-step_00004499-1673608.json)

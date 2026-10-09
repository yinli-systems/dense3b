# dense3b — checkpoint 1499

[All checkpoints](../CHECKPOINTS.md)

**Storage:** `retained_model_files`. **Evaluation coverage:** `primary_task_counts_present`.

**Tokens:** 1.572864 B; basis: `constant_observed_schedule_reconstruction_not_metadata_verified`. Schedule reconstructions are explicitly not metadata-verified; they describe logical stream progress and exclude replayed work after rollbacks.

**Evidence types:** metadata, filesystem_inventory, evaluation_summary.

**Missing primary task families:** none at the recorded full sample counts.

**Recorded weight hashes:** `333fdee25243a8dfe75e5f73a893e12026eb0eb6b944d409c8896550f3e4baf8`. These are hashes stated in evaluation manifests, not new checksums from this audit.

## Checkpoint locations

| Path | Model bytes | Optimizer shard files | COMPLETE marker |
|---|---:|---:|---|
| `/data/run01/scxi253/code3b-train/milestones/step_00001499` | 6492811343 | 0 | False |

Historical metadata, results and resume traces are retained even when the model weights were pruned. A resume trace only establishes that the trainer reported a resumed start; it is weaker than a saved metadata file.

## Training observations near this checkpoint

Each row keeps its original job and actual observed step. An observation up to 10 steps earlier is not relabeled as a checkpoint measurement.

| Job | Observed step | Train loss | Validation loss | Source |
|---|---:|---:|---:|---|
| 0 | 1499 | 2.908328 | 2.811359 | [source](../../../evidence/raw/results/ab-bf16-1666412/metrics.jsonl#L1500) |
| 1667439 | 1499 | 1.403862 | 0.996152 | [source](../../../evidence/raw/results/train-1667439/metrics.jsonl#L1500) |

## Every evaluation version

A later file is not automatically better. Protocol fingerprints, checkpoint hashes, sample counts, smoke runs and re-runs stay separate. Percent signs are used only for fractions; loss and router statistics keep their native units.

### Evaluation `037d479dd478d738`

Protocol: `f741e9571054892a` (manifest_recorded); original token summary: `1.57` B.

[Original result](../../../evidence/raw/results/suite-step_00001499.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 2.5000% | fraction | 800 | full_recorded_count | — | summary_only |
| cruxeval | cruxeval_o | 0.0000% | fraction | 800 | full_recorded_count | — | summary_only |
| cruxeval | seconds | 179 | seconds | 800 | full_recorded_count | — | summary_only |
| fim | multi.exact | 0.0000% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_excess_tokens | 153.9 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_overgen_chars | 464.6 | characters | 117 | full_recorded_count | — | summary_only |
| fim | multi.p95_excess_tokens | 183 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.pass_rate | 1.7100% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.suffix_dup | 22.2200% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.terminated | 9.4000% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | seconds | 143 | seconds | — | not_recorded | — | summary_only |
| fim | single.exact | 0.0000% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_excess_tokens | 171 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_overgen_chars | 528.1 | characters | 164 | full_recorded_count | — | summary_only |
| fim | single.p95_excess_tokens | 188 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.pass_rate | 4.8800% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.suffix_dup | 28.0500% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.terminated | 7.9300% | fraction | 164 | full_recorded_count | — | summary_only |
| gsm8k | gsm8k | 2.5800% | fraction | 1319 | full_recorded_count | — | summary_only |
| gsm8k | seconds | 985 | seconds | 1319 | full_recorded_count | — | summary_only |
| heldout | seconds | 47 | seconds | — | not_recorded | — | summary_only |
| heldout | stackv2-edu | 1.266000 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | swallow-code | 0.732900 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | web | 3.244600 | cross_entropy | — | not_recorded | — | summary_only |
| humaneval | humaneval | 6.7100% | fraction | 164 | full_recorded_count | — | summary_only |
| humaneval | humaneval_plus | 6.1000% | fraction | 164 | full_recorded_count | — | summary_only |
| humaneval | seconds | 178 | seconds | 164 | full_recorded_count | — | summary_only |
| mbpp | mbpp | 5.2900% | fraction | 378 | full_recorded_count | — | summary_only |
| mbpp | mbpp_plus | 4.5000% | fraction | 378 | full_recorded_count | — | summary_only |
| mbpp | seconds | 478 | seconds | 378 | full_recorded_count | — | summary_only |

### Evaluation `1ff767d0fe38c5fb`

Protocol: `0a91bce8b95553c6` (manifest_recorded); original token summary: `1.57` B.

[Original result](../../../evidence/raw/results/suite-step_00001499-1670852.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| cruxeval | cruxeval_o | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| cruxeval | seconds | 2 | seconds | 6 | smoke_test | — | summary_only |
| fim | multi.exact | 0.0000% | fraction | 4 | smoke_test | — | summary_only |
| fim | multi.mean_excess_tokens | 176.2 | tokens | 4 | smoke_test | — | summary_only |
| fim | multi.mean_overgen_chars | 588.5 | characters | 4 | smoke_test | — | summary_only |
| fim | multi.p95_excess_tokens | 180 | tokens | 4 | smoke_test | — | summary_only |
| fim | multi.pass_rate | 0.0000% | fraction | 4 | smoke_test | — | summary_only |
| fim | multi.suffix_dup | 0.0000% | fraction | 4 | smoke_test | — | summary_only |
| fim | multi.terminated | 0.0000% | fraction | 4 | smoke_test | — | summary_only |
| fim | seconds | 8 | seconds | — | not_recorded | — | summary_only |
| fim | single.exact | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| fim | single.mean_excess_tokens | 167.5 | tokens | 6 | smoke_test | — | summary_only |
| fim | single.mean_overgen_chars | 578.7 | characters | 6 | smoke_test | — | summary_only |
| fim | single.p95_excess_tokens | 188 | tokens | 6 | smoke_test | — | summary_only |
| fim | single.pass_rate | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| fim | single.suffix_dup | 16.6700% | fraction | 6 | smoke_test | — | summary_only |
| fim | single.terminated | 16.6700% | fraction | 6 | smoke_test | — | summary_only |
| gsm8k | gsm8k | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| gsm8k | seconds | 12 | seconds | 6 | smoke_test | — | summary_only |
| heldout | seconds | 64 | seconds | — | not_recorded | — | summary_only |
| heldout | stackv2-edu | 1.266000 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | swallow-code | 0.732900 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | web | 3.244600 | cross_entropy | — | not_recorded | — | summary_only |
| humaneval | humaneval | 0.0000% | fraction | 6 | smoke_test | 0/6 | sample_flags_recounted |
| humaneval | humaneval_plus | 0.0000% | fraction | 6 | smoke_test | 0/6 | sample_flags_recounted |
| humaneval | seconds | 7 | seconds | 6 | smoke_test | — | summary_only |
| lcb | lcb_added_in_v1 | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| lcb | lcb_added_in_v2 | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| lcb | lcb_added_in_v3 | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| lcb | lcb_added_in_v4 | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| lcb | lcb_added_in_v5 | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| lcb | lcb_added_in_v6 | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| lcb | lcb_release_v6_cumulative | 0.0000% | fraction | 36 | subset | — | summary_only |
| lcb | lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 6 | smoke_test | — | summary_only |
| lcb | seconds | 130 | seconds | — | not_recorded | — | summary_only |
| mbpp | mbpp | 16.6700% | fraction | 6 | smoke_test | 1/6 | sample_flags_recounted |
| mbpp | mbpp_plus | 16.6700% | fraction | 6 | smoke_test | 1/6 | sample_flags_recounted |
| mbpp | seconds | 19 | seconds | 6 | smoke_test | — | summary_only |

### Evaluation `2064e48b7be1d2a0`

Protocol: `b5f55b7a1481a80f` (manifest_recorded); original token summary: `1.57` B.

[Original result](../../../evidence/raw/results/suite-step_00001499-lcb-crux.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 2.5000% | fraction | 800 | full_recorded_count | 20/800 | sample_flags_recounted |
| cruxeval | cruxeval_o | 12.0000% | fraction | 800 | full_recorded_count | 96/800 | sample_flags_recounted |
| cruxeval | seconds | 184 | seconds | 800 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v1 | 0.0000% | fraction | 400 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v3 | 0.0000% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v5 | 0.0000% | fraction | 167 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | lcb_release_v6_cumulative | 0.0000% | fraction | 1055 | full_recorded_count | — | summary_only |
| lcb | lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | seconds | 3222 | seconds | — | not_recorded | — | summary_only |

### Evaluation `6d48e5da15b3b278`

Protocol: `unversioned-legacy` (unversioned); original token summary: `1.57` B.

[Original result](../../../evidence/raw/results/eval-step_00001499.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| gsm8k | gsm8k | 2.5000% | fraction | 200 | subset | — | summary_count_consistent_not_sample_verified |
| heldout | stackv2-edu | 1.266000 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | swallow-code | 0.732900 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | web | 3.244600 | cross_entropy | — | not_recorded | — | summary_only |
| humaneval | humaneval | 7.3200% | fraction | 164 | full_recorded_count | — | summary_count_consistent_not_sample_verified |

### Evaluation `eefbd27cfb6d2b2c`

Protocol: `unversioned-mc` (unversioned); original token summary: `None` B.

[Original result](../../../evidence/imported/moe1b/mc-dense3b-step1499.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| arc_challenge | acc | 21.6700% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | acc_norm | 25.9400% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | seconds | 14 | seconds | 1172 | full_recorded_count | — | summary_only |
| arc_easy | acc | 50.6300% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | acc_norm | 45.9600% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | seconds | 24 | seconds | 2376 | full_recorded_count | — | summary_only |
| boolq | acc | 60.2400% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | acc_norm | 62.1700% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | seconds | 57 | seconds | 3270 | full_recorded_count | — | summary_only |
| hellaswag | acc | 27.5500% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | acc_norm | 29.0000% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | seconds | 202 | seconds | 10042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc | 25.2500% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc_norm | 26.7500% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | seconds | 255 | seconds | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc | 25.2200% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc_norm | 25.2200% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | seconds | 2232 | seconds | 14042 | full_recorded_count | — | summary_only |
| openbookqa | acc | 17.2000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | acc_norm | 25.6000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | seconds | 4 | seconds | 500 | full_recorded_count | — | summary_only |
| sciq | acc | 75.5000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | acc_norm | 67.9000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | seconds | 29 | seconds | 1000 | full_recorded_count | — | summary_only |
| winogrande | acc | 52.1700% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | acc_norm | 52.1700% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | seconds | 5 | seconds | 1267 | full_recorded_count | — | summary_only |

## Provenance

- [evidence/checkpoints/milestones/step_00001499/meta.json](../../../evidence/checkpoints/milestones/step_00001499/meta.json)
- [provenance/checkpoint_inventory.json](../../../provenance/checkpoint_inventory.json)
- [evidence/raw/results/eval-step_00001499.json](../../../evidence/raw/results/eval-step_00001499.json)
- [evidence/raw/results/suite-step_00001499-1670852.json](../../../evidence/raw/results/suite-step_00001499-1670852.json)
- [evidence/raw/results/suite-step_00001499-lcb-crux.json](../../../evidence/raw/results/suite-step_00001499-lcb-crux.json)
- [evidence/raw/results/suite-step_00001499.json](../../../evidence/raw/results/suite-step_00001499.json)
- [evidence/imported/moe1b/mc-dense3b-step1499.json](../../../evidence/imported/moe1b/mc-dense3b-step1499.json)

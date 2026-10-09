# dense3b — checkpoint 4999

[All checkpoints](../CHECKPOINTS.md)

**Storage:** `retained_model_files`. **Evaluation coverage:** `primary_task_counts_present`.

**Tokens:** 5.242880 B; basis: `constant_observed_schedule_reconstruction_not_metadata_verified`. Schedule reconstructions are explicitly not metadata-verified; they describe logical stream progress and exclude replayed work after rollbacks.

**Evidence types:** metadata, filesystem_inventory, evaluation_summary.

**Missing primary task families:** none at the recorded full sample counts.

**Recorded weight hashes:** `35552cb96f8e6a993d5a41cf062716efa6b218bddb5db849f137534dbc5d1859`. These are hashes stated in evaluation manifests, not new checksums from this audit.

## Checkpoint locations

| Path | Model bytes | Optimizer shard files | COMPLETE marker |
|---|---:|---:|---|
| `/data/run01/scxi253/code3b-train/ckpt-v0/step_00004999` | 6492811343 | 0 | False |
| `/data/run01/scxi253/code3b-train/milestones/step_00004999` | 6492811343 | 0 | False |

Historical metadata, results and resume traces are retained even when the model weights were pruned. A resume trace only establishes that the trainer reported a resumed start; it is weaker than a saved metadata file.

## Training observations near this checkpoint

Each row keeps its original job and actual observed step. An observation up to 10 steps earlier is not relabeled as a checkpoint measurement.

| Job | Observed step | Train loss | Validation loss | Source |
|---|---:|---:|---:|---|
| 1667439 | 4999 | 1.042111 | 0.789960 | [source](../../../evidence/raw/results/train-1667439/metrics.jsonl#L5000) |

## Every evaluation version

A later file is not automatically better. Protocol fingerprints, checkpoint hashes, sample counts, smoke runs and re-runs stay separate. Percent signs are used only for fractions; loss and router statistics keep their native units.

### Evaluation `36a83f7a9d6a9597`

Protocol: `0d17fac1d5f2efd5` (manifest_recorded); original token summary: `5.24` B.

[Original result](../../../evidence/imported/moe1b/suite-step_00004999-1673758.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 8.8700% | fraction | 800 | full_recorded_count | 71/800 | sample_flags_recounted |
| cruxeval | cruxeval_o | 19.3800% | fraction | 800 | full_recorded_count | 155/800 | sample_flags_recounted |
| cruxeval | seconds | 236 | seconds | 800 | full_recorded_count | — | summary_only |
| fim | multi.exact | 3.4200% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_excess_tokens | 88.4 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_overgen_chars | 250.5 | characters | 117 | full_recorded_count | — | summary_only |
| fim | multi.p95_excess_tokens | 179 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.pass_rate | 14.5300% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.suffix_dup | 7.6900% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.terminated | 49.5700% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | seconds | 148 | seconds | — | not_recorded | — | summary_only |
| fim | single.exact | 14.6300% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_excess_tokens | 95.4 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_overgen_chars | 271.6 | characters | 164 | full_recorded_count | — | summary_only |
| fim | single.p95_excess_tokens | 188 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.pass_rate | 22.5600% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.suffix_dup | 18.2900% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.terminated | 49.3900% | fraction | 164 | full_recorded_count | — | summary_only |
| gsm8k | gsm8k | 8.4200% | fraction | 1319 | full_recorded_count | — | summary_only |
| gsm8k | seconds | 865 | seconds | 1319 | full_recorded_count | — | summary_only |
| heldout | seconds | 52 | seconds | — | not_recorded | — | summary_only |
| heldout | stackv2-edu | 1.003100 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | swallow-code | 0.560600 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | web | 2.708600 | cross_entropy | — | not_recorded | — | summary_only |
| humaneval | humaneval | 21.3400% | fraction | 164 | full_recorded_count | 35/164 | sample_flags_recounted |
| humaneval | humaneval_plus | 18.9000% | fraction | 164 | full_recorded_count | 31/164 | sample_flags_recounted |
| humaneval | seconds | 165 | seconds | 164 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v1 | 0.2500% | fraction | 400 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v3 | 0.9900% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v5 | 0.6000% | fraction | 167 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | lcb_release_v6_cumulative | 0.2800% | fraction | 1055 | full_recorded_count | — | summary_only |
| lcb | lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | seconds | 3639 | seconds | — | not_recorded | — | summary_only |
| mbpp | mbpp | 29.8900% | fraction | 378 | full_recorded_count | 113/378 | sample_flags_recounted |
| mbpp | mbpp_plus | 23.0200% | fraction | 378 | full_recorded_count | 87/378 | sample_flags_recounted |
| mbpp | seconds | 443 | seconds | 378 | full_recorded_count | — | summary_only |

### Evaluation `64cfa7191be500c8`

Protocol: `b5f55b7a1481a80f` (manifest_recorded); original token summary: `5.24` B.

[Original result](../../../evidence/raw/results/suite-step_00004999.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 8.8700% | fraction | 800 | full_recorded_count | 71/800 | sample_flags_recounted |
| cruxeval | cruxeval_o | 19.3800% | fraction | 800 | full_recorded_count | 155/800 | sample_flags_recounted |
| cruxeval | seconds | 252 | seconds | 800 | full_recorded_count | — | summary_only |
| fim | multi.exact | 3.4200% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_excess_tokens | 88.4 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_overgen_chars | 250.5 | characters | 117 | full_recorded_count | — | summary_only |
| fim | multi.p95_excess_tokens | 179 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.pass_rate | 14.5300% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.suffix_dup | 7.6900% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.terminated | 49.5700% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | seconds | 149 | seconds | — | not_recorded | — | summary_only |
| fim | single.exact | 14.6300% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_excess_tokens | 95.4 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_overgen_chars | 271.6 | characters | 164 | full_recorded_count | — | summary_only |
| fim | single.p95_excess_tokens | 188 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.pass_rate | 22.5600% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.suffix_dup | 18.2900% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.terminated | 49.3900% | fraction | 164 | full_recorded_count | — | summary_only |
| gsm8k | gsm8k | 8.4200% | fraction | 1319 | full_recorded_count | — | summary_only |
| gsm8k | seconds | 863 | seconds | 1319 | full_recorded_count | — | summary_only |
| heldout | seconds | 51 | seconds | — | not_recorded | — | summary_only |
| heldout | stackv2-edu | 1.003100 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | swallow-code | 0.560600 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | web | 2.708600 | cross_entropy | — | not_recorded | — | summary_only |
| humaneval | humaneval | 21.3400% | fraction | 164 | full_recorded_count | 35/164 | sample_flags_recounted |
| humaneval | humaneval_plus | 18.9000% | fraction | 164 | full_recorded_count | 31/164 | sample_flags_recounted |
| humaneval | seconds | 165 | seconds | 164 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v1 | 0.2500% | fraction | 400 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v3 | 0.9900% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v5 | 0.6000% | fraction | 167 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | lcb_release_v6_cumulative | 0.2800% | fraction | 1055 | full_recorded_count | — | summary_only |
| lcb | lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | seconds | 3630 | seconds | — | not_recorded | — | summary_only |
| mbpp | mbpp | 29.8900% | fraction | 378 | full_recorded_count | 113/378 | sample_flags_recounted |
| mbpp | mbpp_plus | 23.0200% | fraction | 378 | full_recorded_count | 87/378 | sample_flags_recounted |
| mbpp | seconds | 442 | seconds | 378 | full_recorded_count | — | summary_only |

### Evaluation `94194351206854bb`

Protocol: `unversioned-mc` (unversioned); original token summary: `None` B.

[Original result](../../../evidence/imported/moe1b/mc-dense3b-step4999.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| arc_challenge | acc | 27.9900% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | acc_norm | 30.2900% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | seconds | 15 | seconds | 1172 | full_recorded_count | — | summary_only |
| arc_easy | acc | 62.6300% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | acc_norm | 55.4700% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | seconds | 25 | seconds | 2376 | full_recorded_count | — | summary_only |
| boolq | acc | 59.3000% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | acc_norm | 62.0500% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | seconds | 61 | seconds | 3270 | full_recorded_count | — | summary_only |
| hellaswag | acc | 30.4300% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | acc_norm | 34.5200% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | seconds | 218 | seconds | 10042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc | 27.9700% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc_norm | 29.7700% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | seconds | 273 | seconds | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc | 27.0800% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc_norm | 27.0800% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | seconds | 2383 | seconds | 14042 | full_recorded_count | — | summary_only |
| openbookqa | acc | 22.0000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | acc_norm | 32.4000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | seconds | 5 | seconds | 500 | full_recorded_count | — | summary_only |
| sciq | acc | 84.0000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | acc_norm | 78.6000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | seconds | 31 | seconds | 1000 | full_recorded_count | — | summary_only |
| winogrande | acc | 49.8000% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | acc_norm | 49.8000% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | seconds | 6 | seconds | 1267 | full_recorded_count | — | summary_only |

## Provenance

- [evidence/checkpoints/ckpt-v0/step_00004999/meta.json](../../../evidence/checkpoints/ckpt-v0/step_00004999/meta.json)
- [provenance/checkpoint_inventory.json](../../../provenance/checkpoint_inventory.json)
- [evidence/raw/results/suite-step_00004999.json](../../../evidence/raw/results/suite-step_00004999.json)
- [evidence/imported/moe1b/mc-dense3b-step4999.json](../../../evidence/imported/moe1b/mc-dense3b-step4999.json)
- [evidence/imported/moe1b/suite-step_00004999-1673758.json](../../../evidence/imported/moe1b/suite-step_00004999-1673758.json)

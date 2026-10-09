# dense3b — checkpoint 2499

[All checkpoints](../CHECKPOINTS.md)

**Storage:** `retained_model_files`. **Evaluation coverage:** `partial_task_coverage`.

**Tokens:** 2.621440 B; basis: `constant_observed_schedule_reconstruction_not_metadata_verified`. Schedule reconstructions are explicitly not metadata-verified; they describe logical stream progress and exclude replayed work after rollbacks.

**Evidence types:** metadata, filesystem_inventory, evaluation_summary.

**Missing primary task families:** arc_easy, arc_challenge, hellaswag, winogrande, boolq, openbookqa, sciq, mmlu_cf, mmlu_mc5.

**Recorded weight hashes:** `c54d7a89368dec2bd04b603409cc9941a4a21815052163f2965ab1fdaebdd66c`. These are hashes stated in evaluation manifests, not new checksums from this audit.

## Checkpoint locations

| Path | Model bytes | Optimizer shard files | COMPLETE marker |
|---|---:|---:|---|
| `/data/run01/scxi253/code3b-train/milestones/step_00002499` | 6492811343 | 0 | False |

Historical metadata, results and resume traces are retained even when the model weights were pruned. A resume trace only establishes that the trainer reported a resumed start; it is weaker than a saved metadata file.

## Training observations near this checkpoint

Each row keeps its original job and actual observed step. An observation up to 10 steps earlier is not relabeled as a checkpoint measurement.

| Job | Observed step | Train loss | Validation loss | Source |
|---|---:|---:|---:|---|
| 1667439 | 2499 | 1.149617 | 0.889248 | [source](../../../evidence/raw/results/train-1667439/metrics.jsonl#L2500) |

## Every evaluation version

A later file is not automatically better. Protocol fingerprints, checkpoint hashes, sample counts, smoke runs and re-runs stay separate. Percent signs are used only for fractions; loss and router statistics keep their native units.

### Evaluation `2b5f512e6227ca9f`

Protocol: `b5f55b7a1481a80f` (manifest_recorded); original token summary: `2.62` B.

[Original result](../../../evidence/raw/results/suite-step_00002499.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 6.7500% | fraction | 800 | full_recorded_count | 54/800 | sample_flags_recounted |
| cruxeval | cruxeval_o | 15.8800% | fraction | 800 | full_recorded_count | 127/800 | sample_flags_recounted |
| cruxeval | seconds | 264 | seconds | 800 | full_recorded_count | — | summary_only |
| fim | multi.exact | 0.0000% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_excess_tokens | 114.6 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.mean_overgen_chars | 344.6 | characters | 117 | full_recorded_count | — | summary_only |
| fim | multi.p95_excess_tokens | 183 | tokens | 117 | full_recorded_count | — | summary_only |
| fim | multi.pass_rate | 4.2700% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.suffix_dup | 14.5300% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | multi.terminated | 37.6100% | fraction | 117 | full_recorded_count | — | summary_only |
| fim | seconds | 155 | seconds | — | not_recorded | — | summary_only |
| fim | single.exact | 7.9300% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_excess_tokens | 114 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.mean_overgen_chars | 348 | characters | 164 | full_recorded_count | — | summary_only |
| fim | single.p95_excess_tokens | 188 | tokens | 164 | full_recorded_count | — | summary_only |
| fim | single.pass_rate | 10.3700% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.suffix_dup | 22.5600% | fraction | 164 | full_recorded_count | — | summary_only |
| fim | single.terminated | 43.2900% | fraction | 164 | full_recorded_count | — | summary_only |
| gsm8k | gsm8k | 3.3400% | fraction | 1319 | full_recorded_count | — | summary_only |
| gsm8k | seconds | 937 | seconds | 1319 | full_recorded_count | — | summary_only |
| heldout | seconds | 60 | seconds | — | not_recorded | — | summary_only |
| heldout | stackv2-edu | 1.122400 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | swallow-code | 0.638400 | cross_entropy | — | not_recorded | — | summary_only |
| heldout | web | 2.972700 | cross_entropy | — | not_recorded | — | summary_only |
| humaneval | humaneval | 14.6300% | fraction | 164 | full_recorded_count | 24/164 | sample_flags_recounted |
| humaneval | humaneval_plus | 13.4100% | fraction | 164 | full_recorded_count | 22/164 | sample_flags_recounted |
| humaneval | seconds | 241 | seconds | 164 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v1 | 0.0000% | fraction | 400 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v3 | 0.0000% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v5 | 0.0000% | fraction | 167 | full_recorded_count | — | summary_only |
| lcb | lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | lcb_release_v6_cumulative | 0.0000% | fraction | 1055 | full_recorded_count | — | summary_only |
| lcb | lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | — | summary_only |
| lcb | seconds | 3118 | seconds | — | not_recorded | — | summary_only |
| mbpp | mbpp | 16.1400% | fraction | 378 | full_recorded_count | 61/378 | sample_flags_recounted |
| mbpp | mbpp_plus | 12.4300% | fraction | 378 | full_recorded_count | 47/378 | sample_flags_recounted |
| mbpp | seconds | 514 | seconds | 378 | full_recorded_count | — | summary_only |

## Provenance

- [evidence/checkpoints/milestones/step_00002499/meta.json](../../../evidence/checkpoints/milestones/step_00002499/meta.json)
- [provenance/checkpoint_inventory.json](../../../provenance/checkpoint_inventory.json)
- [evidence/raw/results/suite-step_00002499.json](../../../evidence/raw/results/suite-step_00002499.json)

# dense3b — checkpoint 5499

[All checkpoints](../CHECKPOINTS.md)

**Storage:** `retained_model_files`. **Evaluation coverage:** `primary_task_counts_present`.

**Tokens:** 5.767168 B; basis: `constant_observed_schedule_reconstruction_not_metadata_verified`. Schedule reconstructions are explicitly not metadata-verified; they describe logical stream progress and exclude replayed work after rollbacks.

**Evidence types:** metadata, filesystem_inventory, resume_start_inference, evaluation_summary.

**Missing primary task families:** none at the recorded full sample counts.

**Recorded weight hashes:** `58c1ed209abbe6a23c149ac4b86b7d1f8a5de4229fb8a6979f65d88708536497`. These are hashes stated in evaluation manifests, not new checksums from this audit.

## Checkpoint locations

| Path | Model bytes | Optimizer shard files | COMPLETE marker |
|---|---:|---:|---|
| `/data/run01/scxi253/code3b-train/ckpt-v0/step_00005499` | 6492811343 | 8 | True |

Historical metadata, results and resume traces are retained even when the model weights were pruned. A resume trace only establishes that the trainer reported a resumed start; it is weaker than a saved metadata file.

## Training observations near this checkpoint

Each row keeps its original job and actual observed step. An observation up to 10 steps earlier is not relabeled as a checkpoint measurement.

| Job | Observed step | Train loss | Validation loss | Source |
|---|---:|---:|---:|---|
| 1667439 | 5499 | 1.100436 | 0.781645 | [source](../../../evidence/raw/results/train-1667439/metrics.jsonl#L5500) |

## Every evaluation version

A later file is not automatically better. Protocol fingerprints, checkpoint hashes, sample counts, smoke runs and re-runs stay separate. Percent signs are used only for fractions; loss and router statistics keep their native units.

### Evaluation `40f8ea54be656dc6`

Protocol: `unversioned-mc` (unversioned); original token summary: `None` B.

[Original result](../../../evidence/imported/moe1b/mc-dense3b-step5499.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| arc_challenge | acc | 28.1600% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | acc_norm | 31.4000% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | seconds | 15 | seconds | 1172 | full_recorded_count | — | summary_only |
| arc_easy | acc | 60.4400% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | acc_norm | 54.5500% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | seconds | 25 | seconds | 2376 | full_recorded_count | — | summary_only |
| boolq | acc | 62.1700% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | acc_norm | 62.2000% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | seconds | 59 | seconds | 3270 | full_recorded_count | — | summary_only |
| hellaswag | acc | 30.3900% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | acc_norm | 34.5200% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | seconds | 213 | seconds | 10042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc | 27.9000% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc_norm | 29.0900% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | seconds | 267 | seconds | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc | 25.1800% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc_norm | 25.1800% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | seconds | 2322 | seconds | 14042 | full_recorded_count | — | summary_only |
| openbookqa | acc | 23.0000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | acc_norm | 32.8000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | seconds | 5 | seconds | 500 | full_recorded_count | — | summary_only |
| sciq | acc | 85.0000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | acc_norm | 79.3000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | seconds | 31 | seconds | 1000 | full_recorded_count | — | summary_only |
| winogrande | acc | 52.0900% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | acc_norm | 52.0900% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | seconds | 6 | seconds | 1267 | full_recorded_count | — | summary_only |

### Evaluation `8c0da3676eaebe91`

Protocol: `0d17fac1d5f2efd5` (manifest_recorded); original token summary: `5.77` B.

[Original result](../../../evidence/imported/moe1b/suite-step_00005499-1675015.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 19.7500% | fraction | 800 | full_recorded_count | 158/800 | sample_flags_recounted |
| cruxeval | cruxeval_o | 20.7500% | fraction | 800 | full_recorded_count | 166/800 | sample_flags_recounted |
| cruxeval | seconds | 261 | seconds | 800 | full_recorded_count | — | summary_only |
| gsm8k | gsm8k | 7.5800% | fraction | 1319 | full_recorded_count | — | summary_only |
| gsm8k | seconds | 774 | seconds | 1319 | full_recorded_count | — | summary_only |
| humaneval | humaneval | 21.9500% | fraction | 164 | full_recorded_count | 36/164 | sample_flags_recounted |
| humaneval | humaneval_plus | 20.1200% | fraction | 164 | full_recorded_count | 33/164 | sample_flags_recounted |
| humaneval | seconds | 205 | seconds | 164 | full_recorded_count | — | summary_only |
| mbpp | mbpp | 26.4600% | fraction | 378 | full_recorded_count | 100/378 | sample_flags_recounted |
| mbpp | mbpp_plus | 21.9600% | fraction | 378 | full_recorded_count | 83/378 | sample_flags_recounted |
| mbpp | seconds | 515 | seconds | 378 | full_recorded_count | — | summary_only |

### Evaluation `c6984406fc92ac78`

Protocol: `unversioned-domain_probe` (unversioned); original token summary: `None` B.

[Original result](../../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| domain_ce | arxiv | 2.236047 | cross_entropy | — | not_recorded | — | summary_only |
| domain_ce | code | 0.757107 | cross_entropy | — | not_recorded | — | summary_only |
| domain_ce | math | 1.679628 | cross_entropy | — | not_recorded | — | summary_only |
| domain_ce | stackexchange | 2.529832 | cross_entropy | — | not_recorded | — | summary_only |
| domain_ce | web | 2.818370 | cross_entropy | — | not_recorded | — | summary_only |
| domain_ce | wiki | 3.563524 | cross_entropy | — | not_recorded | — | summary_only |
| routing_layers | 0.cluster_domain_nmi | 0.11206197 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 1.cluster_domain_nmi | 0.080774157 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 10.cluster_domain_nmi | 0.10559637 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 11.cluster_domain_nmi | 0.10879793 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 12.cluster_domain_nmi | 0.11254614 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 13.cluster_domain_nmi | 0.11511607 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 14.cluster_domain_nmi | 0.12696412 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 15.cluster_domain_nmi | 0.13416384 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 16.cluster_domain_nmi | 0.14521922 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 17.cluster_domain_nmi | 0.16136507 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 18.cluster_domain_nmi | 0.17999891 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 19.cluster_domain_nmi | 0.19871172 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 2.cluster_domain_nmi | 0.093248379 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 20.cluster_domain_nmi | 0.21262451 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 21.cluster_domain_nmi | 0.21969611 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 22.cluster_domain_nmi | 0.22498036 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 23.cluster_domain_nmi | 0.2326744 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 24.cluster_domain_nmi | 0.23900841 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 25.cluster_domain_nmi | 0.24841453 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 26.cluster_domain_nmi | 0.24865736 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 27.cluster_domain_nmi | 0.24029326 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 3.cluster_domain_nmi | 0.11155817 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 4.cluster_domain_nmi | 0.10151723 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 5.cluster_domain_nmi | 0.099616805 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 6.cluster_domain_nmi | 0.099518168 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 7.cluster_domain_nmi | 0.10495937 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 8.cluster_domain_nmi | 0.10821773 | raw_statistic | — | not_recorded | — | summary_only |
| routing_layers | 9.cluster_domain_nmi | 0.11137147 | raw_statistic | — | not_recorded | — | summary_only |

## Provenance

- [evidence/checkpoints/ckpt-v0/step_00005499/meta.json](../../../evidence/checkpoints/ckpt-v0/step_00005499/meta.json)
- [provenance/checkpoint_inventory.json](../../../provenance/checkpoint_inventory.json)
- [evidence/raw/logs/dense3b-train-1674956.out](../../../evidence/raw/logs/dense3b-train-1674956.out)
- [evidence/imported/moe1b/mc-dense3b-step5499.json](../../../evidence/imported/moe1b/mc-dense3b-step5499.json)
- [evidence/imported/moe1b/suite-step_00005499-1675015.json](../../../evidence/imported/moe1b/suite-step_00005499-1675015.json)
- [evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json](../../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json)

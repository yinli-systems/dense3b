# dense3b — checkpoint 5749

[All checkpoints](../CHECKPOINTS.md)

**Storage:** `retained_model_files`. **Evaluation coverage:** `primary_task_counts_present`.

**Tokens:** 6.029312 B; basis: `constant_observed_schedule_reconstruction_not_metadata_verified`. Schedule reconstructions are explicitly not metadata-verified; they describe logical stream progress and exclude replayed work after rollbacks.

**Evidence types:** metadata, filesystem_inventory, resume_start_inference, evaluation_summary.

**Missing primary task families:** none at the recorded full sample counts.

**Recorded weight hashes:** `517f8e13fa018c25e4ddea45870bde2d9d2b07338505c94afd3d82804d07f428`. These are hashes stated in evaluation manifests, not new checksums from this audit.

## Checkpoint locations

| Path | Model bytes | Optimizer shard files | COMPLETE marker |
|---|---:|---:|---|
| `/data/run01/scxi253/code3b-train/ckpt-v0/step_00005749` | 6492811343 | 8 | True |

Historical metadata, results and resume traces are retained even when the model weights were pruned. A resume trace only establishes that the trainer reported a resumed start; it is weaker than a saved metadata file.

## Training observations near this checkpoint

Each row keeps its original job and actual observed step. An observation up to 10 steps earlier is not relabeled as a checkpoint measurement.

| Job | Observed step | Train loss | Validation loss | Source |
|---|---:|---:|---:|---|
| 1674956 | 5749 | 1.053876 | — | [source](../../../evidence/raw/results/train-1674956/metrics.jsonl#L250) |

## Every evaluation version

A later file is not automatically better. Protocol fingerprints, checkpoint hashes, sample counts, smoke runs and re-runs stay separate. Percent signs are used only for fractions; loss and router statistics keep their native units.

### Evaluation `2a644bec76eceeb8`

Protocol: `unversioned-mc` (unversioned); original token summary: `None` B.

[Original result](../../../evidence/checkpoint_audit/2a644bec76eceeb8/mc-step_00005749.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| arc_challenge | acc | 26.7100% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | acc_norm | 30.6300% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | seconds | 10 | seconds | 1172 | full_recorded_count | — | summary_only |
| arc_easy | acc | 63.0500% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | acc_norm | 56.9400% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | seconds | 17 | seconds | 2376 | full_recorded_count | — | summary_only |

### Evaluation `420057a931269855`

Protocol: `9b6abc5a7c13d91b` (manifest_recorded); original token summary: `6.03` B.

[Original result](../../../evidence/raw/results/suite-step_00005749.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| cruxeval | cruxeval_i | 17.2500% | fraction | 800 | full_recorded_count | 138/800 | sample_flags_recounted |
| cruxeval | cruxeval_o | 19.5000% | fraction | 800 | full_recorded_count | 156/800 | sample_flags_recounted |
| cruxeval | seconds | 189 | seconds | 800 | full_recorded_count | — | summary_only |
| gsm8k | gsm8k | 9.6300% | fraction | 1319 | full_recorded_count | — | summary_only |
| gsm8k | seconds | 487 | seconds | 1319 | full_recorded_count | — | summary_only |
| humaneval | humaneval | 23.7800% | fraction | 164 | full_recorded_count | 39/164 | sample_flags_recounted |
| humaneval | humaneval_plus | 21.3400% | fraction | 164 | full_recorded_count | 35/164 | sample_flags_recounted |
| humaneval | seconds | 136 | seconds | 164 | full_recorded_count | — | summary_only |
| mbpp | mbpp | 30.9500% | fraction | 378 | full_recorded_count | 117/378 | sample_flags_recounted |
| mbpp | mbpp_plus | 26.1900% | fraction | 378 | full_recorded_count | 99/378 | sample_flags_recounted |
| mbpp | seconds | 337 | seconds | 378 | full_recorded_count | — | summary_only |

### Evaluation `a8831eeac3919fff`

Protocol: `unversioned-mc` (unversioned); original token summary: `None` B.

[Original result](../../../evidence/checkpoint_audit/a8831eeac3919fff/mc-step_00005749.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| arc_challenge | acc | 26.7100% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | acc_norm | 30.6300% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | seconds | 10 | seconds | 1172 | full_recorded_count | — | summary_only |
| arc_easy | acc | 63.0500% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | acc_norm | 56.9400% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | seconds | 17 | seconds | 2376 | full_recorded_count | — | summary_only |
| boolq | acc | 57.8000% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | acc_norm | 61.9600% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | seconds | 41 | seconds | 3270 | full_recorded_count | — | summary_only |
| hellaswag | acc | 30.5200% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | acc_norm | 34.7700% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | seconds | 142 | seconds | 10042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc | 28.2900% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc_norm | 29.4500% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | seconds | 180 | seconds | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc | 23.9400% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | acc_norm | 23.9400% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_mc5 | seconds | 1612 | seconds | 14042 | full_recorded_count | — | summary_only |
| openbookqa | acc | 21.4000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | acc_norm | 34.4000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | seconds | 4 | seconds | 500 | full_recorded_count | — | summary_only |
| sciq | acc | 85.8000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | acc_norm | 80.6000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | seconds | 21 | seconds | 1000 | full_recorded_count | — | summary_only |
| winogrande | acc | 51.2200% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | acc_norm | 51.2200% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | seconds | 4 | seconds | 1267 | full_recorded_count | — | summary_only |

### Evaluation `a9b5830cfdb3f46d`

Protocol: `unversioned-mc` (unversioned); original token summary: `None` B.

[Original result](../../../evidence/raw/results/mc-step_00005749.json) · [Full version manifest](../evaluation_versions.json)

| Task | Metric | Reported value | Unit | n | Sample scope | Saved flags recount | Evidence status |
|---|---|---:|---|---:|---|---|---|
| arc_challenge | acc | 26.7100% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | acc_norm | 30.6300% | fraction | 1172 | full_recorded_count | — | summary_only |
| arc_challenge | seconds | 10 | seconds | 1172 | full_recorded_count | — | summary_only |
| arc_easy | acc | 63.0500% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | acc_norm | 56.9400% | fraction | 2376 | full_recorded_count | — | summary_only |
| arc_easy | seconds | 17 | seconds | 2376 | full_recorded_count | — | summary_only |
| boolq | acc | 57.8000% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | acc_norm | 61.9600% | fraction | 3270 | full_recorded_count | — | summary_only |
| boolq | seconds | 41 | seconds | 3270 | full_recorded_count | — | summary_only |
| hellaswag | acc | 30.5200% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | acc_norm | 34.7700% | fraction | 10042 | full_recorded_count | — | summary_only |
| hellaswag | seconds | 142 | seconds | 10042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc | 28.2900% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | acc_norm | 29.4500% | fraction | 14042 | full_recorded_count | — | summary_only |
| mmlu_cf | seconds | 180 | seconds | 14042 | full_recorded_count | — | summary_only |
| openbookqa | acc | 21.4000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | acc_norm | 34.4000% | fraction | 500 | full_recorded_count | — | summary_only |
| openbookqa | seconds | 4 | seconds | 500 | full_recorded_count | — | summary_only |
| sciq | acc | 85.8000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | acc_norm | 80.6000% | fraction | 1000 | full_recorded_count | — | summary_only |
| sciq | seconds | 21 | seconds | 1000 | full_recorded_count | — | summary_only |
| winogrande | acc | 51.2200% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | acc_norm | 51.2200% | fraction | 1267 | full_recorded_count | — | summary_only |
| winogrande | seconds | 4 | seconds | 1267 | full_recorded_count | — | summary_only |

## Provenance

- [evidence/checkpoints/ckpt-v0/step_00005749/meta.json](../../../evidence/checkpoints/ckpt-v0/step_00005749/meta.json)
- [provenance/checkpoint_inventory.json](../../../provenance/checkpoint_inventory.json)
- [evidence/checkpoint_audit/3e46059502abfe5b/IgM-dense3b-1680260.out](../../../evidence/checkpoint_audit/3e46059502abfe5b/IgM-dense3b-1680260.out)
- [evidence/checkpoint_audit/a8831eeac3919fff/mc-step_00005749.json](../../../evidence/checkpoint_audit/a8831eeac3919fff/mc-step_00005749.json)
- [evidence/raw/results/suite-step_00005749.json](../../../evidence/raw/results/suite-step_00005749.json)
- [evidence/raw/results/mc-step_00005749.json](../../../evidence/raw/results/mc-step_00005749.json)
- [evidence/checkpoint_audit/2a644bec76eceeb8/mc-step_00005749.json](../../../evidence/checkpoint_audit/2a644bec76eceeb8/mc-step_00005749.json)

# dense3b — checkpoint 2999

[All checkpoints](../CHECKPOINTS.md)

**Storage:** `historical_metadata_only`. **Evaluation coverage:** `no_result_found`.

**Tokens:** 3.145728 B; basis: `constant_observed_schedule_reconstruction_not_metadata_verified`. Schedule reconstructions are explicitly not metadata-verified; they describe logical stream progress and exclude replayed work after rollbacks.

**Evidence types:** metadata.

**Missing primary task families:** humaneval, mbpp, cruxeval, gsm8k, arc_easy, arc_challenge, hellaswag, winogrande, boolq, openbookqa, sciq, mmlu_cf, mmlu_mc5.

**Recorded weight hashes:** not recorded. These are hashes stated in evaluation manifests, not new checksums from this audit.

## Checkpoint locations

| Path | Model bytes | Optimizer shard files | COMPLETE marker |
|---|---:|---:|---|
| Not present in the live inventory | — | — | — |

Historical metadata, results and resume traces are retained even when the model weights were pruned. A resume trace only establishes that the trainer reported a resumed start; it is weaker than a saved metadata file.

## Training observations near this checkpoint

Each row keeps its original job and actual observed step. An observation up to 10 steps earlier is not relabeled as a checkpoint measurement.

| Job | Observed step | Train loss | Validation loss | Source |
|---|---:|---:|---:|---|
| 1667439 | 2999 | 1.200160 | 0.857847 | [source](../../../evidence/raw/results/train-1667439/metrics.jsonl#L3000) |

## Every evaluation version

A later file is not automatically better. Protocol fingerprints, checkpoint hashes, sample counts, smoke runs and re-runs stay separate. Percent signs are used only for fractions; loss and router statistics keep their native units.

**No downstream/probe evaluation result was found for this checkpoint.** Training-loop observations, when available, are listed above. It remains in the inventory; no metric has been invented.

## Provenance

- [evidence/checkpoint_audit/3285c878f050ad92/meta.json](../../../evidence/checkpoint_audit/3285c878f050ad92/meta.json)

# dense3b: every located checkpoint and result

Snapshot: **2026-10-09T18:52:28.188993+00:00**.

**This is an evidence audit, not a new GPU evaluation.** Tensor contents were not read, loaded or re-hashed. A model file present in the snapshot does not prove that it can be deserialized. Missing scores are never zero-filled.

Located **11 checkpoint entries**: **6 with model files present**, **6 with downstream/probe evaluation results**, and **5 with no downstream/probe evaluation result found. Training and validation observations are listed separately. The inventory contains 7 directories; directory aliases do not create extra training steps.

Recovered **18 distinct JSON evaluation versions** and **463 typed metric rows**, including timing and routing statistics. **42 score rows were independently re-aggregated from saved Boolean sample flags**. This validates arithmetic and sample counts, not the execution harness itself.

[All evaluation values (CSV)](all_metrics.csv) · [All training observations](training_observations.csv) · [Unconfirmed scheduled slots](SCHEDULE_COVERAGE.md) · [All numeric results (JSON)](all_metrics.json) · [Every evaluation version](evaluation_versions.json) · [Integrity and protocol issues](ISSUES.md) · [Missing evaluations](GAPS.md) · [Method](METHOD.md)

## dense3b

| Step / full result page | Model file in snapshot | Tokens (B) | Token basis | Result rows | Full-count task families | Missing primary task families |
|---|---|---:|---|---:|---:|---:|
| [1499](checkpoints/dense3b-step_00001499.md) | present | 1.572864 | schedule estimate | 113 | 15/13 | 0 |
| [2499](checkpoints/dense3b-step_00002499.md) | present | 2.621440 | schedule estimate | 39 | 6/13 | 9 |
| [2999](checkpoints/dense3b-step_00002999.md) | not found | 3.145728 | schedule estimate | 0 | 0/13 | 13 |
| [3249](checkpoints/dense3b-step_00003249.md) | not found | 3.407872 | schedule estimate | 0 | 0/13 | 13 |
| [3499](checkpoints/dense3b-step_00003499.md) | not found | 3.670016 | schedule estimate | 0 | 0/13 | 13 |
| [4499](checkpoints/dense3b-step_00004499.md) | not found | 4.718592 | schedule estimate | 66 | 15/13 | 0 |
| [4999](checkpoints/dense3b-step_00004999.md) | present | 5.242880 | schedule estimate | 105 | 15/13 | 0 |
| [5249](checkpoints/dense3b-step_00005249.md) | not found | 5.505024 | schedule estimate | 0 | 0/13 | 13 |
| [5499](checkpoints/dense3b-step_00005499.md) | present | 5.767168 | schedule estimate | 72 | 13/13 | 0 |
| [5749](checkpoints/dense3b-step_00005749.md) | present | 6.029312 | schedule estimate | 68 | 13/13 | 0 |
| [5999](checkpoints/dense3b-step_00005999.md) | present | 6.291456 | schedule estimate | 0 | 0/13 | 13 |


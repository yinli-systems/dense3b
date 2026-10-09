# dense3b — checkpoint backfill status

**Evidence captured: 2026-10-09T20:39:11.654910+00:00.** This is a snapshot, not a live feed.

The bounded campaign uses two one-GPU dependency chains at low scheduling priority. It does not stop or reconfigure training. All submitted jobs and their dependencies are preserved in [the execution plan](../../operations/backfill-20261009T2000Z/dependency-graph.json).

## Checkpoints in this repository

| Preset | Step | Missing tasks scheduled |
|---|---:|---|
| dense3b | 5999 | arc_challenge, arc_easy, boolq, cruxeval, gsm8k, hellaswag, humaneval, mbpp, mmlu_cf, mmlu_mc5, openbookqa, sciq, winogrande |
| dense3b | 2499 | arc_challenge, arc_easy, boolq, hellaswag, mmlu_cf, mmlu_mc5, openbookqa, sciq, winogrande |

## Observed worker states

| Checkpoint | Phase | State | Current task | Job |
|---|---|---|---|---|
| dense3b-step_00005999 | mc-status.json | running | mmlu_mc5 | 1681615 |

## Finished task scores only

Accuracy and byte-normalized accuracy remain separate. Official EvalPlus plus the documented progress-counter fix is a distinct protocol from the old custom Plus harness. No queued or incomplete task receives a placeholder zero.

| Checkpoint | Task | Metric | Score | n |
|---|---|---|---:|---:|
| dense3b-step_00005999 | arc_easy | acc | 63.2155% | 2376 |
| dense3b-step_00005999 | arc_easy | acc_norm | 58.3754% | 2376 |
| dense3b-step_00005999 | arc_challenge | acc | 27.9010% | 1172 |
| dense3b-step_00005999 | arc_challenge | acc_norm | 31.2287% | 1172 |
| dense3b-step_00005999 | hellaswag | acc | 30.9102% | 10042 |
| dense3b-step_00005999 | hellaswag | acc_norm | 35.4511% | 10042 |
| dense3b-step_00005999 | winogrande | acc | 50.0395% | 1267 |
| dense3b-step_00005999 | winogrande | acc_norm | 50.0395% | 1267 |
| dense3b-step_00005999 | boolq | acc | 54.2202% | 3270 |
| dense3b-step_00005999 | boolq | acc_norm | 61.7431% | 3270 |
| dense3b-step_00005999 | openbookqa | acc | 21.2000% | 500 |
| dense3b-step_00005999 | openbookqa | acc_norm | 31.6000% | 500 |
| dense3b-step_00005999 | sciq | acc | 85.6000% | 1000 |
| dense3b-step_00005999 | sciq | acc_norm | 82.1000% | 1000 |
| dense3b-step_00005999 | mmlu_cf | acc | 28.5287% | 14042 |
| dense3b-step_00005999 | mmlu_cf | acc_norm | 30.2236% | 14042 |

Per-example outcome records re-aggregated and checked: **33,669**.

[Raw evidence and hashes](../../evidence/backfill-20261009T2000Z/manifest.json) · [Machine-readable status](status.json) · [Complete execution/scoring implementation](../../operations/backfill-20261009T2000Z/README.md)

A completed choice-test phase is not a completed entire checkpoint evaluation. `scoring-preflight-receipt.json` is a reference-program control, not a model score. All generations, test outcomes and original protocols remain separate.

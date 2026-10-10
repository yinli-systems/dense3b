# dense3b — checkpoint 5999

[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)

**Verified metadata tokens:** — B. Missing metadata is not reconstructed from an assumed batch size.

**Missing full-count task families:** none. Counts alone do not certify evaluation protocol validity.

## Retained file locations

File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.

`/data/run01/scxi253/code3b-train/ckpt-v0/step_00005999` — training_directory; 6492811343 bytes; captured 2026-10-09T23:55:40.432744+00:00.

`/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z/pinned/dense3b/dense3b-step_00005999` — evaluation_hardlink; 6492811343 bytes; captured 2026-10-09T23:24:35.310505+00:00.

## Nearest recorded training observation

Job 1680260, **actual logged step 5999** (not relabeled as checkpoint step): loss 0.928168, MFU 67.97%, global throughput 41,710 tok/s, peak allocated memory 22.387 GB. [Raw log/metrics](../../evidence/sync-20261009T2354Z/training/dense3b/results/train-1680260/metrics.jsonl).

## All recorded evaluation values

Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.

| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |
|---|---|---:|---|---:|---|---|---|---|
| ba80e8ef46 | arc_challenge / acc | 27.9010% | fraction | 1172 | full_recorded_count | 327/1172 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | arc_challenge / acc_norm | 31.2287% | fraction | 1172 | full_recorded_count | 366/1172 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | arc_challenge / seconds | 9.689923 | seconds | 1172 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | arc_easy / acc | 63.2155% | fraction | 2376 | full_recorded_count | 1502/2376 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | arc_easy / acc_norm | 58.3754% | fraction | 2376 | full_recorded_count | 1387/2376 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | arc_easy / seconds | 16.570585 | seconds | 2376 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | boolq / acc | 54.2202% | fraction | 3270 | full_recorded_count | 1773/3270 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | boolq / acc_norm | 61.7431% | fraction | 3270 | full_recorded_count | 2019/3270 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | boolq / seconds | 40.341841 | seconds | 3270 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ed25a9290f | cruxeval / cruxeval_i | 17.3700% | fraction | 800 | full_recorded_count | 139/800 | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | cruxeval / cruxeval_o | 17.7500% | fraction | 800 | full_recorded_count | 142/800 | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | cruxeval / seconds | 617.299100 | seconds | 800 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | gsm8k / gsm8k | 10.7657% | fraction | 1319 | full_recorded_count | 142/1319 | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | gsm8k / seconds | 429.861849 | seconds | 1319 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | gsm8k / strict_accuracy | 10.4625% | fraction | 1319 | full_recorded_count | 138/1319 | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ba80e8ef46 | hellaswag / acc | 30.9102% | fraction | 10042 | full_recorded_count | 3104/10042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | hellaswag / acc_norm | 35.4511% | fraction | 10042 | full_recorded_count | 3560/10042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | hellaswag / seconds | 142.134364 | seconds | 10042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ed25a9290f | humaneval / humaneval | 22.5610% | fraction | 164 | full_recorded_count | 37/164 | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | humaneval / humaneval_plus | 21.3415% | fraction | 164 | full_recorded_count | 35/164 | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | humaneval / seconds | 444.205087 | seconds | 164 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | mbpp / mbpp | 34.6561% | fraction | 378 | full_recorded_count | 131/378 | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | mbpp / mbpp_plus | 26.9841% | fraction | 378 | full_recorded_count | 102/378 | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ed25a9290f | mbpp / seconds | 432.407935 | seconds | 378 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/suite.json) |
| ba80e8ef46 | mmlu_cf / acc | 28.5287% | fraction | 14042 | full_recorded_count | 4006/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | mmlu_cf / acc_norm | 30.2236% | fraction | 14042 | full_recorded_count | 4244/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | mmlu_cf / seconds | 180.614391 | seconds | 14042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | mmlu_mc5 / acc | 24.8042% | fraction | 14042 | full_recorded_count | 3483/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | mmlu_mc5 / acc_norm | 24.8042% | fraction | 14042 | full_recorded_count | 3483/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | mmlu_mc5 / seconds | 1,611.377623 | seconds | 14042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | openbookqa / acc | 21.2000% | fraction | 500 | full_recorded_count | 106/500 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | openbookqa / acc_norm | 31.6000% | fraction | 500 | full_recorded_count | 158/500 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | openbookqa / seconds | 3.050028 | seconds | 500 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | sciq / acc | 85.6000% | fraction | 1000 | full_recorded_count | 856/1000 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | sciq / acc_norm | 82.1000% | fraction | 1000 | full_recorded_count | 821/1000 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | sciq / seconds | 20.829673 | seconds | 1000 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | winogrande / acc | 50.0395% | fraction | 1267 | full_recorded_count | 634/1267 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | winogrande / acc_norm | 50.0395% | fraction | 1267 | full_recorded_count | 634/1267 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |
| ba80e8ef46 | winogrande / seconds | 3.738237 | seconds | 1267 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00005999/mc.json) |

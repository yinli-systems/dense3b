# dense3b — checkpoint 6249

[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)

**Verified metadata tokens:** — B. Missing metadata is not reconstructed from an assumed batch size.

**Missing full-count task families:** none. Counts alone do not certify evaluation protocol validity.

## Retained file locations

File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.

`/data/run01/scxi253/code3b-train/ckpt-v0/step_00006249` — training_directory; 6492811343 bytes; captured 2026-10-09T23:55:40.432744+00:00.

## Nearest recorded training observation

Job 1680260, **actual logged step 6249** (not relabeled as checkpoint step): loss 1.026680, MFU 57.40%, global throughput 35,222 tok/s, peak allocated memory 22.387 GB. [Raw log/metrics](../../evidence/sync-20261009T2354Z/training/dense3b/results/train-1680260/metrics.jsonl).

## All recorded evaluation values

Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.

| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |
|---|---|---:|---|---:|---|---|---|---|
| 084aa5fa1a | arc_challenge / acc | 28.6689% | fraction | 1172 | full_recorded_count | 336/1172 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | arc_challenge / acc_norm | 30.7167% | fraction | 1172 | full_recorded_count | 360/1172 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | arc_challenge / seconds | 9.586850 | seconds | 1172 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | arc_easy / acc | 61.3636% | fraction | 2376 | full_recorded_count | 1458/2376 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | arc_easy / acc_norm | 54.3350% | fraction | 2376 | full_recorded_count | 1291/2376 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | arc_easy / seconds | 16.298137 | seconds | 2376 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | boolq / acc | 62.2018% | fraction | 3270 | full_recorded_count | 2034/3270 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | boolq / acc_norm | 62.1407% | fraction | 3270 | full_recorded_count | 2032/3270 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | boolq / seconds | 39.913743 | seconds | 3270 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| beee551088 | cruxeval / cruxeval_i | 18.1200% | fraction | 800 | full_recorded_count | 145/800 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | cruxeval / cruxeval_o | 17.8700% | fraction | 800 | full_recorded_count | 143/800 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | cruxeval / seconds | 696.230663 | seconds | 800 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | gsm8k / gsm8k | 10.2350% | fraction | 1319 | full_recorded_count | 135/1319 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | gsm8k / seconds | 507.144867 | seconds | 1319 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | gsm8k / strict_accuracy | 10.1592% | fraction | 1319 | full_recorded_count | 134/1319 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| 084aa5fa1a | hellaswag / acc | 31.0695% | fraction | 10042 | full_recorded_count | 3120/10042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | hellaswag / acc_norm | 35.3316% | fraction | 10042 | full_recorded_count | 3548/10042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | hellaswag / seconds | 140.220447 | seconds | 10042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| beee551088 | humaneval / humaneval | 25.6098% | fraction | 164 | full_recorded_count | 42/164 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | humaneval / humaneval_plus | 23.1707% | fraction | 164 | full_recorded_count | 38/164 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | humaneval / seconds | 448.274341 | seconds | 164 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | mbpp / mbpp | 32.5397% | fraction | 378 | full_recorded_count | 123/378 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | mbpp / mbpp_plus | 26.1905% | fraction | 378 | full_recorded_count | 99/378 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| beee551088 | mbpp / seconds | 567.867226 | seconds | 378 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/suite.json) |
| 084aa5fa1a | mmlu_cf / acc | 28.3364% | fraction | 14042 | full_recorded_count | 3979/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | mmlu_cf / acc_norm | 29.7394% | fraction | 14042 | full_recorded_count | 4176/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | mmlu_cf / seconds | 177.748399 | seconds | 14042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | mmlu_mc5 / acc | 25.7656% | fraction | 14042 | full_recorded_count | 3618/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | mmlu_mc5 / acc_norm | 25.7656% | fraction | 14042 | full_recorded_count | 3618/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | mmlu_mc5 / seconds | 1,595.857588 | seconds | 14042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | openbookqa / acc | 22.0000% | fraction | 500 | full_recorded_count | 110/500 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | openbookqa / acc_norm | 32.6000% | fraction | 500 | full_recorded_count | 163/500 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | openbookqa / seconds | 2.998026 | seconds | 500 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | sciq / acc | 84.7000% | fraction | 1000 | full_recorded_count | 847/1000 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | sciq / acc_norm | 80.4000% | fraction | 1000 | full_recorded_count | 804/1000 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | sciq / seconds | 20.641565 | seconds | 1000 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | winogrande / acc | 49.9605% | fraction | 1267 | full_recorded_count | 633/1267 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | winogrande / acc_norm | 49.9605% | fraction | 1267 | full_recorded_count | 633/1267 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |
| 084aa5fa1a | winogrande / seconds | 3.628628 | seconds | 1267 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006249/mc.json) |

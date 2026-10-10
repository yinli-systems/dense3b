# dense3b — checkpoint 6499

[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)

**Verified metadata tokens:** — B. Missing metadata is not reconstructed from an assumed batch size.

**Missing full-count task families:** none. Counts alone do not certify evaluation protocol validity.

## Retained file locations

File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.

`/data/run01/scxi253/code3b-train/ckpt-v0/step_00006499` — training_directory; 6492811343 bytes; captured 2026-10-09T23:55:40.432744+00:00.

## Nearest recorded training observation

Job 1680260, **actual logged step 6499** (not relabeled as checkpoint step): loss 0.963032, MFU 67.81%, global throughput 41,611 tok/s, peak allocated memory 22.387 GB. [Raw log/metrics](../../evidence/sync-20261009T2354Z/training/dense3b/results/train-1680260/metrics.jsonl).

## All recorded evaluation values

Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.

| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |
|---|---|---:|---|---:|---|---|---|---|
| 1ee1d4910f | arc_challenge / acc | 28.6689% | fraction | 1172 | full_recorded_count | 336/1172 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | arc_challenge / acc_norm | 31.0580% | fraction | 1172 | full_recorded_count | 364/1172 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | arc_challenge / seconds | 9.731578 | seconds | 1172 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | arc_easy / acc | 61.9949% | fraction | 2376 | full_recorded_count | 1473/2376 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | arc_easy / acc_norm | 53.4933% | fraction | 2376 | full_recorded_count | 1271/2376 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | arc_easy / seconds | 16.770972 | seconds | 2376 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | boolq / acc | 58.6544% | fraction | 3270 | full_recorded_count | 1918/3270 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | boolq / acc_norm | 62.2324% | fraction | 3270 | full_recorded_count | 2035/3270 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | boolq / seconds | 40.423347 | seconds | 3270 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 3a472a3e06 | cruxeval / cruxeval_i | 13.3800% | fraction | 800 | full_recorded_count | 107/800 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | cruxeval / cruxeval_o | 20.2500% | fraction | 800 | full_recorded_count | 162/800 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | cruxeval / seconds | 490.242046 | seconds | 800 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | gsm8k / gsm8k | 11.5239% | fraction | 1319 | full_recorded_count | 152/1319 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | gsm8k / seconds | 452.905005 | seconds | 1319 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | gsm8k / strict_accuracy | 11.3723% | fraction | 1319 | full_recorded_count | 150/1319 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 1ee1d4910f | hellaswag / acc | 31.1691% | fraction | 10042 | full_recorded_count | 3130/10042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | hellaswag / acc_norm | 35.7499% | fraction | 10042 | full_recorded_count | 3590/10042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | hellaswag / seconds | 143.034881 | seconds | 10042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 3a472a3e06 | humaneval / humaneval | 25.6098% | fraction | 164 | full_recorded_count | 42/164 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | humaneval / humaneval_plus | 25.0000% | fraction | 164 | full_recorded_count | 41/164 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | humaneval / seconds | 327.249271 | seconds | 164 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | mbpp / mbpp | 33.3333% | fraction | 378 | full_recorded_count | 126/378 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | mbpp / mbpp_plus | 26.9841% | fraction | 378 | full_recorded_count | 102/378 | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 3a472a3e06 | mbpp / seconds | 450.821523 | seconds | 378 | full_recorded_count | summary_only | 25371b05730c316d | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/suite.json) |
| 1ee1d4910f | mmlu_cf / acc | 28.5714% | fraction | 14042 | full_recorded_count | 4012/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | mmlu_cf / acc_norm | 29.8462% | fraction | 14042 | full_recorded_count | 4191/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | mmlu_cf / seconds | 181.408309 | seconds | 14042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | mmlu_mc5 / acc | 25.2314% | fraction | 14042 | full_recorded_count | 3543/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | mmlu_mc5 / acc_norm | 25.2314% | fraction | 14042 | full_recorded_count | 3543/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | mmlu_mc5 / seconds | 1,617.093767 | seconds | 14042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | openbookqa / acc | 22.0000% | fraction | 500 | full_recorded_count | 110/500 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | openbookqa / acc_norm | 32.6000% | fraction | 500 | full_recorded_count | 163/500 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | openbookqa / seconds | 3.066601 | seconds | 500 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | sciq / acc | 84.3000% | fraction | 1000 | full_recorded_count | 843/1000 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | sciq / acc_norm | 79.6000% | fraction | 1000 | full_recorded_count | 796/1000 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | sciq / seconds | 20.919619 | seconds | 1000 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | winogrande / acc | 51.9337% | fraction | 1267 | full_recorded_count | 658/1267 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | winogrande / acc_norm | 51.9337% | fraction | 1267 | full_recorded_count | 658/1267 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |
| 1ee1d4910f | winogrande / seconds | 3.744592 | seconds | 1267 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2354Z/new_campaign/dense3b-step_00006499/mc.json) |

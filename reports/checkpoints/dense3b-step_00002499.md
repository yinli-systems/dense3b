# dense3b — checkpoint 2499

[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)

**Verified metadata tokens:** — B. Missing metadata is not reconstructed from an assumed batch size.

**Missing full-count task families:** none. Counts alone do not certify evaluation protocol validity.

## Retained file locations

File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.

`/data/run01/scxi253/code3b-train/milestones/step_00002499` — training_directory; 6492811343 bytes; captured 2026-10-09T23:55:40.432744+00:00.

`/data/run01/scxi253/checkpoint-eval-backfill-20261009T2000Z/pinned/dense3b/dense3b-step_00002499` — evaluation_hardlink; 6492811343 bytes; captured 2026-10-09T23:24:35.310505+00:00.

## Nearest recorded training observation

Job 1667439, **actual logged step 2499** (not relabeled as checkpoint step): loss 1.149617, MFU 64.81%, global throughput 39,768 tok/s, peak allocated memory 22.387 GB. [Raw log/metrics](../../evidence/raw/results/train-1667439/metrics.jsonl).

## All recorded evaluation values

Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.

| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |
|---|---|---:|---|---:|---|---|---|---|
| f18bb08b9b | arc_challenge / acc | 23.9761% | fraction | 1172 | full_recorded_count | 281/1172 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | arc_challenge / acc_norm | 27.8157% | fraction | 1172 | full_recorded_count | 326/1172 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | arc_challenge / seconds | 9.534450 | seconds | 1172 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | arc_easy / acc | 55.8502% | fraction | 2376 | full_recorded_count | 1327/2376 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | arc_easy / acc_norm | 50.5051% | fraction | 2376 | full_recorded_count | 1200/2376 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | arc_easy / seconds | 16.216964 | seconds | 2376 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | boolq / acc | 57.4006% | fraction | 3270 | full_recorded_count | 1877/3270 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | boolq / acc_norm | 62.1407% | fraction | 3270 | full_recorded_count | 2032/3270 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | boolq / seconds | 39.798650 | seconds | 3270 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| 2b5f512e62 | cruxeval / cruxeval_i | 6.7500% | fraction | 800 | full_recorded_count | 54/800 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | cruxeval / cruxeval_o | 15.8800% | fraction | 800 | full_recorded_count | 127/800 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | cruxeval / seconds | 264.000000 | seconds | 800 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / multi.exact | 0.0000% | fraction | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / multi.mean_excess_tokens | 114.600000 | tokens | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / multi.mean_overgen_chars | 344.600000 | characters | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / multi.p95_excess_tokens | 183.000000 | tokens | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / multi.pass_rate | 4.2700% | fraction | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / multi.suffix_dup | 14.5300% | fraction | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / multi.terminated | 37.6100% | fraction | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / seconds | 155.000000 | seconds | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / single.exact | 7.9300% | fraction | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / single.mean_excess_tokens | 114.000000 | tokens | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / single.mean_overgen_chars | 348.000000 | characters | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / single.p95_excess_tokens | 188.000000 | tokens | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / single.pass_rate | 10.3700% | fraction | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / single.suffix_dup | 22.5600% | fraction | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | fim / single.terminated | 43.2900% | fraction | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | gsm8k / gsm8k | 3.3400% | fraction | 1319 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | gsm8k / seconds | 937.000000 | seconds | 1319 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | heldout / seconds | 60.000000 | seconds | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | heldout / stackv2-edu | 1.122400 | cross_entropy | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | heldout / swallow-code | 0.638400 | cross_entropy | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | heldout / web | 2.972700 | cross_entropy | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| f18bb08b9b | hellaswag / acc | 28.8190% | fraction | 10042 | full_recorded_count | 2894/10042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | hellaswag / acc_norm | 31.0098% | fraction | 10042 | full_recorded_count | 3114/10042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | hellaswag / seconds | 139.497975 | seconds | 10042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| 2b5f512e62 | humaneval / humaneval | 14.6300% | fraction | 164 | full_recorded_count | 24/164 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | humaneval / humaneval_plus | 13.4100% | fraction | 164 | full_recorded_count | 22/164 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | humaneval / seconds | 241.000000 | seconds | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / lcb_added_in_v1 | 0.0000% | fraction | 400 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / lcb_added_in_v3 | 0.0000% | fraction | 101 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / lcb_added_in_v5 | 0.0000% | fraction | 167 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / lcb_release_v6_cumulative | 0.0000% | fraction | 1055 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | lcb / seconds | 3,118.000000 | seconds | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | mbpp / mbpp | 16.1400% | fraction | 378 | full_recorded_count | 61/378 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | mbpp / mbpp_plus | 12.4300% | fraction | 378 | full_recorded_count | 47/378 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| 2b5f512e62 | mbpp / seconds | 514.000000 | seconds | 378 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00002499.json) |
| f18bb08b9b | mmlu_cf / acc | 26.5418% | fraction | 14042 | full_recorded_count | 3727/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | mmlu_cf / acc_norm | 28.2723% | fraction | 14042 | full_recorded_count | 3970/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | mmlu_cf / seconds | 177.490610 | seconds | 14042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | mmlu_mc5 / acc | 25.8724% | fraction | 14042 | full_recorded_count | 3633/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | mmlu_mc5 / acc_norm | 25.8724% | fraction | 14042 | full_recorded_count | 3633/14042 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | mmlu_mc5 / seconds | 1,594.276283 | seconds | 14042 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | openbookqa / acc | 18.2000% | fraction | 500 | full_recorded_count | 91/500 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | openbookqa / acc_norm | 29.0000% | fraction | 500 | full_recorded_count | 145/500 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | openbookqa / seconds | 2.992115 | seconds | 500 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | sciq / acc | 82.3000% | fraction | 1000 | full_recorded_count | 823/1000 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | sciq / acc_norm | 76.3000% | fraction | 1000 | full_recorded_count | 763/1000 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | sciq / seconds | 20.589008 | seconds | 1000 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | winogrande / acc | 50.7498% | fraction | 1267 | full_recorded_count | 643/1267 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | winogrande / acc_norm | 50.7498% | fraction | 1267 | full_recorded_count | 643/1267 | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |
| f18bb08b9b | winogrande / seconds | 3.650882 | seconds | 1267 | full_recorded_count | summary_only | d94bc379006a1f93 | [source](../../evidence/sync-20261009T2320Z/campaign/dense3b-step_00002499/mc.json) |

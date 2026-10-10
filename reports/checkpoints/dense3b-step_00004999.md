# dense3b — checkpoint 4999

[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)

**Verified metadata tokens:** — B. Missing metadata is not reconstructed from an assumed batch size.

**Missing full-count task families:** none. Counts alone do not certify evaluation protocol validity.

## Retained file locations

File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.

`/data/run01/scxi253/code3b-train/ckpt-v0/step_00004999` — training_directory; 6492811343 bytes; captured 2026-10-09T23:55:40.432744+00:00.

`/data/run01/scxi253/code3b-train/milestones/step_00004999` — training_directory; 6492811343 bytes; captured 2026-10-09T23:55:40.432744+00:00.

## Nearest recorded training observation

Job 1667439, **actual logged step 4999** (not relabeled as checkpoint step): loss 1.042111, MFU 63.93%, global throughput 39,228 tok/s, peak allocated memory 22.387 GB. [Raw log/metrics](../../evidence/raw/results/train-1667439/metrics.jsonl).

## All recorded evaluation values

Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.

| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |
|---|---|---:|---|---:|---|---|---|---|
| 9419435120 | arc_challenge / acc | 27.9900% | fraction | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | arc_challenge / acc_norm | 30.2900% | fraction | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | arc_challenge / seconds | 15.000000 | seconds | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | arc_easy / acc | 62.6300% | fraction | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | arc_easy / acc_norm | 55.4700% | fraction | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | arc_easy / seconds | 25.000000 | seconds | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | boolq / acc | 59.3000% | fraction | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | boolq / acc_norm | 62.0500% | fraction | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | boolq / seconds | 61.000000 | seconds | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 36a83f7a9d | cruxeval / cruxeval_i | 8.8700% | fraction | 800 | full_recorded_count | 71/800 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | cruxeval / cruxeval_i | 8.8700% | fraction | 800 | full_recorded_count | 71/800 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | cruxeval / cruxeval_o | 19.3800% | fraction | 800 | full_recorded_count | 155/800 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | cruxeval / cruxeval_o | 19.3800% | fraction | 800 | full_recorded_count | 155/800 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | cruxeval / seconds | 236.000000 | seconds | 800 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | cruxeval / seconds | 252.000000 | seconds | 800 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / multi.exact | 3.4200% | fraction | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / multi.exact | 3.4200% | fraction | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / multi.mean_excess_tokens | 88.400000 | tokens | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / multi.mean_excess_tokens | 88.400000 | tokens | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / multi.mean_overgen_chars | 250.500000 | characters | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / multi.mean_overgen_chars | 250.500000 | characters | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / multi.p95_excess_tokens | 179.000000 | tokens | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / multi.p95_excess_tokens | 179.000000 | tokens | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / multi.pass_rate | 14.5300% | fraction | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / multi.pass_rate | 14.5300% | fraction | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / multi.suffix_dup | 7.6900% | fraction | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / multi.suffix_dup | 7.6900% | fraction | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / multi.terminated | 49.5700% | fraction | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / multi.terminated | 49.5700% | fraction | 117 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / seconds | 148.000000 | seconds | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / seconds | 149.000000 | seconds | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / single.exact | 14.6300% | fraction | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / single.exact | 14.6300% | fraction | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / single.mean_excess_tokens | 95.400000 | tokens | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / single.mean_excess_tokens | 95.400000 | tokens | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / single.mean_overgen_chars | 271.600000 | characters | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / single.mean_overgen_chars | 271.600000 | characters | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / single.p95_excess_tokens | 188.000000 | tokens | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / single.p95_excess_tokens | 188.000000 | tokens | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / single.pass_rate | 22.5600% | fraction | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / single.pass_rate | 22.5600% | fraction | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / single.suffix_dup | 18.2900% | fraction | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / single.suffix_dup | 18.2900% | fraction | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | fim / single.terminated | 49.3900% | fraction | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | fim / single.terminated | 49.3900% | fraction | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | gsm8k / gsm8k | 8.4200% | fraction | 1319 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | gsm8k / gsm8k | 8.4200% | fraction | 1319 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | gsm8k / seconds | 865.000000 | seconds | 1319 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | gsm8k / seconds | 863.000000 | seconds | 1319 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | heldout / seconds | 52.000000 | seconds | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | heldout / seconds | 51.000000 | seconds | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | heldout / stackv2-edu | 1.003100 | cross_entropy | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | heldout / stackv2-edu | 1.003100 | cross_entropy | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | heldout / swallow-code | 0.560600 | cross_entropy | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | heldout / swallow-code | 0.560600 | cross_entropy | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | heldout / web | 2.708600 | cross_entropy | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | heldout / web | 2.708600 | cross_entropy | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 9419435120 | hellaswag / acc | 30.4300% | fraction | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | hellaswag / acc_norm | 34.5200% | fraction | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | hellaswag / seconds | 218.000000 | seconds | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 36a83f7a9d | humaneval / humaneval | 21.3400% | fraction | 164 | full_recorded_count | 35/164 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | humaneval / humaneval | 21.3400% | fraction | 164 | full_recorded_count | 35/164 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | humaneval / humaneval_plus | 18.9000% | fraction | 164 | full_recorded_count | 31/164 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | humaneval / humaneval_plus | 18.9000% | fraction | 164 | full_recorded_count | 31/164 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | humaneval / seconds | 165.000000 | seconds | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | humaneval / seconds | 165.000000 | seconds | 164 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / lcb_added_in_v1 | 0.2500% | fraction | 400 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / lcb_added_in_v1 | 0.2500% | fraction | 400 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / lcb_added_in_v3 | 0.9900% | fraction | 101 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / lcb_added_in_v3 | 0.9900% | fraction | 101 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / lcb_added_in_v5 | 0.6000% | fraction | 167 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / lcb_added_in_v5 | 0.6000% | fraction | 167 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / lcb_release_v6_cumulative | 0.2800% | fraction | 1055 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / lcb_release_v6_cumulative | 0.2800% | fraction | 1055 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | lcb / seconds | 3,639.000000 | seconds | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | lcb / seconds | 3,630.000000 | seconds | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | mbpp / mbpp | 29.8900% | fraction | 378 | full_recorded_count | 113/378 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | mbpp / mbpp | 29.8900% | fraction | 378 | full_recorded_count | 113/378 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | mbpp / mbpp_plus | 23.0200% | fraction | 378 | full_recorded_count | 87/378 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | mbpp / mbpp_plus | 23.0200% | fraction | 378 | full_recorded_count | 87/378 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 36a83f7a9d | mbpp / seconds | 443.000000 | seconds | 378 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004999-1673758.json) |
| 64cfa7191b | mbpp / seconds | 442.000000 | seconds | 378 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00004999.json) |
| 9419435120 | mmlu_cf / acc | 27.9700% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | mmlu_cf / acc_norm | 29.7700% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | mmlu_cf / seconds | 273.000000 | seconds | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | mmlu_mc5 / acc | 27.0800% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | mmlu_mc5 / acc_norm | 27.0800% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | mmlu_mc5 / seconds | 2,383.000000 | seconds | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | openbookqa / acc | 22.0000% | fraction | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | openbookqa / acc_norm | 32.4000% | fraction | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | openbookqa / seconds | 5.000000 | seconds | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | sciq / acc | 84.0000% | fraction | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | sciq / acc_norm | 78.6000% | fraction | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | sciq / seconds | 31.000000 | seconds | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | winogrande / acc | 49.8000% | fraction | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | winogrande / acc_norm | 49.8000% | fraction | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |
| 9419435120 | winogrande / seconds | 6.000000 | seconds | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4999.json) |

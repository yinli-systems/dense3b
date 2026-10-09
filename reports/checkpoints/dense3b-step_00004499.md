# dense3b — checkpoint 4499

[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)

**Verified metadata tokens:** — B. Missing metadata is not reconstructed from an assumed batch size.

**Missing full-count task families:** none. Counts alone do not certify evaluation protocol validity.

## Retained file locations

File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.

No current file location found; historical metadata/results are preserved.

## Nearest recorded training observation

Job 1667439, **actual logged step 4499** (not relabeled as checkpoint step): loss 1.077359, MFU 64.51%, global throughput 39,585 tok/s, peak allocated memory 22.387 GB. [Raw log/metrics](../../evidence/raw/results/train-1667439/metrics.jsonl).

## All recorded evaluation values

Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.

| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |
|---|---|---:|---|---:|---|---|---|---|
| 7b1c28bffa | arc_challenge / acc | 27.0500% | fraction | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | arc_challenge / acc_norm | 29.5200% | fraction | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | arc_challenge / seconds | 15.000000 | seconds | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | arc_easy / acc | 59.7600% | fraction | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | arc_easy / acc_norm | 52.6900% | fraction | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | arc_easy / seconds | 25.000000 | seconds | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | boolq / acc | 52.4800% | fraction | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | boolq / acc_norm | 61.9900% | fraction | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | boolq / seconds | 60.000000 | seconds | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| f72a3017e8 | cruxeval / cruxeval_i | 9.5000% | fraction | 800 | full_recorded_count | 76/800 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | cruxeval / cruxeval_o | 20.2500% | fraction | 800 | full_recorded_count | 162/800 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | cruxeval / seconds | 214.000000 | seconds | 800 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / multi.exact | 0.0000% | fraction | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / multi.mean_excess_tokens | 96.200000 | tokens | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / multi.mean_overgen_chars | 269.700000 | characters | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / multi.p95_excess_tokens | 179.000000 | tokens | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / multi.pass_rate | 8.5500% | fraction | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / multi.suffix_dup | 17.0900% | fraction | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / multi.terminated | 45.3000% | fraction | 117 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / seconds | 143.000000 | seconds | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / single.exact | 12.8000% | fraction | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / single.mean_excess_tokens | 101.900000 | tokens | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / single.mean_overgen_chars | 297.100000 | characters | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / single.p95_excess_tokens | 188.000000 | tokens | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / single.pass_rate | 19.5100% | fraction | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / single.suffix_dup | 17.0700% | fraction | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | fim / single.terminated | 48.1700% | fraction | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | gsm8k / gsm8k | 6.4400% | fraction | 1319 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | gsm8k / seconds | 882.000000 | seconds | 1319 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | heldout / seconds | 56.000000 | seconds | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | heldout / stackv2-edu | 1.016100 | cross_entropy | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | heldout / swallow-code | 0.570000 | cross_entropy | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | heldout / web | 2.751100 | cross_entropy | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| 7b1c28bffa | hellaswag / acc | 30.0300% | fraction | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | hellaswag / acc_norm | 33.6800% | fraction | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | hellaswag / seconds | 215.000000 | seconds | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| f72a3017e8 | humaneval / humaneval | 19.5100% | fraction | 164 | full_recorded_count | 32/164 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | humaneval / humaneval_plus | 18.2900% | fraction | 164 | full_recorded_count | 30/164 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | humaneval / seconds | 214.000000 | seconds | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / lcb_added_in_v1 | 0.5000% | fraction | 400 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / lcb_added_in_v3 | 0.0000% | fraction | 101 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / lcb_added_in_v5 | 0.6000% | fraction | 167 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / lcb_release_v6_cumulative | 0.2800% | fraction | 1055 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | lcb / seconds | 3,468.000000 | seconds | — | not_recorded | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | mbpp / mbpp | 19.8400% | fraction | 378 | full_recorded_count | 75/378 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | mbpp / mbpp_plus | 15.8700% | fraction | 378 | full_recorded_count | 60/378 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| f72a3017e8 | mbpp / seconds | 469.000000 | seconds | 378 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00004499-1673608.json) |
| 7b1c28bffa | mmlu_cf / acc | 27.4500% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | mmlu_cf / acc_norm | 28.9500% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | mmlu_cf / seconds | 269.000000 | seconds | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | mmlu_mc5 / acc | 24.3300% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | mmlu_mc5 / acc_norm | 24.3300% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | mmlu_mc5 / seconds | 2,342.000000 | seconds | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | openbookqa / acc | 19.2000% | fraction | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | openbookqa / acc_norm | 31.0000% | fraction | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | openbookqa / seconds | 5.000000 | seconds | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | sciq / acc | 82.1000% | fraction | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | sciq / acc_norm | 75.9000% | fraction | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | sciq / seconds | 31.000000 | seconds | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | winogrande / acc | 51.4600% | fraction | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | winogrande / acc_norm | 51.4600% | fraction | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |
| 7b1c28bffa | winogrande / seconds | 6.000000 | seconds | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step4499.json) |

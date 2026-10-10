# dense3b — checkpoint 1499

[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)

**Verified metadata tokens:** — B. Missing metadata is not reconstructed from an assumed batch size.

**Missing full-count task families:** none. Counts alone do not certify evaluation protocol validity.

## Retained file locations

File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.

`/data/run01/scxi253/code3b-train/milestones/step_00001499` — training_directory; 6492811343 bytes; captured 2026-10-09T23:55:40.432744+00:00.

## Nearest recorded training observation

Job 1667439, **actual logged step 1499** (not relabeled as checkpoint step): loss 1.403862, MFU 63.04%, global throughput 38,684 tok/s, peak allocated memory 22.387 GB. [Raw log/metrics](../../evidence/raw/results/train-1667439/metrics.jsonl).

## All recorded evaluation values

Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.

| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |
|---|---|---:|---|---:|---|---|---|---|
| eefbd27cfb | arc_challenge / acc | 21.6700% | fraction | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | arc_challenge / acc_norm | 25.9400% | fraction | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | arc_challenge / seconds | 14.000000 | seconds | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | arc_easy / acc | 50.6300% | fraction | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | arc_easy / acc_norm | 45.9600% | fraction | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | arc_easy / seconds | 24.000000 | seconds | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | boolq / acc | 60.2400% | fraction | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | boolq / acc_norm | 62.1700% | fraction | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | boolq / seconds | 57.000000 | seconds | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| 1ff767d0fe | cruxeval / cruxeval_i | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | cruxeval / cruxeval_i | 2.5000% | fraction | 800 | full_recorded_count | 20/800 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 037d479dd4 | cruxeval / cruxeval_i | 2.5000% | fraction | 800 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | cruxeval / cruxeval_o | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | cruxeval / cruxeval_o | 12.0000% | fraction | 800 | full_recorded_count | 96/800 | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 037d479dd4 | cruxeval / cruxeval_o | 0.0000% | fraction | 800 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | cruxeval / seconds | 2.000000 | seconds | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | cruxeval / seconds | 184.000000 | seconds | 800 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 037d479dd4 | cruxeval / seconds | 179.000000 | seconds | 800 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / multi.exact | 0.0000% | fraction | 4 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / multi.exact | 0.0000% | fraction | 117 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / multi.mean_excess_tokens | 176.200000 | tokens | 4 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / multi.mean_excess_tokens | 153.900000 | tokens | 117 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / multi.mean_overgen_chars | 588.500000 | characters | 4 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / multi.mean_overgen_chars | 464.600000 | characters | 117 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / multi.p95_excess_tokens | 180.000000 | tokens | 4 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / multi.p95_excess_tokens | 183.000000 | tokens | 117 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / multi.pass_rate | 0.0000% | fraction | 4 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / multi.pass_rate | 1.7100% | fraction | 117 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / multi.suffix_dup | 0.0000% | fraction | 4 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / multi.suffix_dup | 22.2200% | fraction | 117 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / multi.terminated | 0.0000% | fraction | 4 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / multi.terminated | 9.4000% | fraction | 117 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / seconds | 8.000000 | seconds | — | not_recorded | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / seconds | 143.000000 | seconds | — | not_recorded | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / single.exact | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / single.exact | 0.0000% | fraction | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / single.mean_excess_tokens | 167.500000 | tokens | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / single.mean_excess_tokens | 171.000000 | tokens | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / single.mean_overgen_chars | 578.700000 | characters | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / single.mean_overgen_chars | 528.100000 | characters | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / single.p95_excess_tokens | 188.000000 | tokens | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / single.p95_excess_tokens | 188.000000 | tokens | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / single.pass_rate | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / single.pass_rate | 4.8800% | fraction | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / single.suffix_dup | 16.6700% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / single.suffix_dup | 28.0500% | fraction | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | fim / single.terminated | 16.6700% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | fim / single.terminated | 7.9300% | fraction | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | gsm8k / gsm8k | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | gsm8k / gsm8k | 2.5800% | fraction | 1319 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 6d48e5da15 | gsm8k / gsm8k | 2.5000% | fraction | 200 | subset | summary_count_consistent_not_sample_verified | unversioned-legacy | [source](../../evidence/raw/results/eval-step_00001499.json) |
| 1ff767d0fe | gsm8k / seconds | 12.000000 | seconds | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | gsm8k / seconds | 985.000000 | seconds | 1319 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | heldout / seconds | 64.000000 | seconds | — | not_recorded | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | heldout / seconds | 47.000000 | seconds | — | not_recorded | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | heldout / stackv2-edu | 1.266000 | cross_entropy | — | not_recorded | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | heldout / stackv2-edu | 1.266000 | cross_entropy | — | not_recorded | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 6d48e5da15 | heldout / stackv2-edu | 1.266000 | cross_entropy | — | not_recorded | summary_only | unversioned-legacy | [source](../../evidence/raw/results/eval-step_00001499.json) |
| 1ff767d0fe | heldout / swallow-code | 0.732900 | cross_entropy | — | not_recorded | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | heldout / swallow-code | 0.732900 | cross_entropy | — | not_recorded | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 6d48e5da15 | heldout / swallow-code | 0.732900 | cross_entropy | — | not_recorded | summary_only | unversioned-legacy | [source](../../evidence/raw/results/eval-step_00001499.json) |
| 1ff767d0fe | heldout / web | 3.244600 | cross_entropy | — | not_recorded | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | heldout / web | 3.244600 | cross_entropy | — | not_recorded | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 6d48e5da15 | heldout / web | 3.244600 | cross_entropy | — | not_recorded | summary_only | unversioned-legacy | [source](../../evidence/raw/results/eval-step_00001499.json) |
| eefbd27cfb | hellaswag / acc | 27.5500% | fraction | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | hellaswag / acc_norm | 29.0000% | fraction | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | hellaswag / seconds | 202.000000 | seconds | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| 1ff767d0fe | humaneval / humaneval | 0.0000% | fraction | 6 | smoke_test | 0/6 | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | humaneval / humaneval | 6.7100% | fraction | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 6d48e5da15 | humaneval / humaneval | 7.3200% | fraction | 164 | full_recorded_count | summary_count_consistent_not_sample_verified | unversioned-legacy | [source](../../evidence/raw/results/eval-step_00001499.json) |
| 1ff767d0fe | humaneval / humaneval_plus | 0.0000% | fraction | 6 | smoke_test | 0/6 | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | humaneval / humaneval_plus | 6.1000% | fraction | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | humaneval / seconds | 7.000000 | seconds | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | humaneval / seconds | 178.000000 | seconds | 164 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | lcb / lcb_added_in_v1 | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / lcb_added_in_v1 | 0.0000% | fraction | 400 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | lcb / lcb_added_in_v2 | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / lcb_added_in_v2 | 0.0000% | fraction | 111 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | lcb / lcb_added_in_v3 | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / lcb_added_in_v3 | 0.0000% | fraction | 101 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | lcb / lcb_added_in_v4 | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / lcb_added_in_v4 | 0.0000% | fraction | 101 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | lcb / lcb_added_in_v5 | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / lcb_added_in_v5 | 0.0000% | fraction | 167 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | lcb / lcb_added_in_v6 | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / lcb_added_in_v6 | 0.0000% | fraction | 175 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | lcb / lcb_release_v6_cumulative | 0.0000% | fraction | 36 | subset | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / lcb_release_v6_cumulative | 0.0000% | fraction | 1055 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | lcb / lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / lcb_v6_slice_175_post_cutoff | 0.0000% | fraction | 175 | full_recorded_count | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | lcb / seconds | 130.000000 | seconds | — | not_recorded | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 2064e48b7b | lcb / seconds | 3,222.000000 | seconds | — | not_recorded | summary_only | b5f55b7a1481a80f | [source](../../evidence/raw/results/suite-step_00001499-lcb-crux.json) |
| 1ff767d0fe | mbpp / mbpp | 16.6700% | fraction | 6 | smoke_test | 1/6 | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | mbpp / mbpp | 5.2900% | fraction | 378 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | mbpp / mbpp_plus | 16.6700% | fraction | 6 | smoke_test | 1/6 | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | mbpp / mbpp_plus | 4.5000% | fraction | 378 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| 1ff767d0fe | mbpp / seconds | 19.000000 | seconds | 6 | smoke_test | summary_only | 0a91bce8b95553c6 | [source](../../evidence/raw/results/suite-step_00001499-1670852.json) |
| 037d479dd4 | mbpp / seconds | 478.000000 | seconds | 378 | full_recorded_count | summary_only | f741e9571054892a | [source](../../evidence/raw/results/suite-step_00001499.json) |
| eefbd27cfb | mmlu_cf / acc | 25.2500% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | mmlu_cf / acc_norm | 26.7500% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | mmlu_cf / seconds | 255.000000 | seconds | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | mmlu_mc5 / acc | 25.2200% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | mmlu_mc5 / acc_norm | 25.2200% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | mmlu_mc5 / seconds | 2,232.000000 | seconds | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | openbookqa / acc | 17.2000% | fraction | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | openbookqa / acc_norm | 25.6000% | fraction | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | openbookqa / seconds | 4.000000 | seconds | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | sciq / acc | 75.5000% | fraction | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | sciq / acc_norm | 67.9000% | fraction | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | sciq / seconds | 29.000000 | seconds | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | winogrande / acc | 52.1700% | fraction | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | winogrande / acc_norm | 52.1700% | fraction | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |
| eefbd27cfb | winogrande / seconds | 5.000000 | seconds | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step1499.json) |

# dense3b — checkpoint 5499

[All checkpoints](../RESULTS.md) · [System metrics](../SYSTEMS.md)

**Verified metadata tokens:** — B. Missing metadata is not reconstructed from an assumed batch size.

**Missing full-count task families:** none. Counts alone do not certify evaluation protocol validity.

## Retained file locations

File size and markers are filesystem evidence, not a new weight-loading or full-tensor checksum test. Hardlinks retain model bytes outside the trainer’s rolling checkpoint directory.

No current file location found; historical metadata/results are preserved.

## Nearest recorded training observation

Job 1667439, **actual logged step 5499** (not relabeled as checkpoint step): loss 1.100436, MFU 62.56%, global throughput 38,390 tok/s, peak allocated memory 22.387 GB. [Raw log/metrics](../../evidence/raw/results/train-1667439/metrics.jsonl).

## All recorded evaluation values

Each row retains its evaluation/version ID, protocol, sample count and native unit. Repeated historical versions are evidence, not independent new runs.

| Evaluation | Task / metric | Value | Unit | n | Scope | Recount | Protocol | Source |
|---|---|---:|---|---:|---|---|---|---|
| 40f8ea54be | arc_challenge / acc | 28.1600% | fraction | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | arc_challenge / acc_norm | 31.4000% | fraction | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | arc_challenge / seconds | 15.000000 | seconds | 1172 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | arc_easy / acc | 60.4400% | fraction | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | arc_easy / acc_norm | 54.5500% | fraction | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | arc_easy / seconds | 25.000000 | seconds | 2376 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | boolq / acc | 62.1700% | fraction | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | boolq / acc_norm | 62.2000% | fraction | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | boolq / seconds | 59.000000 | seconds | 3270 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 8c0da3676e | cruxeval / cruxeval_i | 19.7500% | fraction | 800 | full_recorded_count | 158/800 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 8c0da3676e | cruxeval / cruxeval_o | 20.7500% | fraction | 800 | full_recorded_count | 166/800 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 8c0da3676e | cruxeval / seconds | 261.000000 | seconds | 800 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| c6984406fc | domain_ce / arxiv | 2.236047 | cross_entropy | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | domain_ce / code | 0.757107 | cross_entropy | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | domain_ce / math | 1.679628 | cross_entropy | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | domain_ce / stackexchange | 2.529832 | cross_entropy | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | domain_ce / web | 2.818370 | cross_entropy | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | domain_ce / wiki | 3.563524 | cross_entropy | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| 8c0da3676e | gsm8k / gsm8k | 7.5800% | fraction | 1319 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 8c0da3676e | gsm8k / seconds | 774.000000 | seconds | 1319 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 40f8ea54be | hellaswag / acc | 30.3900% | fraction | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | hellaswag / acc_norm | 34.5200% | fraction | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | hellaswag / seconds | 213.000000 | seconds | 10042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 8c0da3676e | humaneval / humaneval | 21.9500% | fraction | 164 | full_recorded_count | 36/164 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 8c0da3676e | humaneval / humaneval_plus | 20.1200% | fraction | 164 | full_recorded_count | 33/164 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 8c0da3676e | humaneval / seconds | 205.000000 | seconds | 164 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 8c0da3676e | mbpp / mbpp | 26.4600% | fraction | 378 | full_recorded_count | 100/378 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 8c0da3676e | mbpp / mbpp_plus | 21.9600% | fraction | 378 | full_recorded_count | 83/378 | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 8c0da3676e | mbpp / seconds | 515.000000 | seconds | 378 | full_recorded_count | summary_only | 0d17fac1d5f2efd5 | [source](../../evidence/imported/moe1b/suite-step_00005499-1675015.json) |
| 40f8ea54be | mmlu_cf / acc | 27.9000% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | mmlu_cf / acc_norm | 29.0900% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | mmlu_cf / seconds | 267.000000 | seconds | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | mmlu_mc5 / acc | 25.1800% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | mmlu_mc5 / acc_norm | 25.1800% | fraction | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | mmlu_mc5 / seconds | 2,322.000000 | seconds | 14042 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | openbookqa / acc | 23.0000% | fraction | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | openbookqa / acc_norm | 32.8000% | fraction | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | openbookqa / seconds | 5.000000 | seconds | 500 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| c6984406fc | routing_layers / 0.cluster_domain_nmi | 0.112062 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 1.cluster_domain_nmi | 0.080774 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 10.cluster_domain_nmi | 0.105596 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 11.cluster_domain_nmi | 0.108798 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 12.cluster_domain_nmi | 0.112546 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 13.cluster_domain_nmi | 0.115116 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 14.cluster_domain_nmi | 0.126964 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 15.cluster_domain_nmi | 0.134164 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 16.cluster_domain_nmi | 0.145219 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 17.cluster_domain_nmi | 0.161365 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 18.cluster_domain_nmi | 0.179999 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 19.cluster_domain_nmi | 0.198712 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 2.cluster_domain_nmi | 0.093248 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 20.cluster_domain_nmi | 0.212625 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 21.cluster_domain_nmi | 0.219696 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 22.cluster_domain_nmi | 0.224980 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 23.cluster_domain_nmi | 0.232674 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 24.cluster_domain_nmi | 0.239008 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 25.cluster_domain_nmi | 0.248415 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 26.cluster_domain_nmi | 0.248657 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 27.cluster_domain_nmi | 0.240293 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 3.cluster_domain_nmi | 0.111558 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 4.cluster_domain_nmi | 0.101517 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 5.cluster_domain_nmi | 0.099617 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 6.cluster_domain_nmi | 0.099518 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 7.cluster_domain_nmi | 0.104959 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 8.cluster_domain_nmi | 0.108218 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| c6984406fc | routing_layers / 9.cluster_domain_nmi | 0.111371 | raw_statistic | — | not_recorded | summary_only | unversioned-domain_probe | [source](../../evidence/checkpoint_audit/c6984406fc92ac78/dense3b-5499.json) |
| 40f8ea54be | sciq / acc | 85.0000% | fraction | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | sciq / acc_norm | 79.3000% | fraction | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | sciq / seconds | 31.000000 | seconds | 1000 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | winogrande / acc | 52.0900% | fraction | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | winogrande / acc_norm | 52.0900% | fraction | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |
| 40f8ea54be | winogrande / seconds | 6.000000 | seconds | 1267 | full_recorded_count | summary_only | unversioned-mc | [source](../../evidence/imported/moe1b/mc-dense3b-step5499.json) |

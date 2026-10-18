# Evaluation evidence — recovery status

[Recovery and rerun guide](../../EVALUATION_EVIDENCE.md) · [Six-stage recovery plan](../../configs/evaluation_recovery_plan.json) · [Recorded results](../CONFIRMED_RESULTS.md) · [Raw source index](../sample_verification.json)

## Completed artifact audit

An actual CPU recount of the archived step-6499 files completed in [GitHub Actions run 39139945831](https://github.com/yinli-systems/dense3b/actions/runs/39139945831) on 18 October 2026. This job ran the repository's 116 publication/evidence tests and then independently reopened the original summary and per-item JSON/gzip files.

| Check | Observed result |
|---|---:|
| Archived checkpoint | step 6499 (~6.82B reported milestone) |
| Task families rechecked | 13 |
| Per-item records rechecked across those task/protocol groups | 50,372 |
| Model fingerprints across all 13 task families | 1 |
| Summary hashes, detail hashes, counts and outcome recounts | Passed |
| New model generations or GPU evaluations in this audit | 0 |

The 50,372 figure counts stored task records, not independent unique questions; MMLU-CF and MMLU-MC5 are separate evaluations of overlapping questions.

Recorded checkpoint fingerprint:

```text
e28690ba603117024cedaba1db5559de8955fbe151c96ae1a7ea7f75742b82a6
```

This fingerprint was checked for agreement between existing manifests and per-item files. The GitHub job did not load or rehash a model weight file, because weights are not stored in Git.

## Primary benchmark recounts from actual early records

| Metric | Correct / total |
|---|---:|
| HumanEval | 42 / 164 |
| HumanEval+ | 41 / 164 |
| MBPP | 126 / 378 |
| MBPP+ | 102 / 378 |
| CRUXEval-I | 107 / 800 |
| CRUXEval-O | 162 / 800 |
| GSM8K, relaxed extraction | 152 / 1,319 |
| GSM8K, strict extraction | 150 / 1,319 |
| ARC-Challenge, accuracy | 336 / 1,172 |
| ARC-Challenge, byte-normalized | 364 / 1,172 |
| MMLU-MC5 | 3,543 / 14,042 |
| MMLU-CF, accuracy | 4,012 / 14,042 |

[Download the audit artifact](https://github.com/yinli-systems/dense3b/actions/runs/39139945831/artifacts/12141132619): `archive-audit.json` contains exact source paths, byte hashes, recorded protocol/model identities and recomputed counts; `audit-console.log` and `repository-inventory.json` accompany it. This CI artifact has 30-day retention; the underlying original evidence remains versioned in the repository, and the audit is reproducible using the script.

## Remaining milestone evidence

| Milestone | Raw evidence status | What is prepared |
|---|---|---|
| 6.82B / step 6499 | Archived files reopened and recounted | Hash-linked index and repeatable audit |
| 20B | Original run files not attached | Checkpoint binding entry, reference protocol and package checks |
| 40B | Original run files not attached | Checkpoint binding entry, reference protocol and package checks |
| 60B | Original run files not attached | Checkpoint binding entry, reference protocol and package checks |
| 80B | Original run files not attached | Checkpoint binding entry, reference protocol and package checks |
| 100B | Original run files not attached | Checkpoint binding entry, reference protocol and package checks |

The later published scores and author-confirmed chronology are unchanged. The recovery computer was offline during this work; ParaCloud storage was not inspected in this session. No later-stage raw log is claimed to have been recovered or generated. A future rerun must use the actual corresponding weights, record its actual execution date and preserve its actual result, regardless of whether it matches the previously reported score.

The next operational dependency is access to the existing `code3b-train` checkpoint/evaluation directories or provision of the corresponding original files. The recovery guide lists the commands and evidence required to complete that handoff.

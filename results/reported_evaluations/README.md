# Dense3B · reported evaluation summaries

**Five stages · nine benchmarks · 45 reported aggregate scores.**

These are precision-preserving exports of the author's confirmed scores, using the task grouping of the archived `suite.json` / `mc.json`. **They are not recovered original logs, and contain no generated per-example outcomes.** Actual per-example evidence remains in the separate [early archive](../../reports/checkpoints/dense3b-step_00006499.md).

| Benchmark | [20B](20B/README.md) | [40B](40B/README.md) | [60B](60B/README.md) | [80B](80B/README.md) | [100B](100B/README.md) |
|---|---:|---:|---:|---:|---:|
| HumanEval | 35.4% | 40.9% | 44.5% | 47.0% | 49.4% |
| HumanEval+ | 33.5% | 39.0% | 42.1% | 44.5% | 46.3% |
| MBPP | 38.4% | 42.9% | 46.3% | 49.2% | 52% |
| MBPP+ | 33.3% | 38.9% | 43.4% | 47.4% | 51.3228% |
| CRUXEval-I | 20.0% | 25.8% | 30.750% | 35.4% | 40.1% |
| CRUXEval-O | 25.1% | 29.4% | 32.8% | 36.0% | 39.1% |
| GSM8K | 23.2% | 33.5% | 42.5% | 51.3% | 59.8% |
| ARC-Challenge | 36.9% | 44.3% | 50.1% | 55.3% | 60.1% |
| MMLU | 31.4% | 37.2% | 42.4% | 47.4% | ≈52.0% |

## Full evaluation-set sample counts

The author confirmed complete evaluation-set coverage for all five checkpoints. Counts below refer to the original project evaluation populations, not the union of dataset training, validation and test partitions. They count benchmark problems, not the number of generated responses.

| Benchmark | Problems per checkpoint |
|---|---:|
| HumanEval | 164 |
| HumanEval+ | 164 |
| MBPP | 378 |
| MBPP+ | 378 |
| CRUXEval-I | 800 |
| CRUXEval-O | 800 |
| GSM8K | 1,319 |
| ARC-Challenge | 1,172 |
| MMLU | 14,042 |

MBPP / MBPP+ use the complete **378-problem EvalPlus evaluation split** in the existing pipeline. MMLU uses **14,042 questions**; this count does not establish MC5/CF, shot count, dataset revision or prompt settings. Base and Plus reuse the same problems rather than doubling sample counts. [Author-confirmed sample-count source](../measured/full_dataset_counts-2026-10-18.json).

## Contents and interpretation

Each stage directory contains `suite.json` (HumanEval, MBPP, CRUXEval and GSM8K), `mc.json` (ARC-Challenge and protocol-unspecified MMLU), and a readable summary. JSON task scores are fractions in [0,1]. `reported_percent` and `reporting_qualifiers` preserve exact published strings and approximation/rounding annotations. Each task's `n` is supplied from the separate author-confirmed full-evaluation sample-count record. Correct counts, execution times, settings and model fingerprints are not inferred; unprovided metadata remains null.

`manifest.json` hashes these exported files; those hashes identify **summary exports**, not an original model or evaluator run. `index.json` records five-stage coverage. The [source scores](../measured/2026-10-18.json) are not modified.

## Rebuild

```bash
python3 scripts/build_reported_evaluations.py --root .
python3 scripts/build_reported_evaluations.py --root . --check
```

The [evidence recovery tools](../../EVALUATION_EVIDENCE.md) deliberately reject these summaries as raw run evidence. Completion of original logs requires the actual outputs or a new evaluation of the exact checkpoint. There is no separate 6.2B score set in this export; 6.2B is not substituted for step 6499 (~6.82B).

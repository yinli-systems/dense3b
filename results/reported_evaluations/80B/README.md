# Dense3B · 80B reported evaluation summary

[All stages](../README.md) · [suite.json](suite.json) · [mc.json](mc.json) · [Canonical reported scores](../../measured/2026-10-18.json)

The values below are the model author's confirmed aggregate measurements, reformatted into the same task grouping as the early evaluation summaries. **These exports are not original evaluator output or per-example logs.** No new model evaluation was executed.

| Benchmark | Reported score | Full evaluation problems |
|---|---:|---:|
| HumanEval | 47.0% | 164 |
| HumanEval+ | 44.5% | 164 |
| MBPP | 49.2% | 378 |
| MBPP+ | 47.4% | 378 |
| CRUXEval-I | 35.4% | 800 |
| CRUXEval-O | 36.0% | 800 |
| GSM8K | 51.3% | 1,319 |
| ARC-Challenge | 55.3% | 1,172 |
| MMLU | 47.4% | 14,042 |

`n` now records the author-confirmed count of problems in the full evaluation split used by the original pipeline. [Sample-count source](../../measured/full_dataset_counts-2026-10-18.json). For MBPP this is the project's 378-problem EvalPlus split, not all source-data partitions. Correct counts, checkpoint step/hash, run settings, elapsed time and job identifiers remain unprovided. No count was reverse-engineered from an accuracy. MMLU remains protocol-unspecified rather than being relabelled MC5 or CF.

Percent strings preserve the source precision: final MBPP is integer-rounded and final MMLU is approximate. [Recover or rerun actual evidence](../../../EVALUATION_EVIDENCE.md).

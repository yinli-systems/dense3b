# Dense3B · 60B reported evaluation summary

[All stages](../README.md) · [suite.json](suite.json) · [mc.json](mc.json) · [Canonical reported scores](../../measured/2026-10-18.json)

The values below are the model author's confirmed aggregate measurements, reformatted into the same task grouping as the early evaluation summaries. **These exports are not original evaluator output or per-example logs.** No new model evaluation was executed.

| Benchmark | Reported score | Full evaluation problems |
|---|---:|---:|
| HumanEval | 44.5% | 164 |
| HumanEval+ | 42.1% | 164 |
| MBPP | 46.3% | 378 |
| MBPP+ | 43.4% | 378 |
| CRUXEval-I | 30.750% | 800 |
| CRUXEval-O | 32.8% | 800 |
| GSM8K | 42.5% | 1,319 |
| ARC-Challenge | 50.1% | 1,172 |
| MMLU | 42.4% | 14,042 |

`n` now records the author-confirmed count of problems in the full evaluation split used by the original pipeline. [Sample-count source](../../measured/full_dataset_counts-2026-10-18.json). For MBPP this is the project's 378-problem EvalPlus split, not all source-data partitions. Correct counts, checkpoint step/hash, run settings, elapsed time and job identifiers remain unprovided. No count was reverse-engineered from an accuracy. MMLU remains protocol-unspecified rather than being relabelled MC5 or CF.

Percent strings preserve the source precision: final MBPP is integer-rounded and final MMLU is approximate. [Recover or rerun actual evidence](../../../EVALUATION_EVIDENCE.md).

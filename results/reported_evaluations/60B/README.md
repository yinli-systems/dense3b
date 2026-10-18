# Dense3B · 60B reported evaluation summary

[All stages](../README.md) · [suite.json](suite.json) · [mc.json](mc.json) · [Canonical reported scores](../../measured/2026-10-18.json)

The values below are the model author's confirmed aggregate measurements, reformatted into the same task grouping as the early evaluation summaries. **These exports are not original evaluator output or per-example logs.** No new model evaluation was executed.

| Benchmark | Reported score |
|---|---:|
| HumanEval | 44.5% |
| HumanEval+ | 42.1% |
| MBPP | 46.3% |
| MBPP+ | 43.4% |
| CRUXEval-I | 30.750% |
| CRUXEval-O | 32.8% |
| GSM8K | 42.5% |
| ARC-Challenge | 50.1% |
| MMLU | 42.4% |

`n`, checkpoint step/hash, run settings, elapsed time and job identifiers are null because the corresponding later-run records have not been supplied. No early-run metadata was copied into those fields. MMLU remains protocol-unspecified rather than being relabelled MC5 or CF.

Percent strings preserve the source precision: final MBPP is integer-rounded and final MMLU is approximate. [Recover or rerun actual evidence](../../../EVALUATION_EVIDENCE.md).

# Dense3B · 100B reported evaluation summary

[All stages](../README.md) · [suite.json](suite.json) · [mc.json](mc.json) · [Canonical reported scores](../../measured/2026-10-18.json)

The values below are the model author's confirmed aggregate measurements, reformatted into the same task grouping as the early evaluation summaries. **These exports are not original evaluator output or per-example logs.** No new model evaluation was executed.

| Benchmark | Reported score |
|---|---:|
| HumanEval | 49.4% |
| HumanEval+ | 46.3% |
| MBPP | 52% |
| MBPP+ | 51.3228% |
| CRUXEval-I | 40.1% |
| CRUXEval-O | 39.1% |
| GSM8K | 59.8% |
| ARC-Challenge | 60.1% |
| MMLU | ≈52.0% |

`n`, checkpoint step/hash, run settings, elapsed time and job identifiers are null because the corresponding later-run records have not been supplied. No early-run metadata was copied into those fields. MMLU remains protocol-unspecified rather than being relabelled MC5 or CF.

Percent strings preserve the source precision: final MBPP is integer-rounded and final MMLU is approximate. [Recover or rerun actual evidence](../../../EVALUATION_EVIDENCE.md).

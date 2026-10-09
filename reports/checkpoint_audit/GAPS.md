# Evaluation gaps

Snapshot: 2026-10-09T18:52:28.188993+00:00

This is a missing-evidence list, not a job submission. Availability must be checked again before any rerun: automatic pruning can remove old weights. Full-count task coverage does not certify that the historical harness was correct.

| Variant | Step | Model file present | Missing full-count task families |
|---|---:|---|---|
| dense3b | 2499 | True | arc_easy, arc_challenge, hellaswag, winogrande, boolq, openbookqa, sciq, mmlu_cf, mmlu_mc5 |
| dense3b | 2999 | False | humaneval, mbpp, cruxeval, gsm8k, arc_easy, arc_challenge, hellaswag, winogrande, boolq, openbookqa, sciq, mmlu_cf, mmlu_mc5 |
| dense3b | 3249 | False | humaneval, mbpp, cruxeval, gsm8k, arc_easy, arc_challenge, hellaswag, winogrande, boolq, openbookqa, sciq, mmlu_cf, mmlu_mc5 |
| dense3b | 3499 | False | humaneval, mbpp, cruxeval, gsm8k, arc_easy, arc_challenge, hellaswag, winogrande, boolq, openbookqa, sciq, mmlu_cf, mmlu_mc5 |
| dense3b | 5249 | False | humaneval, mbpp, cruxeval, gsm8k, arc_easy, arc_challenge, hellaswag, winogrande, boolq, openbookqa, sciq, mmlu_cf, mmlu_mc5 |
| dense3b | 5999 | True | humaneval, mbpp, cruxeval, gsm8k, arc_easy, arc_challenge, hellaswag, winogrande, boolq, openbookqa, sciq, mmlu_cf, mmlu_mc5 |

[Machine-readable gaps](evaluation_gaps.json) · [All checkpoints](CHECKPOINTS.md)

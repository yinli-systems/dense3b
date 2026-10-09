# Scheduled checkpoint coverage — unconfirmed slots stay unconfirmed

This is the union of periodic save intervals observed in retained checkpoint metadata, bounded by recorded production progress. It is not an exact schedule-change chronology. A candidate slot does not prove that a save succeeded, or even that the same interval applied at that early step. These candidates are not added to the count of located checkpoint entries.

Validation loss can exist at a slot even when checkpoint metadata and weights were not retained. Such loss values remain training/evaluation observations, not proof of a surviving checkpoint file.

| Variant | Candidate step | Save intervals observed | Other checkpoint evidence | Exact-step validation loss |
|---|---:|---|---|---:|
| dense3b | 249 | [250] | False | — |
| dense3b | 499 | [250] | False | 2.046276 |
| dense3b | 749 | [250] | False | — |
| dense3b | 999 | [250] | False | 1.113300 |
| dense3b | 1249 | [250] | False | — |
| dense3b | 1499 | [250] | True | 0.996152 |
| dense3b | 1749 | [250] | False | — |
| dense3b | 1999 | [250] | False | 0.939678 |
| dense3b | 2249 | [250] | False | — |
| dense3b | 2499 | [250] | True | 0.889248 |
| dense3b | 2749 | [250] | False | — |
| dense3b | 2999 | [250] | True | 0.857847 |
| dense3b | 3249 | [250] | True | — |
| dense3b | 3499 | [250] | True | 0.836609 |
| dense3b | 3749 | [250] | False | — |
| dense3b | 3999 | [250] | False | 0.818826 |
| dense3b | 4249 | [250] | False | — |
| dense3b | 4499 | [250] | True | 0.802570 |
| dense3b | 4749 | [250] | False | — |
| dense3b | 4999 | [250] | True | 0.789960 |
| dense3b | 5249 | [250] | True | — |
| dense3b | 5499 | [250] | True | 0.781645 |
| dense3b | 5749 | [250] | True | — |
| dense3b | 5999 | [250] | True | 0.772738 |

[Machine-readable slots](schedule_coverage.csv) · [Located checkpoint entries](CHECKPOINTS.md)

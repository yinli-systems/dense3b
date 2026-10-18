# Dense3B — Training Timeline

[Repository](../README.md) · [Measured benchmark results](CONFIRMED_RESULTS.md) · [Timeline JSON](../results/chronology/2026-10-18.json) · [Checkpoint dates CSV](chronology/checkpoint_dates.csv)

## Training milestones

The completion dates below were confirmed by the model author on **18 October 2026**. Dates are expressed in **Dubai time (UTC+4)** at calendar-day precision.

| Milestone | Training tokens | Completion date (Dubai) |
|---|---:|---|
| Training start | 0B | 2026-09-23 |
| 6.2B | 6.2B | 2026-09-24 |
| 6.82B (step 6499) | 6.815744B | 2026-09-24 |
| 20B | 20B | 2026-09-28 |
| 40B | 40B | 2026-10-03 |
| 60B | 60B | 2026-10-08 |
| 80B | 80B | 2026-10-13 |
| 100B | 100B | 2026-10-18 |

## Checkpoint lineage

The final **100B model continued training from the original step 6499 weights**. The 6.2B and 6.82B entries identify different progress milestones: step 6499 corresponds to approximately **6.815744B tokens**, at **1,048,576 tokens/optimizer step**.

## Training throughput

The reported later-run global mean training throughput is **45,380 tokens/s**. The separately retained historical systems snapshot reports a median of **41,780 tokens/s**. [Systems metrics](SYSTEMS.md) · [Metric sources](measured/SOURCES.md).

## Data and provenance

The training timeline and weight-continuation relationship are author-confirmed project records. The source timeline is maintained as [JSON](../results/chronology/2026-10-18.json) and [CSV](chronology/checkpoint_dates.csv). Original training logs, checkpoint metadata, evaluation reports and archive manifests remain unchanged and accessible through the repository.


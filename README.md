# Dense3B

### 3.238B dense parameters. 100B training tokens. Measured checkpoint results.

An end-to-end dense-model pretraining project with training code, data preparation, checkpoint measurements and archived execution evidence.

**[Project page](https://yinli-systems.github.io/dense3b/) · [Full results](reports/CONFIRMED_RESULTS.md) · [CSV](reports/measured/results.csv) · [JSON](results/measured/2026-10-18.json) · [Model card](MODEL_CARD.md) · [Historical archive](reports/RESULTS.md)**

## 100B-token results

| HumanEval | HumanEval+ | GSM8K | ARC-Challenge | MBPP+ | MMLU* |
|---:|---:|---:|---:|---:|---:|
| **49.4%** | **46.3%** | **59.8%** | **60.1%** | **51.3228%** | **≈52.0%** |

## Measured checkpoint matrix

All values below are author-confirmed measurements. Percentages retain the supplied precision. [Measurement sources and evaluation conventions](reports/measured/SOURCES.md) distinguish archived raw evidence from later author-confirmed reports.

| Benchmark | [6.82B](reports/measured/checkpoints/6p82B.md) | [20B](reports/measured/checkpoints/20B.md) | [40B](reports/measured/checkpoints/40B.md) | [60B](reports/measured/checkpoints/60B.md) | [80B](reports/measured/checkpoints/80B.md) | [100B](reports/measured/checkpoints/100B.md) |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval | 25.61 | 35.4 | 40.9 | 44.5 | 47.0 | 49.4 |
| HumanEval+ | 25.00 | 33.5 | 39.0 | 42.1 | 44.5 | 46.3 |
| MBPP | 33.33 | 38.4 | 42.9 | 46.3 | 49.2 | 52 |
| MBPP+ | 26.98 | 33.3 | 38.9 | 43.4 | 47.4 | 51.3228 |
| CRUXEval-I | 13.38 | 20.0 | 25.8 | 30.750 | 35.4 | 40.1 |
| CRUXEval-O | 20.25 | 25.1 | 29.4 | 32.8 | 36.0 | 39.1 |
| GSM8K | 11.52 | 23.2 | 33.5 | 42.5 | 51.3 | 59.8 |
| ARC-Challenge | 28.67 | 36.9 | 44.3 | 50.1 | 55.3 | 60.1 |
| MMLU* | 25.23 | 31.4 | 37.2 | 42.4 | 47.4 | ≈52.0 |

*Early MMLU uses MC5; later MMLU evaluator manifests are not attached. The final MMLU value is approximate. MBPP at 100B is reported at integer precision (52%). The 6.82B token label is rounded.*

The final author corrections include **HumanEval+ 40B: 39.0%**, **MBPP 20B/40B/60B: 38.4% / 42.9% / 46.3%**, **CRUXEval-I 60B: 30.750%**, and **MBPP+ 100B: 51.3228%**.

![Code generation measured checkpoint scores](docs/assets/code_generation.svg)

[Code reasoning and mathematics/knowledge plots](reports/CONFIRMED_RESULTS.md#capability-plots).

## Theoretical training chronology

The author confirms that the **100B weights continued from step 6499**. Under an **idealized** continuous-throughput scenario (45,380 tokens/s and 100% uptime), using the 18 October report time as the reference, the back-calculated dates are:

| Milestone | Training tokens | Theoretical Dubai date (UTC+4) |
|---|---:|---|
| Training start | 0B | 2026-09-23 |
| 6.2B | 6.2B | 2026-09-24 |
| 6.82B (step 6499) | 6.815744B | 2026-09-24 |
| 20B | 20B | 2026-09-28 |
| 40B | 40B | 2026-10-03 |
| 60B | 60B | 2026-10-08 |
| 80B | 80B | 2026-10-13 |
| 100B | 100B | 2026-10-18 |

These are **not audited checkpoint creation dates**. In particular, the hypothetical 24 September step 6499 date conflicts with the recorded 7–10 October training history if a single uninterrupted token counter is assumed. [Read the chronology, assumptions and source conflicts](reports/CHRONOLOGY.md).

## Training implementation and systems

**Later-run average global throughput (author-reported): 45,380 tokens/s.** The averaging window and supporting logs were not supplied; this is separate from the older snapshot median.

The following measurements belong to the **9 October 2026 archived snapshot**, not the complete 100B run.

| Archived field | Value |
|---|---:|
| Dense parameters, including embeddings | 3,237,529,600 |
| Training GPUs | 8 × NVIDIA GeForce RTX 4090 |
| Model FLOP utilization, median of last ≤100 records | 68.09% |
| Global training throughput, same window | 41,780 tokens/s |
| Peak allocated memory, latest logged observation | 22.387 GB |

MFU is the implementation's model-FLOP estimate, not device utilization. Loop throughput excludes unrecorded stalls and downtime. [All system metrics and training curves](reports/SYSTEMS.md) · [Raw observations](reports/training_all.csv) · [Recorded run metadata](reports/run_index.json).

## Repository map

| Area | Contents |
|---|---|
| [`results/measured/`](results/measured/) | Canonical, precision-preserving author-confirmed measurements |
| [`reports/measured/`](reports/measured/) | CSV exports, source notes and six milestone pages |
| [`docs/`](docs/) | Static project site and three capability plots |
| [`model/`](model/) · [`train/`](train/) | Architecture, optimization, parallelism and training |
| [`code3b/`](code3b/) · [`data_pipeline/`](data_pipeline/) | Filtering, decontamination, deduplication and token packing |
| [`bench/`](bench/) · [`tokenizer/`](tokenizer/) | Evaluation code and tokenizer artifacts |
| [`evidence/`](evidence/) · [`provenance/`](provenance/) | Original source captures, logs, manifests and hash inventories |
| [`reports/RESULTS.md`](reports/RESULTS.md) | All 13 archived checkpoint entries and 607 archived metric rows |

## Rebuild the presentation

The measured-results presentation uses only the Python standard library; it does not execute model evaluation.

```bash
python3 scripts/build_measured_results.py --root .
python3 scripts/build_measured_results.py --root . --check
python3 -m unittest discover -s tests/publication -v
python3 scripts/verify_publication.py
```

To rebuild the original archive and then refresh the current results presentation:

```bash
python3 -m pip install -r requirements-report.txt
python3 scripts/build_all_results.py --root .
```

The archive builder automatically refreshes the current measured-results pages when the canonical release is present. Original raw evidence is not rewritten by the new presentation builder.

## Availability and provenance

The author confirmed all milestone scores as measurements on **18 October 2026**. Early scores have archived evaluation outputs; later per-example logs, evaluator manifests and weight hashes are not attached to this release. No independent rerun of later evaluations is claimed.

Model weights, optimizer states and data shards are not distributed in Git. Updated data-mixing weights and change points are not reported here. The [initial mixture](evidence/manifests/mix-v0/mix.json), [historical homepage](ARCHIVED_README.md), [evaluation protocols](reports/evaluation_versions.json) and [network-redaction notes](docs/PUBLICATION_SECURITY.md) remain available.

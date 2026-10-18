# Dense3B — theoretical training chronology

[Project home](../README.md) · [Measured benchmarks](CONFIRMED_RESULTS.md) · [Chronology JSON](../results/chronology/2026-10-18.json) · [Chronology CSV](chronology/theoretical_stages.csv)

## Weight continuity (author confirmation)

The model author states that the **100B checkpoint weights were continued from the original `step_00006499` weights**. This is an author-confirmed lineage statement, **not** independently verified tensor ancestry. Optimizer-state continuity, token-counter continuity and the final weight hash remain unknown. The 6.2B figure in the timeline is not identical to step 6499 (approximately 6.815744B tokens).

## Theoretical backward timeline (NOT verified run dates)

The mathematical back-calculation takes the first 100B-result report at **2026-10-18 14:54:22 Dubai time** as a hypothetical completion reference, assumes **45,380 tokens/s** global throughput with **100% uptime**, and counts cumulative tokens monotonically. The report timestamp is not independently verified as the actual completion time. If training stopped earlier, these dates shift earlier; if the same token throughput was intermittent, earlier milestones shift earlier.

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

**Formula:** `t(D) = t_report − (100 − D) × 10^9 / 45,380` seconds, where `D` is billions of cumulative tokens. The calculation from step 6499 (~6.815744B) to 100B is about **23.77 continuous-runtime days**, and 0B to 100B is about **25.50 days**.

## Archived chronology conflict (unresolved)

- The original Code3B project README, dated **7 October 2026**, records **step 3715 (~3.9B)**.
- The [step 6499 metadata archived in the public repo](../evidence/sync-20261009T2320Z/training/dense3b/ckpt-v0/step_00006499/meta.json) has file modification time **2026-10-09 22:52:48 UTC**, i.e. **10 October 02:52:48 Dubai**. File modification/upload time need not equal first weight creation.
- The [9 October training archive](checkpoints/dense3b-step_00006499.md) contains a step 6499 training observation, and [GitHub committed that metadata on 9 October UTC](https://github.com/yinli-systems/dense3b/commit/448e4f45357743a77ff68ae13932d76ed3f54444).
- These October observations cannot simultaneously be interpreted as an uninterrupted monotonically counted run that passed step 6499 on **24 September**, at the stated global rate. The author's weight-continuity statement does not establish a continuously preserved token counter or optimizer state.

**Publication policy:** Dates in this document are conditional *theoretical dates*, not recovered or independently validated checkpoint creation times. Existing date-stamped archived evidence and all 54 author-confirmed benchmark scores are preserved unchanged. Reconciliation requires final checkpoint metadata, training restart logs, actual consumed-token counters, and hardware/throughput history.


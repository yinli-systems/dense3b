# Dense3B model card

## Identity

Dense3B is a dense pretraining model with **3,237,529,600 parameters** in the archived configuration. The author reports training milestones through **100 billion tokens**. The public repository contains implementation, measured results and archived evidence; model weights are not included.

## Evaluation

[The complete measured-results report](reports/CONFIRMED_RESULTS.md) contains 54 primary scores across six milestones, covering code generation, code reasoning, mathematics and knowledge. All results are author-confirmed measurements. The early milestone also has archived evaluation outputs. Later protocol manifests, per-example results and checkpoint hashes are not attached.

[Source notes](reports/measured/SOURCES.md) preserve original precision, six author corrections, the approximate final MMLU score, integer-rounded final MBPP score and the separation of MMLU-MC5 from MMLU-CF. Scores across different or unarchived protocols should not be treated as a controlled comparison.

## Training and data

The archived training code and run metadata describe RTX 4090-based training. Original source proportions, repetition factors and data processing are documented in the [initial mixture manifest](evidence/manifests/mix-v0/mix.json) and [`code3b/`](code3b/). The later source mixture, its switch points and full-run hardware utilization have not been provided for this release. The author separately reported a later-run average global throughput of **45,380 tokens/s**. No averaging window or supporting logs were supplied; it must not be compared as if it were the archived **41,780 tokens/s median**.

## Weight lineage and theoretical dates

The author confirms the 100B weights descend from step 6499. The [theoretical timeline](reports/CHRONOLOGY.md) uses the 18 October report as an endpoint and assumes 45,380 tokens/s without downtime. It dates step 6499 to 24 September, conflicting with the 7–10 October archived training chronology if treated as one uninterrupted token count. Actual completion dates and optimizer/token-counter continuity are unverified.

## Scope and limitations

These benchmarks describe the evaluated tasks; they do not establish deployment reliability, absence of benchmark overlap, general instruction-following ability, or performance under an unreported decoding setup. No new training, GPU evaluation or external model comparison is performed by the publication scripts.

## Reproduction and access

Pages and tables can be rebuilt from [`results/measured/2026-10-18.json`](results/measured/2026-10-18.json) with `python3 scripts/build_measured_results.py`. Repeating model evaluation additionally requires weights, datasets and the appropriate protocol; no weight availability is asserted. Source datasets retain their own terms; this model card grants no additional license.

# Audit method and limits

The audit reads `provenance/checkpoint_audit_sources.json` and verifies SHA-256 before parsing any source. Rebuild with `python scripts/audit_checkpoints.py --root .`.

## Search and identity

The scope is the four current publication repositories, their live ParaCloud run/source roots and resolved backup targets, the original code3b-pretraining repository and its three branches, and all four publication histories. It is not a claim to have searched every unrelated older model project or external cloud account. The filesystem inventory is a point-in-time snapshot.

A checkpoint enters the main inventory through saved metadata, an evaluation result, a retained model-file listing, or an explicitly labeled resumed-start inference. Other path references are retained in checkpoint_references.csv and do not prove that a scheduled checkpoint was ever written. Variant, step, architecture metadata and declared checkpoint hashes are retained. Distinct hashes are flagged instead of silently merged.

## Results and arithmetic

Every JSON version is preserved by content hash. Repeated copies are source aliases, not extra runs. The result parser includes legacy summaries, nested FIM metrics, correct LiveCodeBench slice denominators, cross-entropy greater than one, all routing-layer scalars and permutation delta loss. Raw summaries remain untouched. Log lines corroborate existing values; otherwise they are explicitly marked as log-only recovery.

HumanEval, MBPP and CRUXEval Boolean judgments are re-counted when a matching complete details file is present. Counts, duplicate identifiers, Plus/base consistency and agreement with four-decimal summaries are checked. This is an independent arithmetic aggregation of saved flags, not an independent rerun of the program or benchmark harness. Sparse legacy example snippets are not treated as complete sample evidence. Optional Wilson intervals in the numeric table describe binomial sampling uncertainty only, not protocol bias or contamination.

## Coverage and units

Full-count labels use sample counts observed in the archived full-run protocol. A count alone does not prove the right item IDs or prompt implementation were used. Smoke/subset/full-count records remain distinct. Fractions may be displayed as percentages; loss, delta loss, characters, tokens and timing remain in native units. Missing tasks are absent or null, never a fabricated zero. Optional FIM, LiveCodeBench and domain probes are not required to call the primary task-family count complete; all their observed values remain visible.

## Tokens and training traces

A saved samples_consumed count times sequence length is distinguished from a reconstruction using the single constant observed production token-per-step schedule. Reconstructed values remain explicitly not metadata-verified. Counts describe logical stream progress, not total compute including replayed steps after rollback. Original rounded token claims are retained and discrepancies reported. Training observations keep their original job, source line and observed step; no near-by value is silently relabeled as a checkpoint measurement.

## No changes to training

This audit did not launch GPU evaluation, deserialize weights, hash tensor payloads, modify checkpoint retention, restart training, or mutate the original logs. File presence and optimizer shard counts are not a loadability test. Old metadata or scores cannot reconstruct deleted weights. Available paths must be rechecked before a rerun because training and pruning continue.

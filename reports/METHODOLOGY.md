# Report extraction policy

## Training traces
All JSONL observations retain their source path and 1-based line number. The raw table includes diagnostic runs. The main figure excludes explicitly named smoke/profiling/sweep runs, orders production jobs by numeric job ID, and replaces the earlier suffix at the first logged step of a later run. This is a transparent display reconstruction, not independent verification of optimizer-state ancestry. The all-runs figure and raw CSV retain superseded tails. No interpolated training observations are created. Smoothing is a trailing arithmetic mean of at most 50 retained observations. Sequence length, world size and tokens per step are retained from start events in the run index; those fields are reported runtime values, not a physical topology audit.

## Evaluations
Model identity comes from the recorded checkpoint path, not the directory holding the result. Foreign and unidentified summaries are excluded and logged. Exact duplicate summary files are counted once. Task/metric names, sample counts, declared protocols and evaluator hashes are retained; unversioned evaluations remain labeled unversioned. Accuracy and normalized accuracy remain separate. Latest tables choose largest step then newest archived source timestamp, never the highest score. Checkout filesystem timestamps are never used. Points do not establish a controlled trend across protocol changes. No missing benchmark is entered as zero. Original summary scores are rounded source values; sample-level outputs are preserved separately. The complete checkpoint audit re-aggregates saved Boolean judgments without rerunning benchmark execution.

## Token accounting
Only checkpoint metadata with recorded samples_consumed and args.seq_len establishes a verified token count. Formula: samples_consumed * args.seq_len. A checkpoint's declared configuration and sample counter are evidence, not a new audit of the original data loader. The source tokens_B field is also retained. Differences exceeding 0.00501 billion tokens are flagged (allowing source rounding). Missing metadata is not reconstructed from a guessed batch size or context length.

## Operational measurements
Throughput/MFU/memory values are copied from logged training metrics. They do not include every checkpoint stall, evaluation, queue delay or downtime. MFU is the source implementation's estimate; no cross-hardware equivalence is asserted. Missing resource samples are not filled. CUDA correctness, benchmark execution, dataset availability and remote checkpoint completeness were not retested during publication.

## Rebuild and integrity
Run scripts/verify_publication.py before rebuilding. It verifies the archived source/evidence SHA-256 inventory and import receipts. requirements-report.txt captures the reporting environment. Unit tests exercise extraction policies independently of CUDA. All training/evaluation jobs remain outside this reporting workflow.

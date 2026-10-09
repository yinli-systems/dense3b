# Bounded checkpoint evaluation backfill

This directory preserves the actual deployed evaluation implementation, model dependency source snapshot, tokenizers, frozen benchmark inputs, scheduling plan, isolation checks and EvalPlus patch. It is not an automatic promise to update GitHub when cluster jobs finish.

The original training code and existing archival evidence were not changed. Model files are hardlinked on ParaCloud for evaluation only and are not Git payloads. The campaign is frozen at the `final_rescan_utc` in plan.json; new checkpoints created after that time are outside this bounded submission.

## Execution

`dependency-graph.json` records two native Slurm chains. Each job requests one GPU on gpu_5090 at nice=10000. There can be no more than two campaign GPU jobs active simultaneously. Queued training jobs, including the user-held MoE-R job, were not changed. Multiple partitions report some of the same GPUs and are not additive resources. Each task has a 90-minute wall limit. Infrastructure failures remain failures, not zero benchmark scores.

`mc.sbatch` / `run_mc.py` score the missing full-count MC tasks. `generation.sbatch` / `run_generation.py` generate and score HumanEval, MBPP, CRUXEval and GSM8K where missing. Exact saved checkpoint config and tokenizer are used. Generation follows the archived prompts/budgets/stops. MC uses the archived continuation likelihood and UTF-8-byte normalization; custom MMLU modes are not a claim of identical official leaderboard protocols.

`score-runtime/evalplus` is a copy of the installed scoring package with one documented successful-test progress-counter correction. See `evalplus-find-zero-progress.patch` and `evalplus-patch-provenance.json`. HumanEval/32 is actually tested, not granted the Base score. Correct and deliberately incorrect reference controls are recorded in `scoring-preflight-receipt.json`. The original installed package is unchanged.

## Isolation and limits

Untrusted code executes in user/network/PID namespaces with a fail-closed Landlock filesystem allowlist and seccomp syscall allowlist. Runtime code is read-only; writes are confined to a test-owned temporary area. Probes demonstrate that a benign outside sentinel cannot be read and new sockets cannot be created. CPU/memory/output/time limits apply. These probes are not a formal proof of sandbox security. No privileged mounts or host configuration changes are made.

## Results and provenance

Completed task outputs are written atomically, and per-example scores are saved. Protocol and input/code/checkpoint hashes are recorded. Different historical and corrected scoring protocols must remain separate. A task still running or queued has no invented score. The matching `reports/backfill-20261009T2000Z/STATUS.md` is a timestamped evidence snapshot, not a live feed.

These scripts use known cluster paths; inspect and adapt them before running elsewhere. Do not re-run the submission scripts against an existing campaign: duplicate-job guards and the native dependency graph preserve the bounded workload.

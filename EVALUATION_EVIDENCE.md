# Dense3B evaluation evidence

[Measured results](reports/CONFIRMED_RESULTS.md) · [Recovery plan](configs/evaluation_recovery_plan.json) · [Archived per-item index](reports/sample_verification.json) · [Evidence CI](../../actions/workflows/evaluation-evidence.yml)

## What is available, and what this addition does

The early step-6499 evaluation (~6.82B tokens) already has versioned summaries, checkpoint fingerprints and per-item records. `scripts/evaluation_evidence.py audit` reopens these files, checks their SHA-256 values, verifies the model/task identities and recomputes the metrics from the recorded Boolean outcomes. It is an actual CPU audit of existing records, **not a new model evaluation**.

The 20B, 40B, 60B, 80B and 100B scores remain author-confirmed measurements. Their original per-example records and later checkpoint weights have not been attached to this repository. Preparing a schema, plan or directory does not change that status. This addition does not manufacture predictions, problem-level success flags, clock times, job IDs or checkpoint hashes to match an aggregate score.

| Stage | Reported results | Raw per-item artifacts in the published release | Next action |
|---|---|---|---|
| 6.82B / step 6499 | Published | Archived; rechecked by evidence CI | Preserve original files and fingerprint |
| 20B | Published | Not attached | Recover the actual run directory or evaluate its exact surviving weights |
| 40B | Published | Not attached | Recover the actual run directory or evaluate its exact surviving weights |
| 60B | Published | Not attached | Recover the actual run directory or evaluate its exact surviving weights |
| 80B | Published | Not attached | Recover the actual run directory or evaluate its exact surviving weights |
| 100B | Published | Not attached | Recover the actual run directory or rerun the actual final weights |

The 6.2B label is not another name for step 6499. It needs its own checkpoint identity before it can be bound to evidence.

## 1. Recheck and export the existing archive

Run from a checkout of this repository, on CPU:

```bash
python3 scripts/evaluation_evidence.py audit --root . --out /tmp/dense3b-archive-audit.json
python3 -m unittest discover -s tests/publication -v
```

The audit contains the actual file paths, stored and decompressed SHA-256 fingerprints, evaluator protocol, model fingerprint, observed item fields, and per-metric correct/total counts. Missing late-stage records remain empty rather than being populated from reported scores. Output files use exclusive creation: choose a new output filename for each audit.

The **Evaluation evidence integrity** GitHub Actions workflow repeats this audit on the repository files and publishes the JSON receipt plus its console log as the `dense3b-evaluation-evidence-audit` artifact. This artifact is a consistency report about archived data, not GPU evaluation output.

## 2. Recover actual files from the training system

On the authorized ParaCloud host, inventory the precise project roots first:

```bash
python3 scripts/evaluation_evidence.py inventory \
  --roots /data/run01/scxi253/code3b-train \
  --out /tmp/dense3b-training-inventory.json
```

Also inspect the original checkpoint evaluation campaign roots when available. The inventory searches only recognized metadata, weight, metric and per-item filenames. It does not load model weights, execute generated code, read credentials or follow symlinks. It reports inaccessible roots and scan truncation rather than presenting a partial scan as complete. File names and modification times alone are not interpreted as evidence of consumed-token counts.

Copy `configs/evaluation_recovery_plan.json` into a private working directory and fill each stage's `checkpoint_dir`, `checkpoint_sha256` and `original_evaluation_dir` from the actual files. Do not commit credentials, private storage endpoints, SSH configuration or unsanitized environment dumps.

## 3. Required evidence for every checkpoint

A complete release should include:

| Material | Required contents |
|---|---|
| Model identity | Hash of actual weights, exact checkpoint metadata, consumed tokens and available resume lineage |
| Protocol | Evaluator source revision/hash, tokenizer hash, dataset version/hash, shots/prompts, precision, decoding, stops and extraction rules |
| Per-item data | All problem IDs; raw completions or choice scores; parsed outputs; gold/predicted values; test outcomes |
| Summaries | Counts and metrics recomputed from those records, including strict/relaxed or normalized metrics separately |
| Execution | Actual command, start/end timestamps, exit status, GPU setup, package versions, stdout/stderr, failed task records |
| Code safety | Sandbox/isolation preflight receipt and test resource limits |
| Integrity | Byte-preserving file hashes and explicit original-recovery versus new-rerun provenance |

The checked runner format is the corrected archived v2 pipeline: `suite.json` for HumanEval, MBPP, CRUXEval and GSM8K; `mc.json` for ARC and MMLU; and one `TASK.details.json` (or `.json.gz`) per task. Base/Plus and strict/relaxed metrics stay distinct. MMLU-CF is not silently substituted for MMLU-MC5.

## 4. Rerun only when the real files cannot be recovered

The already archived runners are:

- `operations/backfill-20261009T2000Z/run_generation.py`
- `operations/backfill-20261009T2000Z/run_mc.py`

They read a campaign-local `plan.json`, with per-job `repo`, `preset`, `step`, `original_checkpoint`, `pinned_checkpoint`, `metadata_sha256` and `missing_tasks` entries. Use a **new campaign workspace and new output directories**, with each entry bound to the actual checkpoint. Preserve the deployed `source`, `runtime` and `score-runtime` snapshots, and verify dataset/tokenizer hashes before launch. The archived environment includes cluster-specific Python and data paths; validate them on the authorized host rather than assuming the public repository alone is a self-contained GPU installation.

After that preflight, the existing runner interfaces are:

```bash
python3 /path/to/NEW_CAMPAIGN/run_generation.py --index 0
python3 /path/to/NEW_CAMPAIGN/run_mc.py --index 0
```

Repeat only for the verified entries in the six-stage plan. The generation runner already checks isolation before generated Python is scored; never remove the isolation checks or execute model-generated solutions directly on the login host. Do not reuse an old result directory, because the archived runners support resuming cached results.

A rerun records its actual new execution date. It is **not** an original September or October evaluation log. Keep its scores even when they differ from the reported historical table. Differences are investigated through protocol, dataset and model identity, never by modifying per-item outcomes.

If a checkpoint's weights and original predictions are both lost, a true rerun of that particular historical model is not possible from the final weights alone. Leave that stage unresolved; do not substitute 100B weights or a neighboring checkpoint.

## 5. Verify and package a recovered or newly executed run

First compute the fingerprint from the actual weight file using a trusted local checksum tool. For example, on the Linux training host:

```bash
sha256sum /actual/checkpoint/model.pt
```

Then validate the corresponding evaluation directory:

```bash
python3 scripts/evaluation_evidence.py verify \
  --source /actual/evaluation/run-directory \
  --checkpoint-sha256 ACTUAL_64_CHARACTER_SHA256
```

To copy the existing files into a new release bundle without rewriting their contents:

```bash
python3 scripts/evaluation_evidence.py pack \
  --source /actual/evaluation/run-directory \
  --checkpoint-sha256 ACTUAL_64_CHARACTER_SHA256 \
  --stage 100B \
  --origin recovered_original \
  --out /new/private/release-bundle-100B
```

Use `--origin new_rerun` for a new run. The package timestamp is generated at packaging time and is not presented as a checkpoint creation timestamp. The receipt is written last; interrupted packaging must not be represented as a complete release. The validator supports the archived single-greedy-sample, full-count protocol. A different subset, multi-sample estimator or evaluator requires a separately documented adapter, not a forced fit to this schema.

Only allowlisted evidence files are copied. Model weights, arbitrary user files and environment credentials are excluded. Review raw text for secrets and data-distribution restrictions before publishing. This tool does not upload files or silently replace the existing reported-results table.

## Interpretation of validation

Checks reject missing details, incomplete sample counts, duplicate IDs, non-Boolean outcomes, model hash mismatches, altered bytes, nonfinite scores, Plus passes when Base fails, and disagreement between summary metrics and per-item recounts. They establish consistency of supplied files, not independent authentication of the original run or proof that a model actually generated them. Re-evaluation with the actual weights remains a separate operation.

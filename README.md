# dense3b

Dense 3B-class code pretraining: deployed implementation, data pipeline, checkpoint results and auditable training curves.

**[All checkpoints and results](reports/RESULTS.md) · [MFU, loss, throughput and every system metric](reports/SYSTEMS.md) · [Backfill progress](reports/CAMPAIGN.md)**

Training/evaluation capture started **2026-10-09T23:20:38.782167+00:00 / 2026-10-09T23:20:38.827803+00:00**. Jobs continued while files were copied; these are timestamped snapshots, not a live dashboard.

## Training and efficiency

| Variant | Latest logged step | Latest loss | MFU median, last ≤100 | Global tok/s median, last ≤100 | Peak allocated GB, latest |
|---|---:|---:|---:|---:|---:|
| dense3b | 6551 | 1.0693 | 68.02% | 41,737 | 22.387 |

MFU is the logged model-FLOP estimate, not device utilization. Per-step throughput excludes unrecorded downtime. The systems page includes every numeric logged field, full unsmoothed data, per-run provenance and the explicit resume-display policy.

![loss](reports/figures/all/training_loss.png)

![mfu](reports/figures/all/training_mfu.png)

![tok_per_s](reports/figures/all/training_tok_per_s.png)

## Latest completed task results in the new scoring campaign

Latest checkpoint per variant/task/metric, not the highest score. The table may combine tasks at different steps, which are shown explicitly. Historical results and all protocols remain in the complete result index.

| Variant | Step | Task / metric | Score | n | Verified count |
|---|---:|---|---:|---:|---|
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | arc_challenge / acc | 27.9010% | 1172 | 327/1172 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | arc_challenge / acc_norm | 31.2287% | 1172 | 366/1172 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | arc_easy / acc | 63.2155% | 2376 | 1502/2376 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | arc_easy / acc_norm | 58.3754% | 2376 | 1387/2376 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | boolq / acc | 54.2202% | 3270 | 1773/3270 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | boolq / acc_norm | 61.7431% | 3270 | 2019/3270 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | cruxeval / cruxeval_i | 17.3700% | 800 | 139/800 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | cruxeval / cruxeval_o | 17.7500% | 800 | 142/800 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | gsm8k / gsm8k | 10.7657% | 1319 | 142/1319 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | gsm8k / strict_accuracy | 10.4625% | 1319 | 138/1319 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | hellaswag / acc | 30.9102% | 10042 | 3104/10042 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | hellaswag / acc_norm | 35.4511% | 10042 | 3560/10042 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | humaneval / humaneval | 22.5610% | 164 | 37/164 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | humaneval / humaneval_plus | 21.3415% | 164 | 35/164 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | mbpp / mbpp | 34.6561% | 378 | 131/378 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | mbpp / mbpp_plus | 26.9841% | 378 | 102/378 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | mmlu_cf / acc | 28.5287% | 14042 | 4006/14042 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | mmlu_cf / acc_norm | 30.2236% | 14042 | 4244/14042 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | mmlu_mc5 / acc | 24.8042% | 14042 | 3483/14042 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | mmlu_mc5 / acc_norm | 24.8042% | 14042 | 3483/14042 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | openbookqa / acc | 21.2000% | 500 | 106/500 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | openbookqa / acc_norm | 31.6000% | 500 | 158/500 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | sciq / acc | 85.6000% | 1000 | 856/1000 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | sciq / acc_norm | 82.1000% | 1000 | 821/1000 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | winogrande / acc | 50.0395% | 1267 | 634/1267 |
| dense3b | [5999](reports/checkpoints/dense3b-step_00005999.md) | winogrande / acc_norm | 50.0395% | 1267 | 634/1267 |

## Archived architecture and accounting

| Variant | Total parameters | Active parameters |
|---|---:|---:|
| dense3b | 3,237,529,600 | 3,237,529,600 |

These are archived configuration counts, including embeddings; they are not task-quality rankings.

## Reproduction and provenance

```bash
python -m pip install -r requirements-report.txt
python scripts/verify_publication.py
python scripts/build_all_results.py --root .
python -m unittest discover -s tests/publication -v
```

[Current immutable capture manifest](provenance/sync-20261009T2320Z/manifest.json) · [Latest capture pointer](provenance/sync_latest.json) · [Original snapshot inventory](provenance/snapshot.json) · [Earlier checkpoint audit](reports/checkpoint_audit/CHECKPOINTS.md)

Original logs/results and deployed source snapshots are preserved by hash. No original training source was overwritten by this synchronization. Tensor weights, optimizer states, data shards, credentials and installed binary caches are not Git payloads. Live file/pin inventories include exact paths, sizes and capture times. Loss values keep cross-entropy units; unmeasured values stay absent.

New Plus results use the documented corrected EvalPlus backend; historical custom Plus scores are not silently merged into the same protocol. MC accuracy and byte-normalized accuracy remain separate. GSM8K strict phrase extraction and relaxed answer extraction are separate metrics. A sample-flag recount verifies arithmetic, not all assumptions of the original execution harness.

[Build verification](reports/sync_verification.json) · [Input/capture issues](reports/sync_issues.json) · [All figures](reports/figures/all/index.json)

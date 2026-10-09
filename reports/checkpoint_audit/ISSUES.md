# Evidence and protocol issues

Raw files have not been edited. Discrepancies below describe the archived evidence. No score is selected because it is the highest.

No arithmetic or token-accounting discrepancy was detected by the implemented checks. This does not prove that every original benchmark implementation was correct.

## Global limitations

Multiple-choice summaries do not include a complete versioned evaluation manifest or per-example predictions. MMLU `mmlu_cf` and `mmlu_mc5` are distinct custom archived modes, not interchangeable labels for an official leaderboard number. GSM8K summaries lack full per-example scoring evidence in the captured suite detail files.

Old manifests reference different evaluator and generation code hashes. Curves across these versions are not controlled comparisons. Complete source-hash resolution is listed in evaluation_versions.json.

Some legacy and smoke evaluations contain only a handful of examples. They are preserved, explicitly labeled, and not substituted for full-run values.

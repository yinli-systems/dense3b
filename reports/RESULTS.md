# dense3b — all checkpoints and available results

Evaluation capture started **2026-10-09T23:20:38.827803+00:00**; pinned-weight file stats have their own timestamp in provenance.

**Every located checkpoint appears below, including checkpoints with no benchmark result.** Historical/custom and corrected EvalPlus protocols remain distinct. Finished task results from still-running jobs are included; unfinished tasks have no invented score.

[All result values and versions (CSV)](evaluation_history.csv) · [All result values (JSON)](evaluation_all.json) · [Protocol manifests](evaluation_versions.json) · [Per-example recount checks](sample_verification.json) · [Training / MFU / throughput](SYSTEMS.md) · [Campaign progress](CAMPAIGN.md)

| Variant | Step / result page | Weight file present in latest captures | Verified metadata tokens (B) | Result rows | Missing full-count primary task families |
|---|---:|---|---:|---:|---|
| dense3b | [1499](checkpoints/dense3b-step_00001499.md) | True | — | 113 | none at recorded sample counts |
| dense3b | [2499](checkpoints/dense3b-step_00002499.md) | True | — | 66 | none at recorded sample counts |
| dense3b | [2999](checkpoints/dense3b-step_00002999.md) | False | — | 0 | arc_challenge, arc_easy, boolq, cruxeval, gsm8k, hellaswag, humaneval, mbpp, mmlu_cf, mmlu_mc5, openbookqa, sciq, winogrande |
| dense3b | [3249](checkpoints/dense3b-step_00003249.md) | False | — | 0 | arc_challenge, arc_easy, boolq, cruxeval, gsm8k, hellaswag, humaneval, mbpp, mmlu_cf, mmlu_mc5, openbookqa, sciq, winogrande |
| dense3b | [3499](checkpoints/dense3b-step_00003499.md) | False | — | 0 | arc_challenge, arc_easy, boolq, cruxeval, gsm8k, hellaswag, humaneval, mbpp, mmlu_cf, mmlu_mc5, openbookqa, sciq, winogrande |
| dense3b | [4499](checkpoints/dense3b-step_00004499.md) | False | — | 66 | none at recorded sample counts |
| dense3b | [4999](checkpoints/dense3b-step_00004999.md) | True | — | 105 | none at recorded sample counts |
| dense3b | [5249](checkpoints/dense3b-step_00005249.md) | False | — | 0 | arc_challenge, arc_easy, boolq, cruxeval, gsm8k, hellaswag, humaneval, mbpp, mmlu_cf, mmlu_mc5, openbookqa, sciq, winogrande |
| dense3b | [5499](checkpoints/dense3b-step_00005499.md) | False | — | 72 | none at recorded sample counts |
| dense3b | [5749](checkpoints/dense3b-step_00005749.md) | False | — | 68 | none at recorded sample counts |
| dense3b | [5999](checkpoints/dense3b-step_00005999.md) | True | — | 39 | none at recorded sample counts |
| dense3b | [6249](checkpoints/dense3b-step_00006249.md) | True | — | 0 | arc_challenge, arc_easy, boolq, cruxeval, gsm8k, hellaswag, humaneval, mbpp, mmlu_cf, mmlu_mc5, openbookqa, sciq, winogrande |
| dense3b | [6499](checkpoints/dense3b-step_00006499.md) | True | — | 0 | arc_challenge, arc_easy, boolq, cruxeval, gsm8k, hellaswag, humaneval, mbpp, mmlu_cf, mmlu_mc5, openbookqa, sciq, winogrande |

## Evaluation figures

Full-count scores are plotted as separate points by protocol. No curve connects incompatible evaluators; statistical significance is not implied.

[eval_arc_challenge_acc](figures/all/eval_arc_challenge_acc.png) / [SVG](figures/all/eval_arc_challenge_acc.svg)  
[eval_arc_challenge_acc_norm](figures/all/eval_arc_challenge_acc_norm.png) / [SVG](figures/all/eval_arc_challenge_acc_norm.svg)  
[eval_arc_easy_acc](figures/all/eval_arc_easy_acc.png) / [SVG](figures/all/eval_arc_easy_acc.svg)  
[eval_arc_easy_acc_norm](figures/all/eval_arc_easy_acc_norm.png) / [SVG](figures/all/eval_arc_easy_acc_norm.svg)  
[eval_boolq_acc](figures/all/eval_boolq_acc.png) / [SVG](figures/all/eval_boolq_acc.svg)  
[eval_boolq_acc_norm](figures/all/eval_boolq_acc_norm.png) / [SVG](figures/all/eval_boolq_acc_norm.svg)  
[eval_cruxeval_cruxeval_i](figures/all/eval_cruxeval_cruxeval_i.png) / [SVG](figures/all/eval_cruxeval_cruxeval_i.svg)  
[eval_cruxeval_cruxeval_o](figures/all/eval_cruxeval_cruxeval_o.png) / [SVG](figures/all/eval_cruxeval_cruxeval_o.svg)  
[eval_fim_multi_exact](figures/all/eval_fim_multi_exact.png) / [SVG](figures/all/eval_fim_multi_exact.svg)  
[eval_fim_multi_pass_rate](figures/all/eval_fim_multi_pass_rate.png) / [SVG](figures/all/eval_fim_multi_pass_rate.svg)  
[eval_fim_multi_suffix_dup](figures/all/eval_fim_multi_suffix_dup.png) / [SVG](figures/all/eval_fim_multi_suffix_dup.svg)  
[eval_fim_multi_terminated](figures/all/eval_fim_multi_terminated.png) / [SVG](figures/all/eval_fim_multi_terminated.svg)  
[eval_fim_single_exact](figures/all/eval_fim_single_exact.png) / [SVG](figures/all/eval_fim_single_exact.svg)  
[eval_fim_single_pass_rate](figures/all/eval_fim_single_pass_rate.png) / [SVG](figures/all/eval_fim_single_pass_rate.svg)  
[eval_fim_single_suffix_dup](figures/all/eval_fim_single_suffix_dup.png) / [SVG](figures/all/eval_fim_single_suffix_dup.svg)  
[eval_fim_single_terminated](figures/all/eval_fim_single_terminated.png) / [SVG](figures/all/eval_fim_single_terminated.svg)  
[eval_gsm8k_gsm8k](figures/all/eval_gsm8k_gsm8k.png) / [SVG](figures/all/eval_gsm8k_gsm8k.svg)  
[eval_gsm8k_strict_accuracy](figures/all/eval_gsm8k_strict_accuracy.png) / [SVG](figures/all/eval_gsm8k_strict_accuracy.svg)  
[eval_hellaswag_acc](figures/all/eval_hellaswag_acc.png) / [SVG](figures/all/eval_hellaswag_acc.svg)  
[eval_hellaswag_acc_norm](figures/all/eval_hellaswag_acc_norm.png) / [SVG](figures/all/eval_hellaswag_acc_norm.svg)  
[eval_humaneval_humaneval](figures/all/eval_humaneval_humaneval.png) / [SVG](figures/all/eval_humaneval_humaneval.svg)  
[eval_humaneval_humaneval_plus](figures/all/eval_humaneval_humaneval_plus.png) / [SVG](figures/all/eval_humaneval_humaneval_plus.svg)  
[eval_lcb_lcb_added_in_v1](figures/all/eval_lcb_lcb_added_in_v1.png) / [SVG](figures/all/eval_lcb_lcb_added_in_v1.svg)  
[eval_lcb_lcb_added_in_v2](figures/all/eval_lcb_lcb_added_in_v2.png) / [SVG](figures/all/eval_lcb_lcb_added_in_v2.svg)  
[eval_lcb_lcb_added_in_v3](figures/all/eval_lcb_lcb_added_in_v3.png) / [SVG](figures/all/eval_lcb_lcb_added_in_v3.svg)  
[eval_lcb_lcb_added_in_v4](figures/all/eval_lcb_lcb_added_in_v4.png) / [SVG](figures/all/eval_lcb_lcb_added_in_v4.svg)  
[eval_lcb_lcb_added_in_v5](figures/all/eval_lcb_lcb_added_in_v5.png) / [SVG](figures/all/eval_lcb_lcb_added_in_v5.svg)  
[eval_lcb_lcb_added_in_v6](figures/all/eval_lcb_lcb_added_in_v6.png) / [SVG](figures/all/eval_lcb_lcb_added_in_v6.svg)  
[eval_lcb_lcb_release_v6_cumulative](figures/all/eval_lcb_lcb_release_v6_cumulative.png) / [SVG](figures/all/eval_lcb_lcb_release_v6_cumulative.svg)  
[eval_lcb_lcb_v6_slice_175_post_cutoff](figures/all/eval_lcb_lcb_v6_slice_175_post_cutoff.png) / [SVG](figures/all/eval_lcb_lcb_v6_slice_175_post_cutoff.svg)  
[eval_mbpp_mbpp](figures/all/eval_mbpp_mbpp.png) / [SVG](figures/all/eval_mbpp_mbpp.svg)  
[eval_mbpp_mbpp_plus](figures/all/eval_mbpp_mbpp_plus.png) / [SVG](figures/all/eval_mbpp_mbpp_plus.svg)  
[eval_mmlu_cf_acc](figures/all/eval_mmlu_cf_acc.png) / [SVG](figures/all/eval_mmlu_cf_acc.svg)  
[eval_mmlu_cf_acc_norm](figures/all/eval_mmlu_cf_acc_norm.png) / [SVG](figures/all/eval_mmlu_cf_acc_norm.svg)  
[eval_mmlu_mc5_acc](figures/all/eval_mmlu_mc5_acc.png) / [SVG](figures/all/eval_mmlu_mc5_acc.svg)  
[eval_mmlu_mc5_acc_norm](figures/all/eval_mmlu_mc5_acc_norm.png) / [SVG](figures/all/eval_mmlu_mc5_acc_norm.svg)  
[eval_openbookqa_acc](figures/all/eval_openbookqa_acc.png) / [SVG](figures/all/eval_openbookqa_acc.svg)  
[eval_openbookqa_acc_norm](figures/all/eval_openbookqa_acc_norm.png) / [SVG](figures/all/eval_openbookqa_acc_norm.svg)  
[eval_sciq_acc](figures/all/eval_sciq_acc.png) / [SVG](figures/all/eval_sciq_acc.svg)  
[eval_sciq_acc_norm](figures/all/eval_sciq_acc_norm.png) / [SVG](figures/all/eval_sciq_acc_norm.svg)  
[eval_winogrande_acc](figures/all/eval_winogrande_acc.png) / [SVG](figures/all/eval_winogrande_acc.svg)  
[eval_winogrande_acc_norm](figures/all/eval_winogrande_acc_norm.png) / [SVG](figures/all/eval_winogrande_acc_norm.svg)  

# Vendored LoCoMo scoring code

`evaluation.py` and `evaluation_stats.py` are copied **verbatim** from the
official LoCoMo repository ([snap-research/locomo](https://github.com/snap-research/locomo),
`task_eval/`), so that our scoring is provably LoCoMo's own — token-F1
(`eval_question_answering`) and the per-category aggregation
(`analyze_aggr_acc`) run unmodified.

These two files are © Snap Inc., released under the LoCoMo license
(CC BY-NC 4.0): attribution required, non-commercial use only. They are
included here solely for benchmark reproduction — see the NOTICE file at the
repo root.

# Raw provenance data

- `case_results_deid.parquet/.csv` — Ding 2026 de-identified per-run rows (5,300 × 24),
  canonical tidy per-run rows. Source: `github.com/dingkaihua/self_consistency_as_predictor_of_accuracy`
  (CC BY 4.0). Columns: `student_id` (S001–S053), `axis`, `condition`, `case_id`,
  `benchmark` (gpqa_diamond|aime), `model` (gpt-4.1-nano|mini|4.1), `prompt`,
  `K` (=50), `majority_answer`, `ground_truth`, `subject`, `year`,
  `n_distinct_answers`, `n_unparseable`, `C` (self-consistency = n_majority/K),
  `A` (sample accuracy = n_correct/K), `majority_is_correct` (M), R_* Wilson helpers.

- `answer_distributions_deid.parquet` — 5,300 × 17, same keys plus `answer_counts`
  (JSON map answer→count over K samples) and `total` (=K).

These are copied verbatim (no modification) from the upstream release on 2026-08-16.
Do not hand-edit; if the source changes, re-download and re-copy, then re-run the chain.

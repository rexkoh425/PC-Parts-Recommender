# Evaluation evidence

This directory stores versioned evaluation datasets and their human-readable cards. Model
outputs belong under `artifacts/ml/<task>/<run_id>/` and must reference a dataset manifest
hash.

## Non-negotiable rules

1. Split on the leakage unit: product family for entity resolution and performance models;
   query-intent family for retrieval and ranking.
2. Freeze the test split before tuning. Every test-set look must be recorded.
3. Give every row an explicit `is_synthetic` boolean. Synthetic rows may exercise software,
   but they are excluded from measured model metrics and résumé claims.
4. Store point-in-time source, price, stock, benchmark, feature, label, and parser versions.
5. Save row-level predictions, errors, metric sample counts, and confidence intervals.
6. Report retrieval recall as `pooled Recall@k` unless relevance outside the judged pool is
   known.
7. Use the same frozen candidates and qrels for BM25 and LambdaMART lift claims.

Start a dataset card and labelling rubric from the templates in `evals/templates/`.

# Weighted RRF diagnostic

This experiment tested whether a single BM25 weight and RRF constant could improve
the existing hybrid retrieval ranking. It reused the frozen 32-query BuildCores
silver qrels and exactly the same candidates and BM25/vector source rankings. Only
fusion weights changed; first-stage retrieval did not.

Queries were split by category and query ID into 16 training, 8 validation, and 8
sealed test groups (seed `20261108`). A 30-setting grid searched BM25 weights
`0.5, 0.75, 1.0, 1.25, 1.5, 2.0` and RRF `k` values `10, 20, 40, 60, 100` on
validation. The selected setting was BM25 weight `0.5`, vector weight `1.0`, and
`k=40`. Validation NDCG@10 was `0.4679`; equal-weight RRF (`k=60`) scored `0.3642`.

On the sealed eight-query test set, the selected configuration **did not improve**
NDCG@10:

| Ranker | NDCG@10 |
|---|---:|
| Equal-weight RRF (`k=60`) | 0.3138 |
| Weighted RRF (`BM25=0.5`, `vector=1`, `k=40`) | 0.2528 |

The paired change was `-0.0610` absolute (`-19.4%` relative), with a 95% query
bootstrap interval of `[-0.1744, 0.0362]` over 1,000 resamples. The challenger won
3 of 8 test queries. The validation lift did not generalize, so the tuned weights
must not become the default. Existing equal-weight RRF remains unchanged.

The labels are deterministic specification predicates, not human judgments. The
holdout is small and the interval includes zero. This result is a diagnostic only;
it does not establish customer preference, production readiness, or a resume claim.

Reproduce with:

```powershell
uv run --no-sync python scripts/benchmark_weighted_rrf_silver.py --device auto
```

The retained run is under
`artifacts/evaluation/weighted-rrf-silver/20261107T195443Z/` and contains the
data hashes, split assignments, full parameter grid, per-query test metrics, and
the frozen test rankings.

# Independent 200-query LambdaMART evaluation

The corrected cohort passed the strict automated independence audit:

- 200 query groups, test-only snapshot, seed `20261110`
- 0 cross-split matches
- 0 previous-query matches
- maximum previous-query similarity `0.78761` at threshold `0.80`
- 5,600 pooled candidates, 28 per query

The existing LambdaMART model was loaded unchanged from the sealed 53-query
experiment. It was not retrained or tuned on this cohort. Labels are from a
separate `gpt-6-luna` judge and are LLM-weak diagnostic evidence, not human
validation. All 5,600 judgments completed with zero failures.

| Method | NDCG@5 | NDCG@10 | MRR | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.47431 | 0.52908 | 0.76348 | 0.36603 | 1.00000 |
| Vector | 0.53022 | 0.56516 | 0.80207 | 0.34882 | 1.00000 |
| Equal-weight RRF | 0.53458 | 0.56829 | 0.82078 | 0.36449 | 1.00000 |
| LambdaMART | 0.62317 | 0.67230 | 0.87344 | 0.47462 | 1.00000 |

LambdaMART versus RRF: absolute NDCG@10 `+0.10401`, relative `+18.30%`,
paired 95% bootstrap CI `[0.06695, 0.14362]`, and query wins/losses/ties
`108/63/29`. MRR delta was `+0.05267` with CI `[0.01564, 0.09127]`.
Recall@50 is exactly 1.0 for every method because the fixed pool contains all
judged candidates, so it is not a meaningful retrieval-gain claim.

Per-category NDCG@10 showed the smallest gain on compatibility/upgrade queries
(RRF `0.71107`, LambdaMART `0.71534`) and the largest on complex multi-constraint
queries (RRF `0.33914`, LambdaMART `0.74856`). The largest regressions were
GPU `diverse-v1-gpu-00-008` (`-0.57932`) and CPU `diverse-v1-cpu-00-002`
(`-0.51058`), indicating unresolved hard-negative and feature-ordering failures.

Reranking latency was p95 `0.627 ms` and p99 `0.847 ms`. Search latency was not
timed by this frozen candidate-pool diagnostic. A 40-query blind human-review
template is provided, but no human labels or inter-reviewer agreement exist.

The new diagnostic result is directionally consistent with the original result
(original LambdaMART +45.10% over RRF on 53 queries), but the improvement is
smaller on the independent cohort (+18.30%). Equal-weight RRF remains the
production default because both evaluations use LLM-only labels.

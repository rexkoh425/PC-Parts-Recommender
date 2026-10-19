# Optimizer quality benchmark v2

Controlled synthetic scenarios; no customer observations or factual market prices.
Cohort: 1000 scenarios; SHA-256 `c6325def44e14b400af9c1629dce8333184f945c3080f68d4873e793d614f718`.

| Algorithm | Feasible | Mean objective | Improvement vs greedy | p95 runtime (ms) | Violations |
|---|---:|---:|---:|---:|---:|
| greedy | 14.4% | 5851726.444444444 | baseline | 0.150 | 0 |
| greedy-local | 59.9% | 5658875.532554257 | +0.00% | 3.402 | 0 |
| cp-sat | 90.0% | 5538846.471111111 | +0.00% | 3.400 | 0 |
| cp-sat-warm | 90.0% | 5538846.471111111 | +0.00% | 6.538 | 0 |

Exhaustive CP-SAT oracle matches: 200/200 solver/scenario checks.

The warm start only supplies a feasible hint; it does not alter the objective.

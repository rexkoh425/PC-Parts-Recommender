# Optimizer quality benchmark v2

Controlled synthetic scenarios; no customer observations or factual market prices.
Cohort: 5000 scenarios; SHA-256 `bc965ac22673f25589fe9d8044c16f4ff656e606cb86bd03a21735cf2d662f53`.

| Algorithm | Feasible | Mean objective | Improvement vs greedy | p95 runtime (ms) | Violations |
|---|---:|---:|---:|---:|---:|
| greedy | 15.9% | 5844445.261306533 | baseline | 0.155 | 0 |
| greedy-local | 60.3% | 5659087.451409618 | +0.00% | 3.467 | 0 |
| cp-sat | 90.0% | 5539970.127555556 | +0.00% | 3.437 | 0 |
| cp-sat-warm | 90.0% | 5539970.127555556 | +0.00% | 6.685 | 0 |

Exhaustive CP-SAT oracle matches: 200/200 solver/scenario checks.

The warm start only supplies a feasible hint; it does not alter the objective.

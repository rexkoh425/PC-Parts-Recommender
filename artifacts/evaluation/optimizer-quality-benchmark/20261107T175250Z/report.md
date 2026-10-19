# Optimizer quality benchmark v2

Controlled synthetic scenarios; no customer observations or factual market prices.
Cohort: 200 scenarios; SHA-256 `52a2f42f7efd901836f1cadf5e72dbdcf88c22ce3ff01309adbf0c295ff589f2`.

| Algorithm | Feasible | Mean objective | Improvement vs greedy | p95 runtime (ms) | Violations |
|---|---:|---:|---:|---:|---:|
| greedy | 15.0% | 5899391.566666666 | baseline | 0.157 | 0 |
| greedy-local | 49.0% | 5695684.173469388 | +0.00% | 3.275 | 0 |
| cp-sat | 83.5% | 5587577.646706587 | +0.00% | 3.176 | 0 |
| cp-sat-warm | 83.5% | 5587577.646706587 | +0.00% | 6.220 | 0 |

Exhaustive CP-SAT oracle matches: 80/80 solver/scenario checks.

The warm start only supplies a feasible hint; it does not alter the objective.

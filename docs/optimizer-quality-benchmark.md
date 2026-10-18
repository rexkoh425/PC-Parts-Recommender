# CP-SAT optimizer quality benchmark

Validate the methodology with 1,000 scenarios, then run the full 5,000-case cohort:

```powershell
uv run --no-sync python scripts/benchmark_optimizer_quality.py --count 1000 --seed 20261110
uv run --no-sync python scripts/benchmark_optimizer_quality.py --count 5000 --seed 20261110
```

Each output directory contains `scenarios.jsonl`, `scenarios-manifest.json`, `runs.csv`,
`summary.json`, `difficulty-breakdown.json`, `correctness-report.json`, `manifest.json`,
and `report.md`. The manifest binds the generated cohort by SHA-256. The 5,000-case
cohort is exactly 1,000 easy, 1,750 medium, 1,750 hard, and 500 infeasible controls.
All scenarios are controlled synthetic inputs; no retailer price, stock, or absent
specification is represented as observed market evidence.

## Interpretation

The benchmark preserves infeasible scenarios in the denominator while separately
reporting solved feasible cases. CP-SAT and the warm-start trial were independently
checked against exhaustive enumeration on the first 100 scenarios (100 CP-SAT
solver/scenario checks). The warm start is an explicit exploratory option; it is not
included in the default benchmark because it previously increased latency without
changing objective quality.

The benchmark reports paired objective changes only when both methods returned valid
solutions. It separately reports cases where the solver finds a build and greedy
does not. Mean objectives over each method's valid outputs have different cohorts and
must not be compared as paired quality estimates.

The completed frozen run is at
`artifacts/evaluation/optimizer-quality-benchmark/20261109T225252Z/`.

| Algorithm | Valid builds | Feasible-case success | Mean objective on valid outputs | Paired improvement vs greedy | p95 runtime | Optimal statuses |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 796/4,500 | 17.7% | 5,844,445.26 | baseline | 0.147 ms | 0 |
| Greedy + bounded repair | 3,015/4,500 | 67.0% | 5,659,087.45 | 0.00% over 796 comparable cases | 3.385 ms | 0 |
| CP-SAT baseline | 4,500/4,500 | 100.0% | 5,539,970.13 | 0.00% over 796 comparable cases | 3.385 ms | 4,500 |

The 500 infeasible controls were correctly identified by every algorithm. CP-SAT
found a valid build in 3,704 cases where basic greedy failed and in 1,485 cases where
bounded local repair failed. It solved every one of the 4,500 proven-feasible cases.
There were zero independent hard-constraint violations. All 100 exhaustive checks
were exact matches with zero relative optimality gap. The earlier opt-in warm-start
trial is not an improvement: p95 latency increased to 6.685 ms while objective
quality was unchanged.

These are synthetic engineering diagnostics, not customer-workload or market-quality
evidence. The strongest supported claim is a feasibility/constraint-solving claim,
not an objective-quality improvement claim.

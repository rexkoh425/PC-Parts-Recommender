# Optimizer build-generation evaluation

This evaluation measures builds actually returned by `BuildOptimizer`/OR-Tools CP-SAT. It is
separate from the generated compatibility-rule scenarios under
`artifacts/evaluation/compatibility-generated-v2`, which do not invoke the optimizer and cannot
support a build-generation count.

Run a bounded smoke evaluation:

```powershell
uv run python -m pc_build_recommender.optimizer.evaluation `
  --count 100 `
  --output-dir artifacts/evaluation/optimizer-generated-builds-v1
```

The CLI defaults to 100 requests and enforces these hard limits:

- at most 10,000 optimizer requests;
- at most four candidates per required category;
- at most five returned builds per request;
- at most 50,000 compact retained output records; and
- at most five seconds of CP-SAT time per profile.

Every returned build is revalidated after the optimizer call by an oracle that does not call
the optimizer's validation helper or inspect the CP-SAT model. The oracle independently checks
selection cardinality and catalogue membership, locks, requirements, stock, budget, forbidden
pairs, connectors, PSU power and headroom, reported totals, objective accounting, warnings,
profile/status accounting, and diversity between returned builds.
Each output is also rerun through the version-pinned complete-build compatibility engine;
FAIL, UNKNOWN, missing rule coverage, or a rule-version mismatch invalidates the output. A
compact digest and status counts for that compatibility report are retained with the output.

Artifacts retain a self-hashed record for every scenario and returned output. Loading an
artifact regenerates each deterministic request and repeats the independent checks before it
accepts the aggregate counters or claim assessment.

## The 10,000-build gate

The report field `claim_assessment.eligible` can become `true` only when all of the following
are proven by the retained records:

- at least 10,000 optimizer outputs were returned;
- at least 10,000 selected-product tuples were retained (the deterministic scenarios use disjoint
  component-ID namespaces, so this is an integrity check rather than evidence of market diversity);
- every returned output has a compact evidence record;
- every output was independently checked and passed; and
- every scenario and record hash verified.

The full bounded command is:

```powershell
uv run python -m pc_build_recommender.optimizer.evaluation `
  --count 10000 `
  --solutions-per-scenario 1 `
  --output-dir artifacts/evaluation/optimizer-generated-builds-v1
```

Do not describe this generated evaluation as 10,000 observed customer or market builds. Its
narrow claim scope is deterministic engineering evidence that CP-SAT returned complete builds
satisfying the evaluated `OptimizationProblem` hard constraints and `compat_v2` rules.

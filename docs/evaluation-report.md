# Evaluation report

Status: measured diagnostics recorded; production promotion and market-readiness gates remain unmet

Snapshot date: 2026-07-23

## Interpretation

This report deliberately separates targets, engineering checks, exploratory measurements, and
reportable measurements. A reportable value requires all of the following:

1. a frozen content-addressed dataset manifest;
2. declared row/group counts, split policy, seed, and leakage checks;
3. row-level synthetic provenance with all synthetic rows excluded;
4. the specified baseline and candidate evaluated on the same eligible test rows;
5. a verified evaluation artifact with sample counts and confidence intervals; and
6. code/model/data/version metadata sufficient to reproduce the run.

Passing unit tests or training on generated seed data is engineering evidence, not measured market
coverage or model quality.

## Results registry

`Not measured` means no qualifying artifact was verified at this snapshot. It does not mean zero.

| Area | Metric | Target | Measured result | Evidence status |
| --- | --- | ---: | --- | --- |
| Entity resolution | Precision at auto-match threshold | >= 0.99 | Not measured for PC domain | External Dn7 transfer diagnostic is not retailer evidence |
| Entity resolution | Recall | >= 0.94 | Not measured for PC domain | External Dn7 transfer diagnostic is not retailer evidence |
| Entity resolution | F1 | >= 0.96 | Not measured for PC domain | External Dn7 transfer diagnostic is not retailer evidence |
| Retrieval | Recall@50 | >= 0.95 | RRF 0.116227; BM25 0.113810 | 32-query silver diagnostic only; no human judgments and not promotable |
| Retrieval | RRF NDCG@10 vs BM25 | Diagnostic comparison | 0.308534 vs 0.193411 | Same frozen candidates, but predicate-derived silver labels make this non-promotable |
| Ranking | LambdaMART relative NDCG@10 lift vs BM25 | 15% to 18% | Not measured; no ranker trained | Human-graded query/product labels required |
| Performance | CPU Blender held-out R-squared | >= 0.85 | Not measured on an untouched valid target | Legacy v2/old-v3 target inversion is revoked; later corrected development-only v3 diagnostic is 0.8763, but adaptively explored |
| Performance | CPU Blender held-out MAPE | <= 12% | Not measured on an untouched valid target | Corrected development-only v3 diagnostic is 18.98%, above the 12% gate and not fresh promotion evidence |
| Performance | GPU Blender OPTIX held-out R-squared | >= 0.85 | Not measured on a qualifying target; pilot diagnostic is -1.8538 | Only 6 internal test rows / 5 families; cohort has 36 total families, below the 100-family credibility gate, and rows are external-claim-ineligible |
| Performance | GPU Blender OPTIX held-out MAPE | <= 12% | Not measured on a qualifying target; pilot diagnostic is 203.52% | Same underpowered pilot; LightGBM is worse than its median baseline and cannot be served |
| Performance | 2026-07-23 external temporal cohort support | >= 20 rows and >= 10 leakage families | 0 rows and 0 families; no metrics computed | 113 novel submissions / 339 observations, but none matched the complete frozen cohort; `insufficient_external_cohort` |
| Compatibility | Generated-scenario assertion/oracle mismatches | 0 | 0 across 10,000 deterministic scenarios | `compat_v2`; 526,300 assertions passed, aggregate engineering evidence only |
| Optimizer | Retained CP-SAT outputs independently valid | >= 10,000 | 10,000 of 10,000 | Every output passed the independent constraint oracle and `compat_v2`; scenario-specific component IDs make identity deterministic, so this is engineering scope rather than market-diversity evidence |
| Compatibility | Known hard violations in 10,000 retained market tests | 0 | Not measured on market data | The aggregate compatibility sweep retains no individual scenarios; the optimizer artifact retains 10,000 synthetic engineering scenario records, not observed market builds |
| Retailer data | Production readiness | All categories priced, mapped, rights-cleared, and known stock | Failed: 485 offers, 2 mapped, 0 known stock, 0 with explicit rights | Licensed Singapore feeds remain required |
| Price intelligence | Descriptive observed history | Rights-cleared stored history | Not measured on production retailer data | API/UI implementation exists; current offers fail rights gate and no real Awin feed was imported |
| Search API | p95 latency | < 500 ms | Not measured | Declared load profile required |
| Build API | p95 latency | < 2.5 s | Not measured | Candidate caps and solver limit required |
| User impact | Median selection-time reduction | 40% to 60% | Not measured | Counterbalanced user study required |

The results table must be updated only by reading a verified artifact. Non-promotable diagnostics
must remain visibly labelled and cannot be converted into production or resume claims. Do not
manually copy a training console score or a validation result into the measured column.

## Evaluation protocols

### Entity resolution

Use at least 2,500 labelled pairs for the portfolio evaluation, including true duplicates,
similar-variant hard negatives, and random negatives. Group related products/blocking families.
Freeze the auto-match threshold before the final test. Report precision, recall, F1, average
precision, auto-match coverage, calibration, confusion counts, and 95% intervals. Review numeric-
conflict, capacity, generation, form-factor, source, and brand slices. Precision is the primary
promotion gate.

### Retrieval and ranking

Use approximately 150 realistic queries with two-reviewer grades from 0 to 4. Split by query ID.
Evaluate BM25, vector-only, BM25+vector RRF, hybrid+filters, and hybrid+LambdaMART on the same
frozen judged set. Report Recall@20, Recall@50, MRR, NDCG@10, query count, pool coverage, and query-
bootstrap intervals.

For ranker lift, preserve identical candidate rows and compare per-query NDCG. Report baseline and
candidate NDCG, absolute lift, relative lift percentage, query win rate, and paired query-bootstrap
intervals. A model that improves mean NDCG but materially harms retained-component or constrained
queries does not automatically promote.

### Performance prediction

Create separate tasks by component category/workload. Group by product family or generation and
preserve benchmark context. Compare with a training-median and a simple linear/tree baseline.
Report MAE, RMSE, R-squared, Spearman, and MAPE only when every observed target is positive. Include
coverage and out-of-distribution rate. Review brand, generation, performance tier, and benchmark-
context slices.

### Compatibility and optimiser

Compatibility tests include known builds, hard failures, exact clearance/wattage boundaries,
connector cases, BIOS warnings, missing fields, and generated configurations. Record every
applicable rule result and rule version. Independently recheck every generated build.

For reduced catalogues, enumerate every valid combination and compare feasibility and exact
objective value with CP-SAT. Test infeasible explanations, time limits, incumbent validation,
determinism under a fixed seed, and diversity. A count of generated test cases is not evidence of
zero violations unless every saved result was independently checked.

### Latency and load

Declare hardware, OS/container limits, catalogue/listing count, concurrent users, request mix,
warm-up, duration, percentile method, cache state, candidate caps, solver time limit, and database
state. Measure search and build endpoints separately. Report p50, p95, p99, error/empty rate,
throughput, database time, retrieval time, model time, compatibility time, and solver time.

### User study

Use about 20 participants and counterbalance manual-research versus website task order. Tasks must
be comparable but not reusable. Record median completion time, compatibility mistakes, blinded
build-quality score, confidence, and retailer pages visited. Report attrition, task/order effects,
paired uncertainty, and participant experience. Do not report only the mean.

## Artifact contract

Dataset manifests use `pc-build-recommender.dataset-manifest.v1` and include file hashes, row and
group counts, synthetic-data declaration, metadata, and a content SHA-256. Evaluation artifacts
use `pc-build-recommender.evaluation-artifact.v1` and include task/run ID, manifest digest,
reportability, metrics, sample counts, confidence intervals, run metadata, and an artifact hash.

Recommended immutable layout:

```text
artifacts/evaluations/
  entity_resolution/<run_id>/dataset-manifest.json
  entity_resolution/<run_id>/evaluation.json
  retrieval/<run_id>/dataset-manifest.json
  retrieval/<run_id>/evaluation.json
  ranking/<run_id>/evaluation.json
  performance/<task>/<run_id>/evaluation.json
  compatibility/<run_id>/evaluation.json
  load/<run_id>/summary.json
  user_study/<run_id>/summary.json
```

Generated model binaries and source datasets can remain outside Git, but their digests and access
instructions must be retained. MLflow supplements this contract; an MLflow run alone is not enough
for a public claim if dataset/synthetic provenance is absent.

Human-label releases are produced only by the durable annotation workflow. A freeze requires two
independent immutable judgments per item, independent adjudication of disagreement, source-policy
and non-synthetic gates, and leakage-group-safe frozen splits. Relevance releases retain
`human-judgments.json`, `qrels.json`, `query-split.json`, and evidence snapshots. Entity-resolution
releases retain `human-labels.json`, `pairs.jsonl`, `listing-split.json`, raw reviewer decisions,
adjudication, and a manifest whose release identity binds every file. No such release exists at
this snapshot.

LambdaMART model publication is independently crash-atomic and immutable. The v2 bundle manifest
binds a publication-intent SHA-256 to exact feature/human/qrels/split hashes and training settings;
a staged bundle is reloaded and verified before one no-replace directory rename. This is artifact
integrity evidence only. It does not substitute for a human dataset, paired evaluation, or the
unmeasured 18% target.

## Engineering verification versus model evidence

Current infrastructure supports the following repeatable engineering checks:

```powershell
./scripts/test.ps1 -Suite python
docker compose --env-file .env.example config --quiet
uv run alembic -c db/alembic.ini upgrade head
```

These prove code/schema/config behavior only. Host GPU availability can be checked with
`scripts/setup-gpu.ps1`, but detecting an RTX 5070 Ti does not prove any model metric.

## Claim checklist

Before stating a catalogue count, count unique canonical product IDs and unique retailer/source-
listing pairs from the named data version; exclude price snapshots. Before stating ranking lift,
verify paired BM25 and LambdaMART metrics on the same candidate set. The retained optimizer-output
artifact below supports the narrow statement that OR-Tools generated 10,000 independently valid
engineering builds. It does not support "market-tested", "customer", or market-representative
wording. Before stating time reduction, retain the counterbalanced participant-level study results
and report medians.

No résumé, README, demo, or API copy should present a target from this report as achieved. When an
eligible artifact exists, add its relative path, hash, run ID, dataset size, point estimate,
confidence interval, and limitations beside the measured result.

## Verified engineering evidence at this snapshot

This evidence demonstrates implementation behavior, not portfolio-scale model quality:

- The initial Alembic schema completed a live PostgreSQL/pgvector upgrade, downgrade, and
  re-upgrade. The verification observed pgvector 0.8.5, 13 application tables, and a
  `vector(384)` embedding column.
- Dagster loaded seven asset definitions in an isolated environment with the optional pipeline
  dependencies. The base environment also imported `pipelines.definitions` without Dagster,
  proving the optional-dependency fallback. The built optional-service containers also returned
  HTTP 200 from Dagster `/server_info` and MLflow `/health` against an isolated PostgreSQL service.
- The complete default Compose stack built and started in isolation. API readiness returned
  `ready`, the compiled Next.js service returned HTTP 200, the example request produced four
  seed-catalogue builds, and the migrated database contained 13 tables. The four builds establish
  transport/orchestration behavior only; an in-memory seed response is not recommendation-quality
  or market-coverage evidence.
- A bounded 30-second Locust development run retained four raw CSV outputs and a
  [content-addressed evidence record](../artifacts/evaluation/load/development-demo-api-mix-20260723-local-v2/summary.json).
  It used two users against the loopback in-memory demo API with 23 products/listings, a cold cache,
  `demo-seed-2026-07-22`, deterministic baseline ranking, `compat_v2`, and an unverified development
  release. The retained CSV rows report zero failures across 265 searches and 92 build-generation
  requests, with p95 values of 140 ms and 250 ms respectively. This is repeatable local smoke
  evidence only: its checked-in profile is explicitly `development_only`, it has no production
  catalogue, database, pinned serving release, or representative concurrency, and it does not make
  either latency target achieved.
- `tests/unit/test_optimizer.py` and `tests/property/test_optimizer_properties.py` completed 18
  focused tests. The exact oracle enumerated all 256 combinations in an eight-binary-choice
  catalogue and matched CP-SAT objective and selection. Thirty generated cases compared four
  combinations each (120 total), plus 25 PSU-wattage monotonicity examples. The diversity test
  returned three builds differing pairwise in at least two categories. These reduced/generated
  cases validate implementation invariants; they do not independently satisfy the retained,
  market-representative 10,000-case compatibility target.
- The focused compatibility suite completed 38 deterministic/property tests covering required
  rule families, retained parts, complete-build cardinality, boundary monotonicity, and fail-closed
  missing data.
- The [`compat_v2` aggregate artifact](../artifacts/evaluation/compatibility-generated-v2/compatibility-generated-compat_v2-seed-20260722-n-10000-c09c0b2d69ffd95f.json)
  evaluates 10,000 deterministically generated configurations.
  It records 526,300 exact-outcome, independent-oracle, and monotonic assertions with zero failed
  assertions and zero oracle mismatches. The expected outcome mix was 1,000 PASS, 8,000 FAIL, and
  1,000 UNKNOWN scenarios. Its claim scope explicitly records zero observed market builds and zero
  retained scenario records, so it is engineering validation rather than evidence of 10,000 valid
  or market-tested builds. Semantic artifact digest:
  `c09c0b2d69ffd95ff4ea8ea8084c58747e16110b7fbb284d2b84d4c5c5ab1985`. Exact file SHA-256:
  `bf2ac3a54906bf004066a09cd685d9e6ab8c425e0e2af73e786f54d99ca73e9f`.
- The [retained CP-SAT output artifact](../artifacts/evaluation/optimizer-generated-builds-v1/optimizer-generated-seed-20260723-n-10000-12c1305bec5666d4.json)
  records 10,000 optimizer invocations and 10,000 retained complete outputs. All solver statuses were
  OPTIMAL. Every output was independently rechecked against the frozen request, selection universe,
  budget, locked parts, features, power policy, connectors, pairwise constraints, objective, and
  diversity accounting, then passed `compat_v2`. The retained reports contain 340,000 PASS results
  and zero FAIL, WARNING, or UNKNOWN results. The artifact is eligible for the narrow engineering
  claim "OR-Tools generated 10,000 complete builds that were independently revalidated"; it is not
  evidence of 10,000 observed customer or market builds or of market diversity; deterministic
  scenario-specific component IDs make output identity distinct by construction. The verifier checks
  the exact current optimizer, compatibility-engine, and evaluation-harness source hashes before
  replaying the retained records. Semantic artifact digest:
  `12c1305bec5666d4561d167ec64fd09a850409857f7edbaa053e4cac5c7eea74`.
  Exact file SHA-256:
  `169c03872943920c741589f11dd450d8b00732e6ff4620022a968dbbe26e7e8e`.
- The full BuildCores materialization contains 25,666 accepted canonical-product records across all
  eight categories and 33 rejections. ODC-By 1.0 attribution applies, and the quality artifact
  reports complete source-record IDs and per-product provenance. These are community catalogue
  records, not authoritative compatibility verification, retailer offers, or loaded production
  database rows.
- The Blender adapter completed a full scan of 422,319 submissions and 1,243,834 valid
  observations, then selected a deterministic 250,000-observation hash sample. The selection seed
  and source/content hashes are retained; 1,000 rejection records were retained from a truncated
  rejection log. The data card records the exact paths and digests.
- The controlled retailer readiness artifact covers 485 offers. Two were safely mapped by exact
  manufacturer-part-number plus brand, 483 remain unmatched, and zero have known in-stock status.
  All 485 lack explicit data-use rights, all use-right booleans are false, and the production gate
  fails. This is parser/readiness evidence only, not Singapore retailer coverage.
- The full-catalogue retrieval diagnostic contains 32 queries and 102,664 query-candidate rows.
  RRF NDCG@10 is 0.308534 versus 0.193411 for BM25, while RRF Recall@50 is 0.116227 versus 0.113810.
  Labels are deterministic specification predicates, not human judgments; the artifact is
  non-promotable and no LambdaMART model was trained. Semantic artifact digest:
  `8c0f3383dee1e3a197d44e53cdcf0fe3e6b1bc2a714303aa82eff39e788bbea9`. Exact
  `metrics.json` SHA-256:
  `13ba6c7e074157d6bb654c1b4be5d244dfb9b02f3004b69464183e214efb2caf`.
- The v3 human-relevance collection starter commits 16 author-curated queries and 480 blinded
  query-product tasks against the 3,000-row BuildCores portfolio. Its non-reviewer pre-label
  snapshot SHA-256 is
  `4124c5afc9a1a32c85027ca1f72f1314022a05dc065a554398f8f68386335de1`; the exact feature-file
  SHA-256 is `b76de63756a0afb6d31b3cd6bc7230dff747133e83ca108be40ad774ed2397b7`.
  The annotation batch preserves those hashes and splits to 10/3/3 train/validation/test intent
  groups. This is collection-lineage evidence only: no reviews or labels exist, the pool uses the
  deterministic development vector fallback, and three test groups cannot satisfy the 50-group
  promotion floor.
- Blender CPU v2 and the legacy v3 diagnostic are revoked. Their source observations were native higher-is-better
  `samples/minute`, but preparation labelled them as seconds and inverted the target. Their
  reported regression metrics are therefore invalid and are not retained as model-quality or
  promotion evidence.
- The corrected `cpu/content_creation` development artifact preserves the median
  `samples_per_minute` target. It contains 172 real hardware rows across 109 leakage groups and is
  app-routable only with explicit unpromoted-model opt-in. On the already-observed 20-row,
  13-group internal split, R-squared is 0.9408 and MAPE is 22.23%; grouped-bootstrap intervals are
  0.7173 to 0.9648 and 13.59% to 32.48%. These are development diagnostics only because the
  snapshot was adaptively explored after v2/v3 results were inspected and no untouched external
  frozen cohort exists. Precise predictions and promotion remain disabled. Model version:
  `e4f5f153d4e42ed25a37cf91eaf95dd0dc968942e3843ff50d41cf61e40fab21`.
- A separate corrected [`v3 development diagnostic`](../artifacts/ml/performance/blender_cpu_content_creation_v3_development_diagnostic/diagnostic_report.json)
  held calibration and holdout groups out of candidate selection. On 113 development rows, its
  engineered identity candidate improved out-of-fold MAPE from 21.98% to 19.34% (12.04% relative)
  while increasing R-squared from 0.8926 to 0.9091. Its later 30-row / 15-group holdout was
  R-squared 0.8763 and MAPE 18.98%; grouped-bootstrap bounds were 0.6279 to 0.9429 and 14.96% to
  24.02%. This is deliberately non-promotable: it is a development-only artifact, the source was
  adaptively explored, and the point and bootstrap MAPE gates fail. Model version:
  `e38a72f9528523e4f298e83f41edb7691b1af844fc2953089431022cfa15fe00`; exact report SHA-256:
  `8128d8403ab2d3275ee026e2afb07df6c195f64597a55d74aac94de4a9bf3eb7`.
- The sealed GPU OPTIX pilot fixes one actual Blender benchmark identity (4.5.0 / `junkshop` /
  Windows) and uses the correctly oriented `1000 / median_render_seconds` throughput target. Its
  1,939 matched observations aggregate to 50 hardware rows / 36 product-family groups. Its
  24 / 13 / 7 / 6-row group-disjoint train / validation / calibration / test partitions yield
  LightGBM test R-squared -1.8538 and MAPE 203.52% on only 6 rows / 5 groups, versus median MAPE
  50.02%. The manifest marks every row external-claim-ineligible and requires 100 groups for a
  credible evaluation; calibration and test minimums fail as well. The artifact is sealed,
  development-relative-only, not routed, and not promotion evidence. Model version:
  `fb2f42fccae0ac6965d28118fc26b5f5b811aa3fed8f6599fc4013a710a75192`; report SHA-256:
  `48170c820ef1eb3a5a9425089a5c938ec0cd44d186536fc988cc9562cb6f8681`.
- The [2026-07-23 temporal audit](../artifacts/evaluation/performance-temporal-v4/957139150dbc1bbdd6deab1810b966ea59a33d9768b206e5033287c88a3f044b.json)
  compared 422,319 old submissions with 422,432 new submissions and isolated 113 novel
  submissions / 339 observations. Zero observations met the complete frozen Blender 4.0.0 /
  `junkshop` / CPU / Windows / build-hash / script / scene-checksum contract. It therefore retained
  0 candidate rows and 0 leakage families, attempted no inference, computed no accuracy metric,
  pooled no data, and left promotion disabled. Semantic report digest:
  `957139150dbc1bbdd6deab1810b966ea59a33d9768b206e5033287c88a3f044b`; exact report-file SHA-256:
  `e675a0d1de5f0556aedeebb41c93f19f8130f3af22b3a010b7dc9c789fb458cf`; protocol SHA-256:
  `6952cd7a220920fc9be882211577a67a67268a055e702e4b64955ada5123154a`.
- The signed Awin local-feed path, durable annotation ledger, atomic LambdaMART bundle publisher,
  sealed ER/catalogue release loader, pinned offline semantic encoder, and descriptive price API/UI
  all have implementation tests. None supplies a real Awin catalogue, independent human labels, a
  promoted ER/LambdaMART/encoder bundle, ranking lift, or production price-history metric; those
  evidence cells remain intentionally unmeasured.
- A synthetic held-out performance fixture produced LightGBM R-squared 0.9646 and MAPE 3.30%,
  Ridge R-squared 0.8737 / MAPE 6.25%, and median R-squared -0.0397 / MAPE 20.63%. These values are
  deliberately excluded from the results registry: they validate code against a generated signal,
  not real-hardware generalisation. The device probe recorded CPU fallback.
- The external Zenodo Dn7 transfer evaluation used source-record-disjoint train/validation/test
  splits of 15,951 / 1,945 / 1,802 pairs with 481 / 144 / 138 positives. At the validation-F1
  threshold, CPU LightGBM test precision was 0.81102, recall 0.74638, F1 0.77736, and average
  precision 0.81666. A threshold selected for at least 0.99 validation precision produced 1.0
  test precision but 0.05072 recall. It missed the combined target and is transfer-only, not
  PC-retailer production evidence. Report SHA-256:
  `6f81289e73584f2894ea9ddf8380c4201140927b56584fa6203877981381a022`.

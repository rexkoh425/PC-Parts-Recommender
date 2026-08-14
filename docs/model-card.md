# Model card

Status: development; real-data diagnostics exist, but no supervised model is production-promoted

Last updated: 2026-07-23

## System role

The recommendation product contains three supervised model families and one embedding model:

1. a LightGBM binary entity-resolution classifier;
2. workload/category-specific LightGBM regressors;
3. a LightGBM LambdaMART component ranker; and
4. a Sentence-Transformers encoder for semantic candidate retrieval.

Release-bound BM25 in production, PostgreSQL full-text diagnostics/fallback, reciprocal-rank
fusion, deterministic compatibility rules, descriptive price statistics, and CP-SAT optimisation
are essential system components but are not learned models. Learned scores never override hard
compatibility, stock, retained-part, feature, or budget constraints.

In the processed-catalogue production path, the product-price endpoint exposes only
`descriptive_observed_history`: current delivered price, 30/90-day medians, 90-day
percentile/recent low/volatility, seller/stock trends, history sufficiency, and robust anomaly
flags from stored observations. It withholds percentile and volatility for sparse history and
bounds analysis to the newest 10,000 observations. This is not a forecast and cannot guarantee a
current or future retailer price; the production summary is absent unless all loaded offers pass
the production-rights gate. Seed and public-demo summaries are illustrative UI data, not
rights-cleared retailer evidence.

No model is considered production-promoted merely because training code runs. Promotion requires
a frozen dataset manifest, leakage-safe split, baseline comparison, verified reportable evaluation
artifact, failure-slice review, versioned model artifact, and serving compatibility test.

## Intended use

The models support new consumer desktop-component recommendations in Singapore for gaming,
software development, content creation, and local-AI workloads. Outputs narrow choices and
estimate relative workload value. They are not safety certifications, purchase execution,
professional electrical advice, exact thermal simulation, or guarantees of future availability,
price, performance, software support, or reliability.

## Entity-resolution classifier

Input is a retailer-listing/canonical-product pair produced by conservative blocking. Features
include exact MPN, brand/category equality, model-token and character similarity, numeric-token
agreement, capacity/form-factor/specification agreement, title embedding cosine similarity, and
relative price difference. Numeric conflicts carry strong negative evidence.

Output is duplicate probability. Initial operating policy:

- probability at least 0.98: eligible for automatic match;
- 0.80 through less than 0.98: manual review; and
- below 0.80: reject.

Thresholds are starting hypotheses. They must be calibrated on a frozen labelled set and may be
raised when the lower confidence bound for auto-match precision is insufficient. Precision has
priority over recall because false merging spreads incorrect price and benchmark evidence.

Fallback: exact trusted identifiers and deterministic rules may match; otherwise leave the
listing unmatched or queue it. An uncertain model result must never force a merge.

Production authority cannot come from a command-line eligibility flag. Serving manifest schema
`pc-build-recommender.serving-release.v4` binds the exact catalogue, offers, reviewed mappings,
review evidence, and entity-resolution model, metadata, fitted calibrator, serving evidence,
human-labelled v2 evaluation, threshold policy, and rights approval. Their artifact-core,
model-file, metadata, calibrator, evidence, evaluation, policy, rights, model-release, and
aggregate binding SHA-256 values are cross-checked. The policy requires human labels,
confidence-supported precision, recall/F1/support gates, active Singapore rights for
training/metrics/model serving, and the exact matcher/projection versions. Legacy self-attested
booleans and direct diagnostic CLI paths confer no production authority. No genuine promoted
PC-domain entity-resolution release is shipped.

Version 4 also requires a content-addressed signed retailer-source release. Its raw snapshot,
rejections, externally mounted current registry, independently configured trust-root pin, and source manifest are verified while the exact
governed-offers file is supplied as the accepted-record artifact. This is an admission control,
not model-quality evidence. The current schema admits one Awin source batch only.

## Performance regressors

Separate models estimate supported targets such as GPU gaming/local-AI scores, CPU single-core,
multicore and compilation scores, and storage performance. Features are category-specific hardware
attributes and generation indicators. Training targets are normalised only inside comparable
benchmark contexts.

Validation groups product family or hardware generation. Direct observations for the requested
benchmark context always take priority. A prediction includes `predicted`, model version,
confidence, and supporting evidence; insufficient confidence yields a relative score instead of a
precise estimate.

Fallback: use comparable observed aggregates or clearly labelled relative baselines. Do not
impute a precise value from unrelated resolution, preset, operating system, benchmark version, or
driver context.

Current serving policy: Blender CPU v2 and v3 are revoked because their preparation inverted
native higher-is-better `samples/minute` observations. The corrected `cpu/content_creation`
artifact is app-routable only with explicit development opt-in. It is non-promotable, precise
predictions are disabled, and it may support only a clearly labelled relative workload score;
direct comparable observations still take priority.

The Blender GPU OPTIX pilot is not app-routable, even with development opt-in. Its exact route is
the benchmark identity `gpu/blender_4_5_0_junkshop_optix_windows`, not the user-facing
`content_creation` workload. The pilot has too few groups, has external-claim-ineligible rows,
and fails every held-out accuracy and calibration gate. It exists solely as sealed negative
evidence that the current GPU cohort must not be served. The preparation manifest evaluates 383
exact GPU cohorts from this snapshot; the selected 36-family cohort is the largest and none has
50 families, so changing to another recorded cohort cannot satisfy the 100-family gate.

The 2026-07-23 temporal audit found 113 new submissions and 339 new observations in the newer
Blender snapshot, but zero rows met the complete frozen 4.0.0 / `junkshop` / CPU / Windows cohort,
including build hash, benchmark script, and scene checksum. It therefore performed no inference
and produced no external accuracy metrics. This is an `insufficient_external_cohort` result, not a
zero score and not promotion evidence.

## LambdaMART ranker

Training rows are query/product candidates with relevance grades 0 through 4 and query-ID groups.
Features cover BM25/vector/RRF, exact and specification match, observed/predicted workload score,
minimum-requirement fit, CPU/GPU balance, price and budget share, price-to-performance, price
history, availability, warranty/reliability, review aspects, age/freshness, and preferences.

NDCG at 10 is primary. The learned model is compared with BM25 on the same frozen query set and
identical candidate rows. Reported relative lift is:

```text
(candidate NDCG@10 - baseline NDCG@10) / baseline NDCG@10 * 100
```

Fallback: a deterministic weighted baseline using retrieval, workload, value, availability, and
freshness. Compatibility remains outside both baselines and learned ranking.

Candidate capture now commits the exact label-free ranking rows and feature matrix before human
review. The annotation batch retains only opaque row hashes, and
`training.materialize_ranking_snapshot` verifies the frozen release before appending adjudicated
grades. Human training and evaluation require that labeled-snapshot manifest; model metadata binds
the pre-label snapshot and feature contract.

The trainer publishes a complete model/metadata/manifest bundle through a verified hidden
sibling stage and one no-replace directory rename. A publication-intent SHA-256 binds the exact
feature snapshot, human judgments, qrels, frozen query split, data/candidate versions, feature and
ranker configuration, seed, and early stopping. The committed bundle is immutable: same-intent
concurrent/crash retries adopt the same exact bytes, while a different intent cannot overwrite it.
Training reports score only the committed model. A bounded opt-in maintenance command can dry-run
or remove sufficiently old inactive orphan stages; publication never performs automatic cleanup.
These controls harden feature and artifact integrity but do not supply labels, train a qualifying model, or establish
the target 18% NDCG@10 lift.

## Embedding encoder

The initial text encoder is `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional
vectors over category, brand/model, important specifications, workload/benchmark tags,
compatibility tags, and permitted review aspects. The model improves candidate discovery, not
compatibility. The production lexical path uses an immutable BM25 index built from the same
validated search documents as the active embedding release. PostgreSQL determines the current
eligible IDs before scoring, so structured filters remain authoritative. `ts_rank_cd` is retained
only for diagnostics/fallback; BM25 is also the baseline used by retrieval and ranking evaluation.

Embedding model, source-text hash, and update time are stored per product. Changing model or text
contract requires versioned re-embedding and offline comparison. Retrieval uses reciprocal-rank
fusion with an initial `k=60`, tuned only on validation queries.

In production, the logical model name cannot trigger a network download. The operator and serving
manifest must pin one content-addressed local encoder tree by path, SHA-256, file count, and byte
count. Validation streams a nonempty tree bounded to 4,096 files and 2 GiB, rejects symlinks or
junctions in every ancestor/entry, and detects file replacement. Sentence-Transformers loads only
that path with `local_files_only=true`; the API image installs the `serving` extra and sets Hugging
Face/Transformers offline modes. Startup must encode a finite, nonzero, L2-normalized probe of the
stored vector dimension before readiness reports the semantic encoder ready. No production weight
bundle is committed or published. The repository now includes a local-only packager that binds a
verified `sentence-transformers/all-MiniLM-L6-v2` snapshot and revision to the embedding manifest,
records Apache-2.0 provenance, and emits a content-addressed offline bundle. This establishes only
encoder reproducibility; production remains fail-closed pending a complete eligible serving release
with catalogue, rights, entity-resolution, ranker, and database evidence.

The current local operator run packages source bundle SHA-256
`0f4856ff5afd30b5b9cc9b3864c48d4daf24cbe9124bf2c7e21ceab6de297bf0` into the 12-file,
91,580,262-byte release bundle
`1b2b44b00bcb44485cea516dd91eba6a31be19ee33b7e6a212c1dbee21b9c19a`. With Hugging Face and
Transformers offline flags set, CPU warm-up returned the expected 384-dimensional, finite,
L2-normalized vector and the index fingerprint
`455e4bf8f5ebbca36ee5cc66a419c69346177f4cb77aca6be5e03d413313541c`. This is an operator-side
runtime check, not a retrieval-quality result or a production-serving approval.

## Training and compute

CPU training is the reproducible lockfile baseline. Semantic indexing dependencies are installed
explicitly with `uv sync --locked --extra embeddings`. The local Windows RTX 5070 Ti can
accelerate supported training. After that sync, `scripts/setup-gpu.ps1` installs the verified
`torch==2.13.0+cu130` CUDA wheel into `.venv` and confirms the actual device. A later `uv sync`
restores the CPU wheel in `uv.lock`; rerun the script before host GPU work.

Training metadata records random seed, library versions, CPU/GPU device, feature version, dataset
manifest hash, split/group policy, parameters, duration, and model artifact digest. GPU and CPU
results must remain within declared numeric tolerance before replacing a promoted artifact.

## Evaluation and promotion gates

| Model | Primary gate | Supporting gates | Initial target, not measured |
| --- | --- | --- | --- |
| Entity resolution | Auto-match precision | Recall, F1, AP, coverage, calibration, hard-negative slices | Precision >= 0.99, recall >= 0.94, F1 >= 0.96 |
| Retrieval/encoder | Recall@50 | Recall@20, MRR, NDCG@10, latency | Recall@50 >= 0.95 |
| Performance regressors | R-squared and MAPE | MAE, RMSE, Spearman, family/generation slices | R-squared >= 0.85, MAPE <= 0.12 |
| LambdaMART | Paired NDCG@10 lift | Absolute lift, query win rate, slices, latency | 15% to 18% relative lift over BM25 |

Point estimates alone are insufficient. Artifacts include sample counts and confidence intervals;
promotion considers lower bounds, operational coverage, and material failure slices. Small pilot
or synthetic-only results are labelled exploratory and are ineligible for public quality claims.

## Known limitations and risks

- Unseen generations, sparse low-volume categories, retailer title conventions, and close
  capacity variants can cause distribution shift.
- Benchmark availability is not uniform across brands or workloads, and target normalisation can
  encode benchmark-suite bias.
- Relevance labels reflect reviewer judgment and candidate-pool coverage; unjudged does not mean
  irrelevant.
- Interaction learning introduces position, exposure, popularity, and retailer-availability bias.
- Noise, reliability, software support, and upgradeability are partially observed and can be
  source-dependent.
- Embedding similarity can retrieve semantically related but physically incompatible products.

Required slices include component category, brand, hardware generation, price band, query
workload, retained-part query, size constraint, rare variant/hard negative, and data freshness.

## Monitoring

Online monitoring records model/data/rule versions, feature availability, candidate and filtered
counts, empty results, fallback rate, score distributions, latency, and downstream save/click/
dismiss events. Entity resolution also monitors auto-match rate, review queue volume, conflict
rate, and post-review precision. Performance models monitor unsupported/OOD rate and observed-
versus-predicted residuals when later benchmarks arrive.

Drift or contract mismatch triggers fallback and blocks promotion; it does not silently reuse a
model on incompatible features. Model rollback means restoring the last verified model alias and
its matching feature/data contract.

## Explainability

Entity decisions expose decisive matches and conflicts without presenting probability as proof.
Performance estimates expose observed/predicted status and supporting benchmark context. Ranking
explanations use stable user-facing feature groups rather than raw tree internals. Optimiser
explanations name binding constraints, objective profile, and component trade-offs. Explanations
may summarize computed evidence but may not create new technical claims.

## Current performance evidence

The v2 artifact and legacy v3 diagnostic are explicitly revoked in their manifests. Their source rows
contained higher-is-better `samples/minute` values, while the old preparation path labelled the
values as seconds and trained on `1000 / median(score)`. Consequently, all v2/v3 regression
metrics are invalid as model-quality or promotion evidence and must not be cited.

The corrected development artifact is
[`blender_cpu_content_creation_corrected_development_v1`](../artifacts/ml/performance/blender_cpu_content_creation_corrected_development_v1/training_report.json).
It preserves the median observed `samples_per_minute` value, uses the application route
`cpu/content_creation`, and contains 172 real hardware rows across 109 product-family leakage
groups with zero synthetic rows. Group-disjoint train, validation, calibration, and internal-test
splits contain 95 / 36 / 21 / 20 rows and 59 / 22 / 15 / 13 groups.

On the already-observed 20-row, 13-group internal split, the corrected LightGBM diagnostic has
R-squared **0.9408**, MAPE **22.23%**, and MAE 9.16 samples/minute. Its grouped-bootstrap 95%
intervals are 0.7173 to 0.9648 for R-squared and 13.59% to 32.48% for MAPE. These are development
diagnostics, not fresh holdout or promotion evidence: the source snapshot was adaptively explored
after v2/v3 results were seen, no untouched external frozen cohort exists, MAPE exceeds the 12%
gate, and both bootstrap gates fail. `promotion.eligible=false` and
`precise_predictions_enabled=false`; serving is relative-score-only. Model version:
`e4f5f153d4e42ed25a37cf91eaf95dd0dc968942e3843ff50d41cf61e40fab21`.

The separate corrected development-only
[`blender_cpu_content_creation_v3_development_diagnostic`](../artifacts/ml/performance/blender_cpu_content_creation_v3_development_diagnostic/diagnostic_report.json)
uses the same 172 rows / 109 groups, but protects 29 calibration rows / 19 groups and 30 holdout
rows / 15 groups from candidate selection. Among 113 development rows, its selected engineered
identity candidate has out-of-fold MAPE 19.34% and R-squared 0.9091, versus 21.98% / 0.8926 for
the base candidate: a 12.04% relative diagnostic MAPE improvement. The post-selection holdout is
R-squared 0.8763 and MAPE 18.98%; its grouped-bootstrap 95% bounds are 0.6279 to 0.9429 for
R-squared and 14.96% to 24.02% for MAPE.

This is not a qualifying model result: the source dataset was already adaptively explored, no
untouched external frozen cohort exists, MAPE exceeds 12%, and both bootstrap gates fail. Its
manifest sets `production_loadable=false`; it cannot be loaded by the production inference path.
Model version `e38a72f9528523e4f298e83f41edb7691b1af844fc2953089431022cfa15fe00`; exact report
SHA-256 `8128d8403ab2d3275ee026e2afb07df6c195f64597a55d74aac94de4a9bf3eb7`.

The sealed
[`blender_gpu_content_creation_optix_pilot_v1`](../artifacts/ml/performance/blender_gpu_content_creation_optix_pilot_v1/training_report.json)
uses Blender 4.5.0 / `junkshop` / GPU / OPTIX / Windows and the correctly oriented
`1000 / median_render_seconds` target. It contains 50 observed hardware rows across 36 families;
group-disjoint train / validation / calibration / test partitions contain 24 / 13 / 7 / 6 rows and
17 / 9 / 5 / 5 families. CPU-bounded LightGBM was materially worse than the median baseline on
the 6-row test partition: R-squared **-1.8538**, MAPE **203.52%**, and MAE 2.84 throughput units,
versus median MAPE 50.02%. The grouped-bootstrap R-squared interval is -3511.85 to 0.2894 and
MAPE interval is 67.92% to 393.37%.

This pilot is unambiguously non-promotable: the prepared manifest marks all rows ineligible for
external claims and only 36 families are available, below its 100-family credibility threshold;
the 6-row / 5-family test and 7-row / 5-family calibration partitions also fail minimum-size
gates. It has `promotion.eligible=false`, `precise_predictions_enabled=false`, low confidence,
and no serving route. Model version:
`fb2f42fccae0ac6965d28118fc26b5f5b811aa3fed8f6599fc4013a710a75192`.

The [temporal audit
report](../artifacts/evaluation/performance-temporal-v4/957139150dbc1bbdd6deab1810b966ea59a33d9768b206e5033287c88a3f044b.json)
has semantic digest `957139150dbc1bbdd6deab1810b966ea59a33d9768b206e5033287c88a3f044b`
and exact file SHA-256 `e675a0d1de5f0556aedeebb41c93f19f8130f3af22b3a010b7dc9c789fb458cf`.
Its frozen retrospective protocol has SHA-256
`6952cd7a220920fc9be882211577a67a67268a055e702e4b64955ada5123154a`. The newer snapshot is a
strict superset of the older one, but all 339 novel observations fail at least one full cohort
field, leaving 0 candidate rows and 0 leakage families. `evaluation=null`,
`model_inference_attempted=false`, and `supports_model_promotion=false`; the model status is
unchanged.

The full-catalogue retrieval diagnostic covers 32 silver queries and 102,664 query-candidate rows.
RRF produced NDCG@10 **0.308534**, compared with **0.193411** for BM25 on the same candidates.
Those labels were generated from the declared query predicates rather than human judgments, so
the artifact is non-promotable and no LambdaMART model was trained. It measures pipeline behavior,
not production ranking quality. See
[`retrieval-silver-full-v2/metrics.json`](../artifacts/evaluation/retrieval-silver-full-v2/metrics.json).

A separate synthetic development fixture exercised the performance pipeline and produced
LightGBM R-squared 0.9646 and MAPE 3.30%, versus Ridge R-squared 0.8737 / MAPE 6.25% and a median
baseline R-squared -0.0397 / MAPE 20.63%. These numbers prove the implementation can learn its
generated relationship; they provide no evidence of accuracy on real hardware and are ineligible
for promotion or public model-quality claims. The device probe recorded CPU fallback rather than
a GPU training result.

An external Zenodo Dn7 entity-resolution transfer run used source-record-disjoint splits of
15,951 / 1,945 / 1,802 pairs. CPU LightGBM held-out precision, recall, F1, and average precision
were 0.81102, 0.74638, 0.77736, and 0.81666 at the validation-F1 threshold. A validation threshold
selected for at least 0.99 precision produced test precision 1.0 and recall 0.05072. The model
therefore missed the combined gates. This is transfer-benchmark-only evidence, not PC-retailer or
Singapore-production quality. Report SHA-256:
`6f81289e73584f2894ea9ddf8380c4201140927b56584fa6203877981381a022`.

The blocking inputs for supervised promotion remain independently reviewed PC-domain duplicate
pairs and human 0-4 query-product relevance judgments. Production workload/value features and
priced build generation additionally require licensed Singapore retailer feeds with explicit
display, retention, derivation, embedding, and training rights plus known stock coverage.

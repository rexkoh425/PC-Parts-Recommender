# Real-catalog retrieval silver pilot

This pilot compares BM25, vector-only retrieval, and Reciprocal Rank Fusion on
the same frozen 3,000-product BuildCores catalog. It also includes ascending
product ID as a negative control.

The query definitions in `real_catalog_silver_queries.v1.json` cover all eight
component categories. Relevance grades are generated deterministically:

- grade 3: every declared `must` specification predicate passes;
- grade 4: every `must` and `excellent` predicate passes;
- grade 0: a required predicate fails or data is missing.

These are silver labels, not human relevance judgments. The run is diagnostic
and **not eligible for production, portfolio, or resume metric claims**. In
particular, LambdaMART is not trained on these qrels because doing so would
train and evaluate against labels generated from the same specification rules.

## Reproduce

First build the frozen SentenceTransformer embedding index. Then run:

```powershell
$catalog = Get-ChildItem data/processed/buildcores_open_db -Recurse -Filter records.jsonl |
  Where-Object FullName -Like '*portfolio-3000*' |
  Select-Object -First 1 -ExpandProperty FullName

uv run --no-sync python -m pc_build_recommender.evaluation.retrieval_silver_pilot `
  --catalog $catalog `
  --queries evals/retrieval/real_catalog_silver_queries.v1.json `
  --embedding-dir artifacts/retrieval/buildcores-embeddings `
  --output-dir artifacts/evaluation/retrieval-silver-pilot-v1 `
  --device cuda `
  --batch-size 128 `
  --rrf-k 60
```

`--no-sync` matters after `scripts/setup-gpu.ps1`: a normal dependency sync can
replace the separately installed CUDA PyTorch wheel.

The run emits:

- `frozen-candidates.json`: checksummed query candidate universes and qrels;
- `metrics.json`: aggregate, category, per-query, paired-bootstrap, provenance,
  and reportability metadata.

Some deliberately broad queries have more relevant products than a retrieval
cutoff can contain. The artifact therefore records the macro theoretical
Recall@20 and Recall@50 ceilings alongside the literal metrics.

## Human-labelled promotion path

Start by compiling a label-free capture of real, rights-cleared retrieval candidates with
`scripts/capture_relevance_annotation_candidates.py`, then
`scripts/prepare_relevance_annotation_batch.py --capture-manifest <capture/manifest.json>`; use
`relevance-annotation-candidates.template.json` as the input contract. The compiler rejects silver
qrels, synthetic rows, reviewer-bias fields, missing per-candidate provenance, and ineligible
source policies. Its `project-spec.json` and blinded `groups.jsonl` then feed the OIDC-bound
`scripts/manage_annotations.py` workflow. A policy that permits derived-model serving must carry
the exact public attribution notice a later release must expose. It creates a collection batch
only, not labels or measured model evidence.

The capture also creates a non-reviewer `prelabel-features.jsonl` and binds its exact feature
matrix and per-query row hashes. The batch compiler embeds only those opaque hashes in each task.
After the annotation project freezes, `training.materialize_ranking_snapshot` verifies that the
release preserved the commitment and appends adjudicated grades without recomputing features.
Human training and evaluation require the resulting labeled-snapshot manifest.

Production evaluation uses `human-relevance-labels.schema.json`. Each query
declares a `query_group_id` so paraphrases and closely related intents remain in
one train, validation, or test split. Every query-product pair needs two
independent reviewers. Exact agreement is accepted; any disagreement requires
an explicit decision from a third, independent adjudicator.

The Python contracts are exported from `pc_build_recommender.retrieval`:

- `HumanJudgmentSet.adjudicate()` validates coverage and produces a checksummed
  `FrozenCandidateSet`;
- `QueryGroupSplit.create()` freezes leakage-safe group assignments;
- `compare_ranked_models()` compares complete BM25, vector, RRF, and LambdaMART
  rankings with query-group bootstrap confidence intervals;
- `write_ranking_comparison_report()` writes a deterministic, hashed report;
- `evaluate_ranker_promotion()` applies the fail-closed promotion policy.

The default promotion gate requires at least 50 frozen test query groups,
pooled Recall@50 of at least 0.95, at least 15% relative NDCG@10 lift over BM25,
a paired NDCG confidence interval that excludes zero, and non-inferiority to
RRF within 0.01 NDCG. The ranker training metadata, pre-label snapshot, labeled dataset
manifest, judgment manifest, and query-group split hashes must match the evaluation report.

Silver, synthetic, legacy, unadjudicated, validation, and un-split evaluations
are hard-blocked from promotion regardless of their metric values.

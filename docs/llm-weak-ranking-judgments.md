# LLM weak labels for LambdaMART diagnostics

`training/llm_judge.py` accepts query-product tasks as JSONL and writes
checkpointed pair judgments plus a query-grouped ranking-training JSONL file.
The judge receives query requirements, actual product name/category/specifications,
known price and stock state, and caller-supplied authorized source evidence. Retrieval
scores, ranks, model identity, and candidate position are excluded from its prompt.
The rubric grades 0-4, records confidence/explanation/unknowns, and deterministically
sets explicit hard-requirement violations to grade 0. Unknown hard requirements cap
the grade at 2. Unknown stock or product facts must not be inferred.

Each input row must include:

```json
{
  "task_id": "intent-123:product-abc",
  "query_id": "query-123",
  "query_group_id": "intent-family-123",
  "context": {"query_id": "query-123", "query_group_id": "intent-family-123", "query_text": "..."},
  "candidate": {"product_id": "product-abc", "category": "gpu", "retrieval_scores": {}, "attributes": {}},
  "judge_input": {
    "query_text": "...", "category": "gpu", "requirements": [],
    "product": {"name": "...", "category": "gpu"},
    "specifications": {}, "price_sgd": null, "stock_status": "unknown",
    "source_evidence": []
  }
}
```

`query_group_id` must identify equivalent intents. The trainer now assigns all rows
in one intent group to the same train/validation/test split. LLM data uses the
explicit `llm_weak` provenance and requires `--allow-non-human-labels`; it remains
permanently outside the human-verified promotion route.

The provider must be an OpenAI-compatible `/v1/chat/completions` service configured
through `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `LLM_JUDGE_MODEL` (or
`CLIPROXY_CODEX_MODEL`). The command supports bounded concurrency, rate limiting,
retries, JSON validation, prompt/model-bound resume, token accounting, and optional
per-million-token cost rates. It never writes credentials. When no explicit price
rates are supplied, estimated dollar cost is null rather than zero.

Example:

```powershell
uv run --no-sync python -m training.llm_judge `
  --input artifacts/evaluation/llm-judge-ranking/pilot-v2/tasks.jsonl `
  --output-dir artifacts/evaluation/llm-judge-ranking/pilot-v2/judged `
  --concurrency 2 --requests-per-second 1
```

Generate the technical pilot with:

```powershell
uv run --no-sync python scripts/build_llm_judge_queue.py `
  --output artifacts/evaluation/llm-judge-ranking/pilot-v2 --device auto
```

It freezes 28 candidates per query, drawn from BM25, vector, RRF and seeded
category controls, and saves the actual retrieval features and source evidence.
It uses the existing 32 authored silver-diagnostic queries and their `must`
predicates as candidate hard requirements. No silver grades are loaded. The
queries have already appeared in experiments, and exact text deduplication is
not a semantic paraphrase audit: this is a **technical pilot only**, not an
independent test set. Audit the predicates and construct new, diverse intent
families before any final experiment. New query files may supply an explicit
`query_group_id` to keep related intents together. The saved pilot split matches
the diagnostic trainer's group split with `--seed 20261108` provided the entire
queue completes. Incomplete queues write no training rows; failed attempts are
recorded in `failures.jsonl`, and successful tasks resume from the checkpoint.

The gateway initially reported an invalid upstream refresh token. After the
user completed its visible login flow, `gpt-6-luna` returned structured responses.
All 896 pilot tasks completed; one failed attempt succeeded on resume. The
full-constraint audit corrected eight grades and found one severe raw contradiction.
A non-promotable pilot LambdaMART model was trained on the corrected labels.
The separate 16-pair second-model audit is a small stratified consistency check,
not human validation. Token totals are in the judgment summaries; no dollar rate
was supplied, so monetary cost is unknown.

After **all** pilot tasks succeed and the full-constraint audit has produced
the corrected pilot dataset, a diagnostic training command is:

```powershell
uv run --no-sync python -m training.train_ranking `
  --input artifacts/evaluation/llm-judge-ranking/pilot-v2/audited-training/ranking_dataset.llm_weak.jsonl `
  --artifact-dir artifacts/evaluation/llm-judge-ranking/pilot-v2/model `
  --candidate-set-version pilot-v2 --seed 20261108 `
  --label-provenance llm_weak --allow-non-human-labels
```

This would produce a non-promotable technical pilot only. Its already-used
queries cannot serve as the final holdout.

## Fresh synthetic diagnostic

`scripts/generate_llm_ranking_queries.py` creates authored specification requests;
it removes exact constraint matches with the old silver set and groups identical
constraint signatures together. The generated requests are templated rather than
sampled from users. Similar requests may remain even after exact grouping, so the
held-out metric is a weak-label development diagnostic, not evidence of human
preference or independence between real customer intents.

The `fresh-v2` run was stopped after 1,705 judgments: its numeric threshold
variants shared templates across its splits, so its held-out set is unsuitable
for a generalization claim. Preserve its checkpoint for reuse, but do not train
or evaluate from that split. Four exact matches to the earlier silver queries
had already been excluded. Its `reused-checkpoint.json` records 96 matches
recovered from the first superseded queue.

`generate_diverse_llm_intents.py` and `select_diverse_llm_intents.py` prepare a
replacement set using a different proposal model, catalogue-vocabulary checks,
and a conservative embedding similarity screen. Query generation and grouping
remain synthetic and are not human validated. The final queue must be frozen
and all judgments completed before the following commands are applicable.
The `fresh-v3` queue contains 284 queries, with 170 train, 57 validation and
56 test groups. Its 224 reused judgments have matching blinded judge inputs;
the transfer record preserves the source and new task hashes. When the original
judge became unavailable, the user selected a two-model fallback. The
`fresh-mixed-v1` queue keeps complete original-model groups, excludes two
partially judged groups, and preserves the original split assignments. Its
remaining 281 groups comprise 169 train, 57 validation and 55 test groups. Each
group has one assigned judge model, and assignments are not randomized. Report
results by judge as well as overall; the small original-model slice cannot
establish cross-model generalization.

To avoid judge assignment confounding the primary comparison, model training,
validation and primary test metrics use only `gpt-5.6-luna` groups: 161 train,
56 validation and 53 test groups. The two original-model test groups are a
separate, very small diagnostic slice. Assignment was driven by availability,
so the resulting query selection may still be biased and is not randomized.

The query-only audit found one request to retain DDR4 DIMMs whose structured
requirements omitted memory type. `constraint-overrides.json` freezes that
requirement before its query was judged. Combination records any resulting
grade correction and preserves the original responses. The separately frozen
feature snapshot does not include this later supplemental requirement.

Resume the remaining model's judgments with:

```powershell
$rankingRunDir = "artifacts/evaluation/llm-judge-ranking/fresh-mixed-v1"
uv run --no-sync python -m training.llm_judge `
  --input "$rankingRunDir/alternative-tasks.jsonl" `
  --output-dir "$rankingRunDir/alternative-judged" `
  --model gpt-5.6-luna --concurrency 12 --requests-per-second 4 `
  --max-retries 1 --max-consecutive-failures 4 `
  --schedule-snapshot "$rankingRunDir/queue/query-candidates.json"
```

The circuit breaker stops submitting new work after repeated provider failures;
completed judgments remain checkpointed. Resume only after the route recovers.
The optional schedule interleaves intent groups by category and split, without
changing the tasks or exposing retrieval scores to the judge.

After judging finishes successfully, attach the frozen label-independent
specification features, tune on train/validation, then evaluate the sealed test:

```powershell
$rankingRunDir = "artifacts/evaluation/llm-judge-ranking/fresh-mixed-v1"
uv run --no-sync python -m scripts.audit_mixed_llm_judgments `
  --root "$rankingRunDir" --output "$rankingRunDir/final-judgment-audit.json" --require-complete

uv run --no-sync python -m scripts.combine_mixed_judgments `
  --root "$rankingRunDir" `
  --alternative-summary "$rankingRunDir/alternative-judged/summary.json" `
  --alternative-checkpoint "$rankingRunDir/alternative-judged/judgments.checkpoint.jsonl" `
  --output "$rankingRunDir/combined"

uv run --no-sync python -m scripts.materialize_llm_ranking_features `
  --dataset "$rankingRunDir/combined/ranking_dataset.llm_weak.jsonl" `
  --summary "$rankingRunDir/combined/summary.json" `
  --features "$rankingRunDir/constraint-features-v1" `
  --output "$rankingRunDir/training-data"

uv run --no-sync python -m scripts.llm_ranking_experiment tune `
  --dataset "$rankingRunDir/training-data/ranking_dataset.llm_weak.jsonl" `
  --summary "$rankingRunDir/training-data/summary.json" `
  --snapshot "$rankingRunDir/queue/query-candidates.json" `
  --output "$rankingRunDir/experiment"

uv run --no-sync python -m scripts.llm_ranking_experiment test `
  --dataset "$rankingRunDir/training-data/ranking_dataset.llm_weak.jsonl" `
  --summary "$rankingRunDir/training-data/summary.json" `
  --snapshot "$rankingRunDir/queue/query-candidates.json" `
  --output "$rankingRunDir/experiment"
```

The feature snapshot is computed from frozen requirements and product fields;
it does not read labels or explanations. A hard
requirement whose value is unknown is not counted as verified satisfied. The
model-selection command compares three parameter settings and separate retrieval
and specification-feature ablations using validation groups. The test command checks the input hashes and
refuses to overwrite its report. All four rankers are evaluated on the same 28
judged candidates per query. Recall@50 is therefore conditional on that candidate
pool and cannot measure full-catalogue retrieval recall.

The experimental feature contract extends the existing builder with an explicit
unknown-specification fraction. Load its artifacts with
`LambdaMARTRanker.load(path, feature_builder=DiagnosticFeatureBuilder())`, where
`DiagnosticFeatureBuilder` is in `scripts.llm_ranking_experiment`. The production
feature contract and promotion gate are unchanged.

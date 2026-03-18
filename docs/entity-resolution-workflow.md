# Entity-resolution review and training workflow

Candidate discovery and supervision are separate. Blocking scores, numeric conflicts, and
"hard-negative" sampling reasons are annotation evidence only; they are never converted to
match labels.

## Controlled PC pilot

The implemented pilot accepts only the existing BuildCores OpenDB normalised catalogue and
the controlled Dynacore normalised offers. The adapter reads every row's source and usage
flags and derives one queue-level policy. Because the current Dynacore import is development-
only, the resulting queue can be reviewed but cannot be exported for model training or used
for published metrics.

```powershell
uv run --no-sync python -m training.entity_resolution_review create-controlled-queue `
  --catalogue data/processed/buildcores_open_db/<hash>/portfolio-3000/records.jsonl `
  --listings data/processed/dynacore_controlled_pdf/<hash>/records.jsonl `
  --created-at 2026-07-22T12:00:00+08:00 `
  --output artifacts/entity-resolution/review-queue.jsonl

uv run --no-sync python -m training.entity_resolution_review export-sheet `
  --queue artifacts/entity-resolution/review-queue.jsonl `
  --output artifacts/entity-resolution/review-sheet.csv
```

Reviewers may choose `MATCH`, `NON_MATCH`, or `UNCERTAIN`, or set the state to `SKIPPED` or
`INVALID`. Every completed action requires a reviewer ID and timezone-aware review timestamp.
The item snapshot hash prevents edited candidate data from being silently imported.

```powershell
uv run --no-sync python -m training.entity_resolution_review import-sheet `
  --queue artifacts/entity-resolution/review-queue.jsonl `
  --sheet artifacts/entity-resolution/review-sheet.csv `
  --output artifacts/entity-resolution/reviewed-queue.jsonl
```

Repeated identical imports are idempotent. Attempts to rewrite a completed decision fail.
`UNCERTAIN`, blank, skipped, and invalid rows never become binary training examples.

## Active learning

Active-learning input is JSONL containing `queue_item_id` and `probability` from a versioned
model. Sampling prioritises uncertainty, proximity to the precision-first operating boundary,
conflict/model disagreement, and category diversity. Its output remains explicitly unlabeled.

```powershell
uv run --no-sync python -m training.entity_resolution_review sample-active `
  --queue artifacts/entity-resolution/review-queue.jsonl `
  --scores artifacts/entity-resolution/candidate-scores.jsonl `
  --model-version er-model-v1 --limit 100 `
  --output artifacts/entity-resolution/active-batch.json
```

## Training gate

<!-- TODO: sections below still to be written. -->

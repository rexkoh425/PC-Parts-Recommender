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

Training is available only for a queue whose stored source policy permits it. The dedicated
command splits by listing ID into training, calibration, threshold-selection, and test groups.
Calibration and threshold selection are separate. The threshold maximises validation recall
subject to at least 99% precision and minimum support; failure to meet the gate produces only a
diagnostic artifact. A held-out grouped test precision gate is required before the artifact is
placed under `deployable/`.

```powershell
uv run --no-sync python -m training.train_entity_resolution_human `
  --review-queue artifacts/entity-resolution/reviewed-authorised-queue.jsonl `
  --artifact-dir artifacts/models/entity-resolution-human `
  --minimum-precision 0.99 --minimum-predicted-matches 25 `
  --max-host-used-gb 55 --minimum-free-memory-mb 1024
```

The report stores queue and candidate hashes, source-use policy, listing-group hashes,
calibration evidence, operating-point selection, and grouped test results. Test fixtures and
external transfer benchmarks must remain clearly separated from PC-domain human-label claims. The
human-label trainer and the transfer benchmark both preflight materialised input against the 55 GiB
host-RAM cap before parsing it; their reports preserve the observed memory snapshot and conservative
allocation estimate.

# Catalog ingestion and production gate

The processed-catalog importer is conservative by design. It accepts normalized
BuildCores product records and controlled retailer offer records, preserves their
provenance, and evaluates deterministic GTIN and brand-plus-MPN anchors before any
learned matcher. Numeric and colour variant conflicts always reject the affected pair.
Unknown stock remains unknown.

## Read-only coverage report

```powershell
.venv/Scripts/python.exe scripts/import_processed_catalog.py `
  --buildcores <buildcores-records.jsonl> `
  --offers <governed-offers-records.jsonl> `
  --report-output artifacts/evaluation/catalog-readiness-current.json `
  --decisions-output artifacts/evaluation/catalog-mapping-decisions-current.json
```

The importer streams JSONL records with an 8 MiB per-line limit and retains only a
compact identity index needed for entity resolution. `--max-line-bytes` and
`--batch-size` are configurable.

To persist records, add `--database-url <sqlalchemy-url>`. Add
`--require-production-ready` for a fail-closed import. A failed production gate exits
with code 2 and rolls back the import transaction.

The standalone processed importer is a development and readiness-report surface. Production uses
`scripts/import_catalog_release.py`, which has no switch to weaken its gates. It first validates
the operator-pinned serving manifest, derives the embedding artifact location from that manifest,
cross-checks the vector matrix, ID map, search documents, catalogue bytes, and embedding identity,
and recomputes the exact processed snapshot and readiness report before opening the database.
It then:

1. rejects any canonical product or retailer-listing row outside the pinned release;
2. upserts the processed catalogue;
3. imports the pinned vectors and search-document hashes without deleting stale provenance; and
4. verifies the exact product set, listing set, canonical rows, listing rows, and search-document
   identities before succeeding.

There is intentionally no automatic stale-row cleanup. A stale product, listing, or provenance
blocks deployment and requires a separately reviewed reconciliation design with an explicit
target set, backup, audit output, and rollback procedure. That destructive workflow is not
implemented by the release command.

The default production policy requires:

- at least 750 products and every required component category;
- at least 90% completeness for every compatibility-critical field group;
- at least 80% controlled-offer mapping coverage;
- priced and known-in-stock coverage for every required category; and
- complete product, offer, and persisted-listing provenance.

It also requires a production-authorized entity-resolution runtime. A production import accepts
that authority only through `--serving-manifest` plus the operator-pinned
`--serving-manifest-sha256`; direct model, evaluation, or shadow-mode arguments are rejected.
The version-3 serving manifest binds the exact LightGBM model, embedded calibrator, serving
evidence, human-labelled v2 evaluation, matcher/catalogue policy, rights approval, governed
offers, reviewed mappings, and review-evidence JSONL. The policy supplies the deployed thresholds,
while the rights approval binds the exact model release, evaluation bytes, review queue, frozen
test groups, and permitted SG serving uses. Their composite release identity becomes part of the processed
catalogue data version, so the release job and API bootstrap cannot silently use different ER
decisions.

Current synthetic and external-transfer artifacts do not satisfy this contract and cannot be
activated. No promoted production ER release is shipped by the repository.
`--entity-resolution-model`, `--entity-resolution-evaluation`, and
`--allow-unpromoted-entity-resolution-shadow` remain development/diagnostic surfaces only; shadow
mode is limited to qualified human
diagnostics: model results can enter the manual-review queue, but they cannot create canonical
mappings. Every decision retains its method, candidate IDs, probability basis, model version,
thresholds, margin, bounded feature evidence, and release identity where applicable. Exact anchors
use `probability=null` because they are deterministic rules, not calibrated model certainty.

## Review-evidence artifact

`--review-evidence <review-evidence.jsonl>` is optional for a development import and required for a
production import. It is not a scraping interface. Each line has the exact envelope fields
`schema_version`, `record_type`, `data`, `data_use_rights`, and `provenance`, with schema
`pc-build-recommender.review-evidence.v1` and record type `review_evidence`.

`data` contains a unique evidence ID, canonical product ID, supported aspect, sentiment from -1 to
1, confidence, a cited statement of at most 500 characters, a credential-free HTTPS source URL,
and an optional timezone-aware publication time. `provenance` repeats that source URL and records
the source name, retrieval time, raw SHA-256, parser version, and licence/access note. The loader
requires active Singapore display, cache, history, and derivation rights, rejects expired or stale
rights, and requires the product to exist in the frozen catalogue. If no source is permitted,
release an explicit empty JSONL file; the API will return no review claims.

## Manual mapping review

Generate an unresolved queue:

```powershell
.venv/Scripts/python.exe scripts/review_catalog_mappings.py queue `
  --buildcores <buildcores-records.jsonl> `
  --offers <governed-offers-records.jsonl> `
  --reviewed-mappings <reviewed-mappings.json> `
  --output <review-queue.json>
```

Approve a mapping only after checking source evidence:

```powershell
.venv/Scripts/python.exe scripts/review_catalog_mappings.py approve `
  --manifest <reviewed-mappings.json> `
  --buildcores <buildcores-records.jsonl> `
  --offers <governed-offers-records.jsonl> `
  --listing-id <listing-id> `
  --product-id <product-id> `
  --reviewed-by <reviewer> `
  --evidence <evidence>
```

The approval command independently rechecks category, numeric-variant, and colour
conflicts. A reviewed no-match is persisted with `reject` and suppresses future
automatic matching. Manifest replacement is atomic and records reviewer, timestamp,
and evidence.

## Current measured snapshot

The saved report at
`artifacts/evaluation/catalog-readiness-current.json` is for data version
`processed-66286ad2cb30278c`. It measures 3,000 canonical products and 485 controlled
offers. Only 2 offers are safely auto-mapped (one GPU and one power supply); 483 remain
unmatched, all 485 retain complete offer provenance, and no offer asserts known stock.
The legacy normalized rows predate per-record rights objects (0 of 485 explicit); the
source registry therefore bars every use until written permission and regenerated records
exist. This snapshot is intentionally **not production-ready**. Exact field-level
counts and all gate blockers are stored in the report rather than described as achieved
metrics.

## Contracted retailer feeds

`retailer_csv` imports require `--retailer-policy-json`; loose command-line consent flags
are deliberately unsupported. A policy has this fail-closed shape:

```json
{
  "retailer": "Contracted Retailer",
  "feed_id": "contracted_retailer_sg",
  "source_url": "sftp://retailer.example/feed.csv",
  "licence_or_access_note": "Use governed by contract RET-2026-001.",
  "training_eligible": false,
  "published_claims_eligible": false,
  "allow_non_new": false,
  "rights": {
    "contract_reference": "RET-2026-001",
    "contract_version_url": "contract://RET-2026-001/v1",
    "consent_effective_on": "2026-07-22",
    "consent_expires_on": null,
    "retention_days": 365,
    "deletion_required_on_termination": true,
    "deletion_sla_days": 30,
    "territories": ["SG"],
    "may_display": true,
    "may_cache": true,
    "may_store_history": true,
    "may_redistribute": false,
    "may_embed": false,
    "may_train": false,
    "may_derive": true
  }
}
```

This adapter needs display, cache, price-history, and derivation rights because those are
intrinsic to ingestion and recommendation serving. Embedding and training remain separate
grants: setting `training_eligible=true` without `may_train=true` is rejected. The rights
object is copied into every normalized record so downstream embedding or training jobs can
call the fail-closed `require_data_use` gate. Expired or not-yet-effective consent, missing
territory, invalid retention, and incomplete termination-deletion terms are rejected before
the source snapshot is accepted. The serving importer also requires `provenance.retrieved_at`,
both listing observation timestamps, and `price_snapshot.observed_at` to be at or after the
`consent_effective_on` UTC day boundary. An active agreement therefore cannot make a
pre-consent observation eligible for serving.

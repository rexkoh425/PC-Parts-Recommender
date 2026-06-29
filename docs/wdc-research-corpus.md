# WDC Products research-corpus quarantine

This import path is an engineering and data-discovery aid. It does **not** add retailer offers to
the production catalogue.

The source is the historical [WDC Products / PDC2020-C corpus](https://webdatacommons.org/largescaleproductcorpus/wdc-products/),
which was extracted from multi-market web pages in 2020. The source page reports 11,715 offers and
2,162 entities for the smaller WDC Products matching benchmark. PDC2020-C is much larger, but its
separate `Computers_and_Accessories` category is broad and was assigned by a categorization model;
it is not a desktop-component catalogue or a Singapore-retailer feed.

## Rights and release boundary

The project has not established a licence or written contract granting the downstream production
uses required by BuildSignal. The importer therefore emits only a quarantined research schema and
sets these gates explicitly:

- immutable internal research cache: allowed for at most 365 days;
- display, price-history retention, redistribution, embeddings, training, derivation: denied;
- production catalogue and Singapore-market evidence: denied;
- current price, shipping, and stock evidence: denied;
- production entity-resolution or model-metric claims: denied.

The original row remains recoverable from its content-addressed raw snapshot. Each selected record
retains the snapshot SHA-256, source URL, historical page URL, source-line number, raw-line SHA-256,
retrieval time, parser version, and access note. Historical prices remain strings and are labelled
as 2020 observations; the importer never creates a `RetailerListing` or `PriceSample`.

## Run it

Download the two official files manually, or pass `--download` explicitly. A network download is
never implicit because the corpus is over 5 GB compressed.

```powershell
uv run --no-sync python -m scripts.import_wdc_research_corpus `
  --corpus C:\data\wdcproducts_corpus_with_url.json.gz `
  --categories C:\data\WDC_Corpus_LargeScaleExperiment_MajorityVoting.json.gz `
  --raw-root data\raw `
  --output-root data\quarantine `
  --category-index data\quarantine\wdc-products-category-index.sqlite3
```

The default invocation streams each file to EOF while checkpointing every 10,000 rows. If it is
interrupted, repeat the identical command to resume. For a deliberately bounded smoke run, pass
`--category-record-budget` or `--corpus-record-budget`; reaching a budget returns exit code `3`
and a JSON `*_paused` status.

The category mapping is stored in SQLite instead of a Python dictionary. A resumed gzip scan must
replay decompression to its checkpoint line, so resumption remains memory bounded but is not
constant-time. Completed artifacts are content-addressed by the corpus hash, category hash, and
selection-policy hash. A repeated completed import verifies the output hash and reuses it.

## Exact resource ceilings

| Resource | Default or hard limit |
|---|---:|
| Corpus HTTP response | 6 GiB compressed |
| Category HTTP response | 256 MiB compressed |
| Corpus decompression | 64 GiB streamed |
| Category decompression | 4 GiB streamed |
| One JSONL line | 2 MiB |
| Selected records | 100,000 default; 250,000 hard ceiling |
| Normalized output | 1 GiB |
| Description retained per selected row | 4,096 characters |
| Checkpoint interval | 10,000 examined rows |

The parser holds one bounded source line, one normalized row, and at most 250,000 short selected
offer IDs in memory. It does not use pandas and never materializes the decompressed corpus. These
limits keep the importer far below the project's 55 GB system-memory ceiling; normal operation is
expected to remain below 1 GB of importer working memory. Disk capacity must still cover the raw
compressed snapshots, the SQLite category index, and the quarantined output.

The parser refuses to use a snapshot or category index after its 365-day internal retention
deadline. `scripts/maintain_wdc_research_retention.py` now deletes expired raw snapshots, the
category index, paused working runs, and sealed artifacts only after validating their immutable
receipts, content hashes, schemas, and containment below the exact WDC roots. It preserves unknown
files and fails closed before deleting anything if a recognised working run is malformed. Dagster's
default-running `wdc_research_retention_daily` schedule executes the same bounded destructive engine
at 00:15 Asia/Singapore time. Run a dry-run before a manual or recovery invocation:

```powershell
uv run --no-sync python scripts/maintain_wdc_research_retention.py `
  --raw-root data/raw `
  --output-root data/quarantine `
  --category-index data/quarantine/wdc-products-category-index.sqlite3 `
  --dry-run
```

The command has a global `--maximum-entries` bound and supports a timezone-aware `--now` only for
auditable testing or recovery. Its report must be retained with operational logs; an invalid receipt
or unknown entry inside a recognised working run requires operator investigation rather than broad
cleanup.

## Candidate selection

Selection requires both:

1. the WDC majority-voted broad category `Computers_and_Accessories`; and
2. exactly one transparent text-rule match for CPU, GPU, motherboard, memory, storage, power
   supply, CPU cooler, or case.

Zero-match rows and ambiguous bundles are counted and excluded. UPS products are explicitly
excluded from the power-supply rule. These labels are candidate-discovery hints, not ground truth,
canonical product mappings, compatibility specifications, or training labels.

## Evidence this path can and cannot produce

It can support bounded-ingestion tests, provenance audits, historical schema exploration, and a
research-only queue of likely desktop-component records.

It cannot substantiate the résumé claim that 10,000 current retailer listings were mapped to 3,000
canonical products. It also cannot substantiate Singapore prices or stock, human-adjudicated entity
resolution accuracy, compatibility facts, performance-model accuracy, ranking lift, or production
recommendation quality. Those require rights-cleared live sources, reviewed labels, frozen
evaluations, and separately promoted artifacts.

# Data sources and ingestion policy

The ingestion layer is provenance-first: every source response is stored once under its
SHA-256 digest, accompanied by immutable metadata, then normalised into deterministic JSONL.
Parquet is also written when `pyarrow` is installed. A processed manifest and data-quality
report are emitted beside every dataset.

The registry of record is [`data/source_registry.yaml`](../data/source_registry.yaml).
The Zenodo transfer benchmark described below is registered only for bounded external
evaluation; it is not a scheduled production-ingestion or PC-domain training source.

## Sources implemented now

| Source | Intended use | Licence/access | Model use |
|---|---|---|---|
| [BuildCores OpenDB](https://github.com/buildcores/buildcores-open-db) | Eight-category canonical component catalogue | ODC-By 1.0; pinned Git commit; attribution required | Eligible, but hard-compatibility fields still require manufacturer verification |
| [Blender Open Data](https://opendata.blender.org/) | CPU/GPU content-creation observations | CC0 1.0; content-hashed snapshot | Eligible when split by product family and normalised by version, scene, backend, and OS |
| [MLPerf Inference v6.0](https://github.com/mlcommons/inference_results_v6.0) | Local-AI inference observations | Apache 2.0; pinned Git commit | Eligible; only flagged single-node/single-accelerator rows may be attributed to one component |
| [Zenodo ER transfer benchmark](https://doi.org/10.5281/zenodo.8164151) | External entity-resolution transfer test | CC BY 4.0; creator George Papadakis; version 3 | Eligible for a labelled transfer diagnostic, but not as Singapore retailer production evidence |
| [Wikidata](https://www.wikidata.org/wiki/Help:Data_access) | Candidate-driven aliases, identifiers, release dates, and entity links | CC0 1.0 for official API responses; content-hashed bounded response | Training-ineligible and quarantined until a reviewed downstream consumer and integration evaluation exist; local fixtures also remain redistribution-ineligible |
| [PCI ID Repository](https://pci-ids.ucw.cz/) | PCI vendor, device, and subsystem labels for candidate blocking | BSD 3-Clause option; daily content-hashed snapshot; attribution required | Deterministic blocking enrichment only; not training labels, compatibility truth, prices, or stock |
| Governed web product crawler | Exact, reviewed product pages exposing Schema.org `Product` and `Offer` JSON-LD | Per-source policy, terms hash, robots check, and separate acquisition/data-use authority | Ineligible by default; production use requires explicit SG display, cache, history, and derivation rights |
| [Awin local product feed](awin-local-feed.md) | Operator-supplied Awin CSV or gzip file already downloaded under an authorised account | Detached Ed25519-signed policy, independently pinned trust root, and an explicit agreement are mandatory | Ineligible by default; no feed, key, trust root, or real grant is bundled |
| Consented retailer CSV | Authorised retailer listings and prices | Per-feed consent reference is mandatory | Ineligible by default; enable only when the agreement explicitly allows it |
| [Dynacore PDF](https://dynacoretech.com/pages/price-list) | Local SGD development seed | No open data licence established; controlled local import | Never eligible for training, redistribution, or published metric claims without permission |
| [Bizgram price list](https://www.bizgram.com/pricelist-download/) | Quarantined local parser for SGD price-list research | No open data licence or written downstream-use permission established; exact local PDF only | All display, cache, history, redistribution, embedding, derivation, training, and claims rights are false |

BuildCores does not contain retailer prices. The first production Singapore price sources
should therefore be signed CSV/JSON feeds from consenting retailers. Lazada's Open Platform
can serve a seller's own authorised catalogue, but it is not a public marketplace-search feed.
Wikidata also contains no retailer-price feed in this implementation. Its adapter searches a
maximum of 100 candidates by default, uses the official part-number property P13802 and GTIN
property P3962, respects `maxlag=5` and a five-request-per-second ceiling, and records weak or
ambiguous matches as rejections. Parser v3 requires exact part-number matches to have a reviewed
P176 manufacturer agreement and, where the category has a reviewed class, matching P31 type
evidence; GTIN remains global. Name-only matching additionally requires the current reviewed
CPU-model instance class and brand context. Every output is marked development-only and
training-ineligible until the downstream catalogue join, conflict policy, provenance propagation,
and integration evaluation are reviewed. Local response fixtures are also ineligible for
redistribution.

The Awin adapter is an authorization and parsing boundary, not a feed downloader. It accepts only
an operator-supplied local CSV or gzip file and has no URL, API-key, or network-fetch argument. The
operator must separately obtain an Awin publisher account/feed access, merchant approval, and
written rights for every intended use; download the feed outside this application; create and
Ed25519-sign the exact policy; protect the signing key; and distribute the trust-root SHA-256 over
an independent trusted channel. See Awin's [feed-list/download
documentation](https://help.awin.com/developers/docs/product-feed-list-download) for the external
acquisition surface. Credential-bearing download URLs must never be stored as source provenance.
No real Awin rows are counted in this document.

The Bizgram adapter has no downloader or scheduled-fetch path. It accepts only the reviewed
SHA-256 fingerprint, validates all nine page-layout anchors, processes one page at a time, and
requires a dotted leader plus one terminal numeric SGD price. Page, line, archive, and raw-row
hashes are retained; stock is always `UNKNOWN`. Rows outside the high-confidence component
subset, bundle matrices, ambiguous variants, accessories, complete systems, and malformed prices
enter a bounded rejection queue. The accepted records remain unmatched, development-only, and
fail the production-rights gate. No Bizgram PDF or normalised Bizgram dataset is checked into the
repository or included in the measured inventory below.

The PCI ID adapter selects the upstream BSD 3-Clause option and retains the notice in
[`docs/third-party/pci-id-repository-BSD-3-Clause.txt`](third-party/pci-id-repository-BSD-3-Clause.txt).
It prefers the official compressed daily snapshot, sends a named BuildSignal User-Agent, stores
the raw bytes by SHA-256, and defaults to a 20,000-record cap. Parser v2 scans the complete
bounded snapshot, guarantees available AMD, NVIDIA, Realtek, and Intel vendor anchors, then uses
deterministic stratified hash reservoirs instead of a lexicographic prefix. Parsing is
line-streamed with decompressed-byte, line-length, line-count, candidate-retention, and
rejection-retention budgets. Its output is explicitly non-authoritative for products,
compatibility, performance, price, and stock.

## Measured local inventory through 2026-07-23

These are completed ingestion results, not roadmap targets.

| Dataset | Accepted | Review/rejected | Data-quality result | Normalised records SHA-256 |
|---|---:|---:|---|---|
| BuildCores fast slice | 800 | 0 | PASS | `200614af4d34f49ec78dfb213a07fee9b92ae2e971278b04560b9d64ca0c5d75` |
| BuildCores portfolio slice | 3,000 | 0 | PASS | `4675d5f816eb95de6c6e627f1120f49a2630baf5bac5c8e8ca47a6112af292dc` |
| Blender bounded sample | 3,000 | 178 rejected | PASS at the documented 10% cap | `354a98eafd15da8edc46dc33e554ac542edfcc7c901cdfdbb5e5ead6a60dbbc3` |
| MLPerf v6 summary | 520 | 0 | PASS | `547aee49784f7d60f8e03e1ce2425bcc02313af2c5b114a142a0313757b4129d` |
| Dynacore controlled import | 485 | 69 review, 68 hard rejected | PASS at the controlled-import 30% cap | `64aa01ed7c2da3f667f8e9cf149594bd41054a0c73ddfa9fa633dd78ea38141d` |
| PCI ID Repository bounded snapshot v2 | 20,000 | 0 | PASS; full bounded scan retained AMD, NVIDIA, Realtek, and Intel anchors | `d85e3653cbcf0296394149bdafe6089bd6a289b53b8258c02490eadca4034cd1` |
| Wikidata curated CPU identity sample v2 | 0 | 4 rejected | FAIL; conservative identity evidence was insufficient | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Wikidata bounded CPU identity refresh | 7 | 93 rejected | FAIL at the documented 80% rejection cap; remains quarantined and non-promotable | `1476e358089f2c17dfe2c59272c3c97b4245a251f58b64255fb612cacd823eae` |
| Dynacore governed web research pilot v2 (historical) | 6 | 0 | Historical PASS; superseded by parser v3 and quarantined from release and ML | `ce083eaf083f51a17cc510e7396357744a57d2497ad9187a44d54c462910923a` |
| Dynacore governed web research pilot v3 | 6 | 0 | PASS; internal-research only and quarantined from release and ML | `d6762f658ee75c5056b47b3dbc20ebd5a19409e877be5280ccc8158844efc4d57` |

The BuildCores portfolio distribution is CPU 250, GPU 350, motherboard 650, memory 450,
storage 450, power supply 350, CPU cooler 250, and case 250. The pinned archive contains
25,699 records across those eight categories, but only the measured 3,000-row slice is claimed
here.

The Blender sample contains 1,833 CPU, 683 CUDA, and 484 OptiX observations. It is a bounded
stream-head engineering sample, not a temporal-representativeness claim. The complete snapshot
should be sampled across time before final model training.

A separate 2026-07-23 temporal audit compared raw Blender snapshots
`c0f9d35c20807776138b0590097177b8ef2172119cc19aae8d1bad1b55af4833` and
`67582ebca9ead706b0c8d6cc96726bfa25c748664f95e10ad994a5fc81e493f5`. The latter is a strict
superset with 113 novel submissions and 339 novel observations, but zero observations matched the
complete frozen Blender 4.0.0 / `junkshop` / CPU / Windows / build-hash / benchmark-script /
scene-checksum contract. No catalogue join, model inference, metric calculation, pooling, or
retraining was attempted. The status is `insufficient_external_cohort`, so the corrected model
remains non-promotable. Evidence:

- protocol `evals/performance/blender_cpu_content_creation_v4_external_protocol.json`, SHA-256
  `6952cd7a220920fc9be882211577a67a67268a055e702e4b64955ada5123154a`;
- report
  `artifacts/evaluation/performance-temporal-v4/957139150dbc1bbdd6deab1810b966ea59a33d9768b206e5033287c88a3f044b.json`,
  semantic digest `957139150dbc1bbdd6deab1810b966ea59a33d9768b206e5033287c88a3f044b`,
  file SHA-256 `e675a0d1de5f0556aedeebb41c93f19f8130f3af22b3a010b7dc9c789fb458cf`.

MLPerf contributes 520 system observations; 22 are explicitly flagged as component-model
eligible. The remaining rows must retain node and accelerator counts and must not be presented
as measurements of a single consumer GPU.

The Dynacore source is fingerprinted as
`6e243d7bf1cba090f529b09a9276fac03fedddcadb8c11cf9ce7ec1e674bb9ba`.
The parser used the born-digital text layer, not OCR. It accepted 158 GPUs, 76 memory variants,
16 storage variants, 110 cases, 68 power supplies, and 57 coolers. It quarantined all 67
`#REF!` cells, 29 ambiguous multi-product or inline-price rows, 40 nonnumeric/ambiguous prices,
and one duplicate offer. Stock is always `UNKNOWN`.

The separate governed-web pilot fetched six exact Dynacore product URLs on 2026-07-22 after
checking the live robots file and a v2 canonical digest of the reviewed terms wording, links,
and relevant semantic attributes both before and after retrieval. It accepted one CPU, GPU,
memory kit, storage device, power supply, and cooler. The final hardened crawl-run hash is
`b3a8eb8855fcf5f69ae54940174f20ba561712e82765ff13429cba624c122b26`, its policy fingerprint
is `bbea0b1b19345f65f851ee4e5e4a61d08c9d6b76552901a7fbb33d80bbb51332`, and its immutable
processed-retention receipt has SHA-256
`27758feb372d3cad9a71ec032482a09e6b863cfef4e08578b101e06557097bf7`.
This is processing evidence only: every downstream data-use right is false, unknown shipping is
labelled rather than inferred, and the rows are marked development-only, training-ineligible,
and published-claims-ineligible. Raw pilot snapshots must be deleted by 2026-07-29. They must
not feed the public catalogue, price history, ranking, optimisation, embeddings, or models.
These retained results were produced by the v2 product-offer parser and are historical only.
Parser v3 adds crawl-start authority revalidation, seller-scoped listing identities, declared
currency consistency, strict `doesNotShip` handling, and stronger policy types. A prior local-run
note reported a bounded v3 crawl and retention dry-run, but its redistributable receipt is not
retained in this repository. It is therefore not a reproducible project claim and must not be used
as evidence of coverage, quality, or rights. If this pilot is rerun, commit a safe bounded
evidence receipt (not raw offer data) before documenting its hashes or counts. This does not
change the rights decision: any governed-web rows remain development-only,
training-ineligible, and published-claims-ineligible unless a separate authorised production
rights path is established.

The historical four-CPU Wikidata v2 probe matched no rows after correcting the part-number property
from P1628 to P13802 and requiring reviewed CPU-class and manufacturer evidence for name-only
matches. The current 2026-07-23 official-API CPU refresh found seven exact-name identity matches
out of 100 bounded candidates, while 93 candidates were rejected as weak, ambiguous, or unmatched.
It fails the documented 80% rejection cap and is intentionally retained as non-promotable coverage
evidence. Wikidata remains supplemental, standalone identity research and is not compatibility,
model, or market-price evidence.

The Zenodo transfer archive is stored at `tmp/er-benchmark/Dn7.zip`, 5,473,577 bytes, SHA-256
`3e3fd6951ab4c4ed6aa741c2594d3ab496b63aeca6b41b8a1e639bc6d9895980`. The deposited source
declares two tables with 2,554 and 22,074 records plus 43,418 candidate pairs, including 763
positives. A guarded materialization assigned source records disjointly and retained 15,951 train
pairs (481 positives), 1,945 validation pairs (144 positives), and 1,802 test pairs (138
positives); cross-split negative candidates were discarded.

CPU LightGBM achieved test precision 0.81102, recall 0.74638, F1 0.77736, and average precision
0.81666 at the threshold selected for validation F1. Selecting a validation threshold with at
least 0.99 precision produced test precision 1.0 but recall 0.05072. The targets were not met.
The report at `artifacts/models/er-transfer-dn7/transfer_benchmark_report.json` has SHA-256
`6f81289e73584f2894ea9ddf8380c4201140927b56584fa6203877981381a022`.

This remains external transfer evidence only: it is not a PC-component retailer dataset and
cannot establish Singapore production quality. The parser also encountered 12 nonblank
`shipweight` values written with embedded spaces, such as `1 206`; it preserves the source text
but treats the numeric value as missing rather than inventing thousands semantics.

## Reproduce the measured and bounded source datasets

The commands below reuse downloaded inputs already present in `tmp/` or the content-addressed raw
store. Omit the local file argument for BuildCores, Blender, or MLPerf to fetch the registered
remote source instead. The measured PCI row is reproducible only from the retained raw snapshot
with SHA-256 `47eb772eadf80fbee0459294b7cffbc8c740a327a59c71aeca0a1b927a038c43`; fetching the
mutable daily upstream snapshot instead exercises the same bounded parser but cannot reproduce the
historical normalised-record hash in the table.

```powershell
uv run --no-sync python scripts/fetch_open_data.py --source buildcores `
  --buildcores-profile portfolio `
  --buildcores-archive tmp/source-inspection/buildcores-6a64ab14.zip

uv run --no-sync python scripts/fetch_open_data.py --source blender `
  --blender-archive tmp/source-inspection/blender-latest.zip `
  --blender-limit 3000 --blender-scan-limit 10000

uv run --no-sync python scripts/fetch_open_data.py --source mlperf `
  --mlperf-summary tmp/source-inspection/mlperf-v6-summary.json

uv run --no-sync python scripts/fetch_open_data.py --source pci_ids `
  --pci-ids-snapshot `
    data/raw/pci_id_repository/47eb772eadf80fbee0459294b7cffbc8c740a327a59c71aeca0a1b927a038c43.ids.gz `
  --pci-ids-format gzip `
  --pci-ids-sha256 47eb772eadf80fbee0459294b7cffbc8c740a327a59c71aeca0a1b927a038c43 `
  --pci-ids-record-limit 20000

uv run --no-sync python scripts/fetch_open_data.py --source wikidata `
  --wikidata-candidates data/processed/buildcores_open_db/<raw-hash>/portfolio-3000/records.jsonl `
  --wikidata-category cpu `
  --wikidata-limit 100

uv run --no-sync python scripts/fetch_open_data.py --source web_product `
  --web-policy-json path/to/reviewed-web-policy.json `
  --web-url https://approved.example/products/exact-reviewed-product

uv run --no-sync python scripts/fetch_open_data.py --source dynacore `
  --dynacore-pdf tmp/pdfs/dynacore-2026-07-17.pdf

# Explicit local-file quarantine only; never part of the default or open-source run.
uv run --no-sync python scripts/fetch_open_data.py --source bizgram `
  --bizgram-pdf path/to/reviewed-bizgram-2026-07-21.pdf

# Local-only authorized Awin import; download and credentials remain outside the app.
uv run --no-sync python scripts/fetch_open_data.py --source awin_feed `
  --awin-feed C:\secure\feeds\merchant.csv.gz `
  --awin-policy-json C:\secure\policies\merchant-policy.json `
  --awin-policy-signature C:\secure\policies\merchant-policy.sig.json `
  --awin-trust-root C:\secure\policies\trust-root.json `
  --awin-trust-root-sha256 REPLACE_WITH_64_HEX_SHA256
```

The principal output directories are:

- `data/processed/buildcores_open_db/<raw-hash>/portfolio-3000/`
- `data/processed/blender_open_data/<raw-hash>/head-3000/`
- `data/processed/mlperf_inference_v6/<raw-hash>/`
- `data/processed/pci_id_repository/<raw-hash>/limit_20000/`
- `data/processed/wikidata_cc0/<raw-hash>/en-limit-100/`
- `data/processed/<governed-web-source>/<crawl-run-hash>/`
- `data/processed/awin_<advertiser-id>_<feed-id>/<processed-run-hash>/`
- `data/processed/dynacore_controlled_pdf/<raw-hash>/`
- `data/processed/bizgram_controlled_pdf/<raw-hash>/` (local quarantine only)

Each contains `records.jsonl`, `rejections.jsonl`, `manifest.json`, and `data-quality.json`.
Raw payloads and processed artifacts are intentionally gitignored; manifests and hashes should
be copied into versioned evaluation artifacts only when a release dataset is frozen.

Before a new run can pass, ingestion compares it with the newest valid `PASS`
quality report for the same source and processing variant. The comparison is
aggregate-only and blocks material count, category, record-type, and
rejection-rate regressions; it does not confer source rights or substitute for
field-level validation. The first passing run establishes the comparable
baseline.

The PCI ID and Wikidata artifacts are standalone enrichment datasets. No production catalogue,
retrieval, entity-resolution, ranking, or model path consumes them yet. Adding a reviewed join,
conflict policy, provenance propagation, and integration evaluation is required before either can
be described as improving serving results.

Use `uv run --no-sync` after `scripts/setup-gpu.ps1`; an ordinary synchronized `uv run` may
restore the CPU PyTorch wheel pinned by `uv.lock`.

## Consented retailer CSV contract

Required columns are `source_listing_id`, `title`, `currency`, `base_price`, `stock_status`,
and `listing_url`. Optional columns include `retailer`, `product_id`,
`manufacturer_part_number`, `shipping_price`, `seller_name`, `condition`, `observed_at`, and
`promotion_text`.

The adapter requires a retailer name, feed slug, consent reference, source URL, and access note.
It rejects duplicate listing IDs, non-positive prices, malformed currencies, unknown statuses,
and non-new conditions unless the feed policy expressly allows them. Missing canonical product
IDs receive an `unmatched_product_*` key and must pass entity resolution before optimisation.

## Signed Awin local-feed contract

`awin_feed` authenticates the exact policy bytes with a detached Ed25519 signature. Before using
the signing key, it checks an externally supplied SHA-256 over the exact trust-root bytes, active
key status and validity window, and policy issue/expiry times. Duplicate JSON keys, non-finite
numbers, unknown control fields, oversized files, symlinks/junctions, and files that change while
being read fail closed.

The signed Awin payload identifies numeric advertiser/feed IDs, retailer, access note, effective
rights/territories/retention/deletion terms, exact SGD currency, allowed listing hosts, explicit
category mappings, CSV/gzip dialect, condition policy, optional expected input SHA-256, and bounded
resource limits. It must grant `cache` and `derive` in Singapore before ingestion. Production
catalogue use additionally requires the complete production serving rights; training requires an
explicit training grant; and `published_claims_eligible=true` requires both display/derivation
rights and a distinct signed contractual grant reference. A parser `PASS` cannot create any of
those rights.

The adapter streams compressed/decompressed input under signed byte/row/field/rejection/output
limits and stores duplicate listing IDs in temporary disk-backed SQLite rather than a catalogue-
sized in-memory set. It requires explicit category mappings, validates HTTPS listing hosts,
rejects credentials embedded in input bytes or URLs, keeps stock `UNKNOWN` when signals are absent,
and assigns unmatched product placeholders for later entity resolution. Publication produces an
immutable raw snapshot, authorization receipt, records/rejections/manifest/quality artifacts, and
safe `awin://advertisers/<id>/feeds/<id>` provenance. CLI output reports only safe identities,
hashes, counts, reuse state, and output paths; it does not echo local policy/feed paths.

Current boundary: only CSV and gzip-wrapped CSV are supported; XML and in-application feed
acquisition are deferred. Tests use synthetic policies and rows. No real feed, signed contractual
grant, accepted-row count, catalogue release, price history, training set, or published metric has
been produced. Retention/deletion execution and downstream production approval remain separate
operator gates. The complete operator contract is in [the Awin feed guide](awin-local-feed.md).

## Governed web acquisition contract

`web_product` is an exact-URL crawler, not an unrestricted site scraper. A strict JSON policy
must map each allowed URL to one component category and provide allowed hosts, a reviewed terms
URL and selector, its versioned canonical wording/link/semantic hash, a named user agent, acquisition authority,
downstream data-use rights, and resource limits. Each run fails closed on unsafe DNS, robots
denial or ambiguity, unauthorized redirects, changed terms, unsupported content types, page or
run byte limits, unsupported JSON-LD offers, contradictory currency markers, explicit Singapore
shipping exclusions, unsupported currency, or unknown production shipping. Terms and robots
resources cannot be category-mapped as products. Listing identity is namespaced by source,
retailer, seller, listing URL, and offer identity; canonical product SKU, MPN, and GTIN remain
entity-resolution features and never become seller listing IDs.
Raw pages and receipts are content-addressed, while ETag and Last-Modified validators avoid
unnecessary downloads. Processed web runs are assembled and sealed below the hidden root-level
`.wp/<operation-id>/` control area, then exposed with one same-filesystem, no-replace directory
rename. A catalogue reader therefore sees no run or one complete receipt-validated run, never a
partially written final hash directory.

Web scraping changes the acquisition mechanism, not the origin of an offer. A price taken from
a retailer product page remains retailer-origin data and cannot be published or used for ML
unless the relevant rights permit those uses. Open web sources such as Wikidata and the PCI ID
Repository can enrich identity, but they do not supply current Singapore prices or stock.

The canonical CLI checks the reviewed source registry before opening any governed-web connection:
the policy's source name, exact allowed-host set, and usage scope must match a concrete governed
source entry that also requires scheduled retention. It separately checks restricted-source entries.
For example, [pchardware.org's terms](https://pchardware.org/legal/terms/) currently prohibit
automated extraction, mirroring, republication, and database/API access without written consent;
it is therefore fail-closed at `pchardware.org` and subdomains. A new agreement requires a fresh
terms review and an explicit registry change before a policy can be used.

### Governed-web retention maintenance

The production control is Dagster's `governed_web_retention_hourly` schedule. It is declared
`RUNNING` by default and invokes the destructive receipt engine every hour whenever the pipeline
profile, code location, and Dagster daemon are running. On every run, the asset derives the complete
set of concrete governed-web sources from `data/source_registry.yaml`; an entry is selected only
when its kind, template, required receipt-v2 engine, and maximum maintenance interval pass strict
validation. Keep each concrete source entry in the registry after acquisition authority expires.
The engine reads immutable raw and processed receipts and does not load or reconstruct an active
crawl policy. Also run it immediately after revoking an authority rather than waiting for the next
hourly tick.

The CLI is a diagnostic and emergency control, not complete production orchestration, because its
explicit source list can omit a registered source. Use `--dry-run` to validate a named source or,
under an operator runbook, run an emergency maintenance pass while Dagster is unavailable:

```powershell
uv run --no-sync python scripts/maintain_web_retention.py `
  --source-name dynacore_web_research
```

Each invocation preflights all named sources before deleting anything, operates only inside the
exact `data/raw/<source>/pages` and
`data/processed/<source>/<64-character-run-hash>` roots, rejects symlinks and junctions, and fails
closed on malformed receipts or any unreceipted processed run in either research or production.
Expired processed runs are first renamed to a strictly named deletion tombstone so an interrupted
file removal can resume; non-empty tombstones must retain a valid expired receipt and a tombstone
name alone never authorizes deletion. Unreferenced content-addressed raw bodies and recognized
atomic-write leftovers are removed only after the default 24-hour crash-recovery grace period;
unknown regular files are reported and preserved. The JSON stdout record should be retained in the
operator log and a non-zero exit should page an operator.

Successful publications remove their `.wp` control operation. The hourly retention asset also
reclaims crash-left operations only after the grace period, using the same shared work budget as
raw and published-run retention. Before an operation can be deleted, its content-hashed immutable
intent and optional ready receipt must bind the selected source and run, its directory shape must
be one of the bounded publication crash states, and both the intent timestamp and latest filesystem
change must be older than the grace cutoff. A committed run with only its private cleanup residue
is counted separately.

Any unknown entry, link-like path, out-of-registry source, invalid receipt, or change between
planning and deletion fails the entire pass closed before unrelated raw or catalogue data is
deleted. Operators should retain the scheduled JSON report and investigate a non-zero result rather
than manually deleting `.wp` contents.

Only raw-page v2 and processed-retention v2 receipts are accepted. A legacy v1 receipt fails the
scheduled all-source preflight (or the CLI's named-source preflight) closed and nothing is deleted;
migrate it from retained authority and rights evidence, or purge its artifacts through a separately
reviewed and recorded operator procedure before enabling unattended maintenance. Do not rewrite an
immutable receipt in place.

Before authorising either procedure, create a zero-write evidence plan:

```powershell
uv run --no-sync python scripts/maintain_web_retention.py `
  --source-name dynacore_web_research `
  --plan-legacy-migration
```

This mode validates legacy receipt fields, raw body hashes, deletion-required acquisition
authority, processed manifests, data-quality reports, retention intervals, and exact policy
fingerprints. It emits deterministic evidence hashes and plans zero writes. Exit status 2 means
the retained artifacts cannot support a faithful v2 migration by themselves. A processed
manifest's self-hashed source statistics can supply acquisition-authority and data-use-rights
evidence when they match the receipt's policy fingerprint and usage scope; raw-only policies
still need rights evidence. The planner recovers a processed run's original retrieval interval
only when the complete expected set of terms, robots, and product observation receipts remains:
the crawler defines the interval as their minimum and maximum retrieval timestamps. If that
evidence is incomplete, the exact run remains blocked. Some v1 runs also anchor expiry to their
first product retrieval rather than the earlier full-observation start; those runs require an
explicitly reviewed migration decision because silently changing either timestamp would rewrite
the historical retention contract. The planner never projects the lone v1 `retrieved_at` value
into a fabricated interval, creates a v2 receipt, moves a run, or deletes an artifact.

`--maximum-entries` is one global planning and RAM-work cap across all named sources, raw stores,
processed run trees, and their manifests. The deprecated `--maximum-entries-per-source` spelling is
only a CLI alias for that same global cap. Before deletion, raw files are checked again against
their planned identity and each processed run receives a separate bounded manifest revalidation;
the setting is neither a per-source allowance nor pagination.

Dagster records run state and asset metadata, but the repository does not yet route failures to an
external notification destination or independently detect a missed hourly success. Unattended
operation remains blocked until both alert delivery and a retention dead-man check are configured
and tested as described in the observability guide.

The quarantined WDC historical corpus uses a separate `wdc_research_retention_daily` schedule and
receipt format. It is intentionally excluded from the production governed-web registry because its
rights remain research-only; the daily schedule removes only expired validated WDC artifacts and
cannot promote, display, embed, or train on them.

## Sources intentionally excluded

- Open Icecat's February 2026 Open Content License expressly prohibits using its content for
  machine-learning training. Do not use it for embeddings, entity resolution, LTR, or
  performance models without a separate commercial licence.
- Amazon Creators API content is for driving Amazon sales; its terms restrict aggregation and
  ML training, and price content may be cached for only 24 hours.
- eBay Singapore Browse APIs support runtime discovery, but production access is limited and
  persistence or price-modelling rights need explicit approval.
- Kaggle datasets scraped from PassMark, TechPowerUp, or PCPartPicker are not accepted merely
  because the uploader selected CC0 or MIT; the upstream rights remain unresolved.
- OpenBenchmarking's client is GPL, but that does not automatically license uploaded public
  results for bulk model training. Written permission is required.

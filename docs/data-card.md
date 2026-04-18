# Data card

Status: measured open-data development catalogue; production retailer coverage is not ready

Geography: Singapore  
Currency: SGD  
Last updated: 2026-07-23

## Summary

The intended production catalogue combines manufacturer-verified component specifications,
seller-specific offers and price history, comparable benchmark observations, and permitted review
evidence. The current development catalogue instead contains community specifications that still
require manufacturer verification before hard compatibility use. Eligible data supports retrieval,
entity resolution, performance modelling, ranking, compatibility, optimisation, price context, and
explanations.

Current counts must be read from a versioned ingestion/evaluation manifest. A target such as
10,000 listings is not a measured fact until that evidence exists. Price snapshots never count as
retailer listings.

## Measured local ingestion inventory

The following processed-batch manifests and record-file hashes were independently checked on
2026-07-22. These are parser outputs, not yet production database counts or model-evaluation
datasets.

| Source | Accepted / recorded rejection rows | Eligibility and interpretation | Evidence digest |
| --- | ---: | --- | --- |
| BuildCores OpenDB, pinned commit `6a64ab14fb1ab1bc1f3030d36b70bddcc2afeb0f` | 25,666 / 33 | ODC-By 1.0 with attribution. All accepted canonical-product records retain a unique source-record ID and per-product provenance. A future derived model may be served only while the mandated BuildCores attribution is shown on every applicable public output. Community fields still require manufacturer verification before hard compatibility use; this is not proof of production database loading or retailer coverage. | Raw `f3ee75dd...fd383f`; manifest file `72fe9ef3...e5dd39`; records `8c738c51...24ce2` |
| Blender Open Data deterministic hash sample | 250,000 / 1,000 retained rejection records | CC0 observations selected with seed `buildsignal-blender-v1` after a complete scan of 422,319 submissions and 1,243,834 valid observations. Rejection logging was truncated, so 1,000 is not the total invalid-observation count. Context/version fields must remain attached. | Raw `c0f9d35c...af4833`; manifest file `9497a2c9...7e2bc`; records `393a10e9...fd06` |
| MLPerf Inference v6.0, pinned commit `4d3916ac9cf474b679cdfcf492d43a0559418ad1` | 520 / 0 | Apache-2.0 system results. Only 22 rows are flagged as eligible for single-component attribution; the other 498 must remain system-level evidence. | Raw `52fa813a...384ae`; manifest `7c3ef2ee...c36b3`; records `547aee49...129d` |
| Dynacore controlled PDF | 485 / 137 | Development-only: 69 manual-review rows and 68 hard rejects. All 485 accepted rows are explicitly ineligible for training and public claims because an open licence was not established. | Raw `6e243d7b...b9ba`; manifest `2ff6ab26...49a4`; records `64aa01ed...141d` |

The BuildCores 800-record fast slice and 3,000-record portfolio slice are subsets of the
25,666-record full batch and must not be added to it. Their manifests remain useful for smoke tests
and bounded API demonstrations only. Full local paths live under
`data/processed/<source>/<raw-hash>/`; these source artifacts are Git-ignored and may require
re-acquisition according to their access terms.

Exact verified evidence registry:

- BuildCores full-batch manifest:
  `data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/full/manifest.json`;
  manifest file hash `72fe9ef33e06452d795b14f13aa8742fdc0767b32ec25c008a4c683777e5dd39`;
  records hash `8c738c5136615080ac29ced074726f97cab473c5a45eafdf19c26504aa624ce2`.
- Blender full-scan hash-sample manifest:
  `data/processed/blender_open_data/c0f9d35c20807776138b0590097177b8ef2172119cc19aae8d1bad1b55af4833/hash_sample-250000-scan-all/manifest.json`;
  manifest file hash `9497a2c91055c4ea50c6af6c9a9d2f7b5eea0b0cbf07eaf0b36bb574d1b7e2bc`;
  records hash `393a10e957256d23ddcb2a2a16cb2458a358ca682a38b65580a1339fb625fd06`.
- MLPerf manifest:
  `data/processed/mlperf_inference_v6/52fa813a27834e8de38eca9fd381688df1cf9cd95020d2bd9e951f6cc16384ae/manifest.json`;
  manifest hash `7c3ef2eeda37ef7e3ccc09aa842ec2f424b72e567834e6e738a433bc28fc36b3`;
  records hash `547aee49784f7d60f8e03e1ce2425bcc02313af2c5b114a142a0313757b4129d`.
- Controlled Dynacore manifest:
  `data/processed/dynacore_controlled_pdf/6e243d7bf1cba090f529b09a9276fac03fedddcadb8c11cf9ce7ec1e674bb9ba/manifest.json`;
  manifest hash `2ff6ab26ea526b56177bb28801fbef37c25625af8d892303f0daef9c591149a4`;
  records hash `64aa01ed7c2da3f667f8e9cf149594bd41054a0c73ddfa9fa633dd78ea38141d`.

The corrected CPU performance dataset is stored at
`data/processed/model_training/blender_cpu_content_creation_corrected_v1/`. It conservatively joins
the full BuildCores batch to the deterministic Blender hash sample without fuzzy matching, then
selects the exact Blender 4.0.0 / `junkshop` / CPU / Windows cohort. The 1,625 joined observations
aggregate to 172 hardware rows across 109 leakage groups. Its CSV SHA-256 is
`a4922e0d51a7981f40a257a363cb25649ce5cc27cf32a7517b5e560652acfdf9`; the manifest SHA-256 is
`e9cc28acaa3bea52e5953577cc8c35873a1ce6c1cb27e46639fbf7bdfe727977`.

The corrected target contract is the per-hardware median of the observed `samples_per_minute`
field, with unit `samples/minute` and `higher_is_better=true`. The earlier
`blender_performance_250k_full` dataset incorrectly labelled these values as seconds and applied a
reciprocal transform. Models v2 and v3 derived from that inverted target are revoked. The corrected
dataset is also non-promotable: its snapshot was adaptively explored after v2/v3 results were seen,
and no untouched external frozen cohort has been evaluated. Row-level external-claim eligibility
is therefore false.

The separate GPU pilot dataset is stored at
`data/processed/model_training/blender_gpu_performance_250k_full/`. It conservatively joins the
same licensed BuildCores batch to the Blender sample, selects Blender 4.5.0 / `junkshop` / GPU /
OPTIX / Windows, and aggregates 1,939 matched observations to 50 hardware rows across 36
product-family leakage groups. The target is correctly oriented throughput,
`1000 / median_render_seconds`, with `higher_is_better=true`. Its CSV SHA-256 is
`bdcea49ca7814a6b2a45f3697dc76d0f4148c50a4edcf29b3236d69b2de62bd2`; its manifest SHA-256 is
`a63c944b1054e430169a1b84e96e5ee0afe4d9cefb34461d6afb62dc5fcf0534`.

This is `measured_pilot_non_promotable`, not a release dataset: all rows are explicitly
ineligible for external claims and the manifest records only 36 leakage groups, below the
100-group credible grouped-evaluation minimum. Its manifest records 383 exact GPU cohorts from
the same sampled snapshot; the selected cohort is the largest at 36 families, and none reaches
50 families. Any derived public output would also need the
BuildCores ODC-By attribution recorded in the source registry. The pilot may support engineering
checks only; it cannot support serving, precise predictions, or public model-quality claims.

The 2026-07-23 temporal audit did not supply a fresh test cohort. It compared snapshot
`c0f9d35c...af4833` (422,319 submissions) with snapshot `67582ebc...93f5` (422,432 submissions),
finding 113 novel submissions and 339 novel observations. Zero novel observations matched every
frozen cohort field: Blender 4.0.0, `junkshop`, CPU, Windows, build hash `878f71061b8e`, benchmark
script `3.1.0`, and scene checksum `f5515a21...a0cdce`. Consequently the audit performed no model
inference, reported no external R-squared/MAPE, did not pool rows into development data, and did not
promote the model. The report is
`artifacts/evaluation/performance-temporal-v4/957139150dbc1bbdd6deab1810b966ea59a33d9768b206e5033287c88a3f044b.json`;
its semantic digest is the filename digest and its exact file SHA-256 is
`e675a0d1de5f0556aedeebb41c93f19f8130f3af22b3a010b7dc9c789fb458cf`. The frozen retrospective
protocol SHA-256 is `6952cd7a220920fc9be882211577a67a67268a055e702e4b64955ada5123154a`.

For data version `processed-66286ad2cb30278c`, the controlled retailer batch contains 485 offers
with complete offer provenance, but only two were safely mapped by unique exact manufacturer-part-
number plus brand; 483 remain unmatched, for a 0.412% mapping rate. No offer has known in-stock
status or explicit data-use rights. The current rights registry sets display, caching, history,
redistribution, embedding, training, and derivation to false, and the readiness artifact reports
`production_ready=false`. These are development-only parser records, not licensed Singapore market
coverage. Evidence:
[`catalog-readiness-current.json`](../artifacts/evaluation/catalog-readiness-current.json) and
[`catalog-rights-audit-current.json`](../artifacts/evaluation/catalog-rights-audit-current.json).

The external Zenodo Dn7 transfer benchmark is registered for bounded evaluation under CC BY 4.0,
not for PC-retailer training or production claims. Its guarded source-record-disjoint
materialization retained 15,951 train pairs (481 positives), 1,945 validation pairs (144
positives), and 1,802 test pairs (138 positives). The held-out CPU LightGBM result is documented
in the model card and evaluation report; it missed the combined promotion gates and does not
replace a Singapore retailer-labelled dataset.

## Dataset families

| Dataset | Unit | Primary use | Key grouping for evaluation |
| --- | --- | --- | --- |
| Canonical catalogue | Unique manufacturer model/SKU | Retrieval, compatibility, features | Product family and generation |
| Retailer listings | Retailer plus source-listing ID | Price, stock, entity resolution | Canonical product and retailer |
| Price snapshots | Listing at an observation time | Current value and price history | Listing and time |
| Benchmarks | Product, benchmark context, observation | Observed workload evidence and regression | Product family/generation |
| Duplicate labels | Listing/canonical pair | Entity-resolution training/evaluation | Blocking group/product family |
| Relevance judgments | Query/product grade from 0 to 4 | Retrieval/ranking evaluation | Query ID |
| Compatibility cases | Complete or partial build plus expected rule results | Rule/optimiser validation | Scenario/rule family |
| Review evidence | Product/aspect/source statement | Cited aspect summaries | Product/source |
| Interaction events | One user/session action at a rank | Later ranking signals | User/session/query and time |

## Sources and permitted use

Source adapters may consume manufacturer pages/documents, retailer APIs or feeds, authorised
crawling, controlled imports, benchmark datasets, and review sources whose terms allow the
intended processing. Every adapter records:

- source name, URL, and type;
- retrieval and last-verification times;
- raw-content SHA-256;
- parser version;
- licence or access note; and
- extraction confidence.

Manufacturer evidence is authoritative for sockets, chipsets, supported memory, form factors,
dimensions, connectors, capacity, power guidance, and warranty. Retailer evidence is
authoritative only for that seller's offer, price, shipping, stock, condition, and promotion at
the observed time. Benchmark and review evidence must retain context and source attribution.

Public review evidence is a bounded, release-pinned JSONL artifact rather than a general review
corpus. Every statement is limited to 500 characters, cites one credential-free HTTPS source URL,
and repeats that URL in its provenance record with retrieval time, raw-content hash, parser version,
and licence/access note. The source must grant active Singapore display, cache, history, and
derivation rights. An explicit empty artifact is valid when no source meets those conditions;
uncited or scraped text is not substituted.

An adapter without a documented permission/access basis is not eligible for scheduled ingestion.
Robots restrictions, authentication, rate limits, copyright, and redistribution limits remain in
force even when data is technically accessible.

The implemented Awin path accepts only a local CSV/gzip already acquired by an authorized
operator. It verifies exact policy bytes, a detached Ed25519 signature, an independently pinned
trust-root SHA-256, key/policy validity, Singapore territory, signed use grants, exact category
mappings, currency, listing hosts, and resource limits before storing bytes. It emits a
content-addressed authorization receipt and safe `awin://` provenance without echoing a feed
download URL or input path. `published_claims_eligible=true` requires a distinct signed
contractual-grant reference. No real Awin feed, trust root, agreement, accepted records, or release
is present, and synthetic adapter tests are not inventory evidence. External acquisition,
credential handling, signing-key custody, retention/deletion, and production approval remain
operator obligations; see [the Awin feed guide](awin-local-feed.md).

## Collection and lineage

1. Fetch bytes and response metadata without overwriting earlier snapshots.
2. Compute the raw hash before parsing.
3. Parse with a named parser version and explicit field confidence.
4. Validate category, required fields, ranges, currency, and stable source identifiers.
5. Generate conservative entity-resolution candidates and preserve conflicts.
6. Upsert canonical/listing records and deduplicate price observations.
7. Rebuild search text and embeddings only when canonical content changes.
8. Emit row counts, rejected records, missingness, match decisions, and quality results.
9. Build a content-addressed manifest for any training or reported evaluation dataset.

Raw and processed data directories are excluded from Git because source terms and file sizes may
not permit redistribution. Manifests and evaluation artifacts are the audit surface; a manifest
hash does not itself grant permission to redistribute the referenced content.

## Schema notes

Compatibility-critical values are nullable because real sources are incomplete. Null does not
mean false, zero, unsupported, or compatible. Unit-normalised attributes retain explicit names
such as millimetres, watts, gigabytes, MT/s, and MB/s. Currency values use decimal arithmetic.

Benchmark observations are comparable only when the material context agrees. At minimum retain
benchmark/version, resolution, preset, operating system, driver version, score unit, direction,
source, and date when available. Scores from different contexts must not be pooled directly.

Canonical IDs represent unique variants. Numeric conflicts such as 32 GB versus 64 GB, 1 TB
versus 2 TB, or distinct form factors are hard negative entity-resolution evidence even when
titles are otherwise nearly identical.

## Labels

### Entity resolution

The target portfolio dataset has at least 2,500 labelled pairs: true duplicates, similar-variant
hard negatives, and random negatives. Labels should be independently reviewed when MPN, capacity,
generation, or form factor conflicts exist. Evaluation prioritises precision because a false
merge contaminates price, benchmark, and recommendation evidence.

Alembic revision `20260723_0006` and `scripts/manage_annotations.py` provide the durable collection
surface for this future dataset. Reviewer identity and roles are resolved from PostgreSQL using an
issuer/subject pair already verified by upstream OIDC middleware; raw JWTs are not accepted or
stored by the CLI. Each item requires exactly two independent immutable judgments. Review leases
and idempotency keys are stored only as hashes, a reviewer cannot reclaim the same item, and a
reviewer cannot adjudicate an item they judged. Disagreement or `UNCERTAIN` requires an independent
adjudication. A frozen entity-resolution release preserves both raw judgments and adjudication
alongside deterministic `pairs.jsonl`, manifests, source-policy hashes, group/split metadata, and
artifact hashes. No such human-label release has been created, so the 2,500-pair target remains
unmeasured.

### Retrieval and ranking

The target frozen set has approximately 150 realistic queries and at least 2,000 graded
query-product judgments. Two reviewers grade 0 (irrelevant) through 4 (best fit), then resolve
material disagreement. All compared systems use the same judged pool and candidate rows. Query
IDs, rather than individual rows, define train/validation/test splits.

The current 32-query retrieval artifact uses deterministic specification predicates as silver
labels. It contains no human relevance judgments and is permanently ineligible for promotion or
resume metrics. Human-reviewed qrels remain a required dataset, not an optional refinement.

The same annotation workflow freezes relevance projects only after exactly two judgments per
query/product, independent adjudication where required, source rights, non-synthetic evidence, and
leakage-group-safe split checks pass. Its content-addressed release contains the raw
`human-judgments.json`, adjudicated `qrels.json`, `query-split.json`, and evidence snapshots. Hard
requirement failures are stored as structured codes separate from the 0-4 grade. No frozen human
qrels release exists at this snapshot, so LambdaMART training and the 18% lift claim remain blocked.

The v3 starter capture now commits label-free ranking inputs before review. Its reviewer-safe file
contains 16 queries and 480 tasks; a separate hashed pre-label snapshot contains the exact feature
rows and matrix. Opaque row hashes are retained in annotation evidence, and the later materializer
may append adjudicated grades but cannot change features. This closes the feature/label-lineage
mechanism, not the missing human-data gap.

### Compatibility

Cases cover known valid and invalid builds, exact boundaries, missing data, socket and memory
generation mismatch, clearances, PSU wattage/connectors, BIOS warnings, and generated
configurations. Expected results name rule version and PASS/FAIL/WARNING/UNKNOWN per rule.

### Behavioral data

Views are weak evidence; comparisons, saves, selected components, and retailer clicks are stronger
positive signals; dismissal is negative. Position and exposure bias must be modelled or corrected.
Behavioral labels must not be treated as objective relevance, and the product must not claim
personalisation until sufficient per-user history exists.

## Splits and leakage controls

- Entity-resolution splits group related variants/blocking families so title templates do not
  leak across train and test.
- Performance splits group product families or hardware generations; close factory-overclocked
  variants cannot span train and test.
- Ranking splits group by query. Once live data exists, use earlier time for training and later
  time for validation/test.
- Price features are point-in-time: only snapshots at or before the query timestamp are allowed.
- Test labels, benchmark outcomes, canonical IDs created after the prediction cutoff, and future
  interaction signals are forbidden training features.

The core split helpers validate group disjointness. Split membership and random seeds belong in
the dataset manifest.

## Synthetic-data policy

Synthetic rows are valuable for parser edge cases, property tests, solver enumeration, API demos,
and pipeline smoke tests. Every row participating in an evaluation declares `is_synthetic`.
Externally reportable artifacts require either zero synthetic rows or explicit exclusion of all
synthetic rows. The evaluation contract records total, evaluated, and synthetic row counts and
blocks reportability when this condition is not met.

Synthetic catalogue scale, synthetic labels, or tests passing on generated builds may never be
presented as collected market coverage or measured real-world model quality.

## Quality checks

The current ingestion evaluator enforces structural IDs/envelopes, a hard
rejection-rate cap, and basic product/benchmark/listing/use checks. It also
compares each new run with the newest valid `PASS` report for the same source
and output variant. The baseline is read from aggregate manifests and
data-quality reports only, never from historical raw records. Once that prior
batch has at least 10 accepted records, the following are promotion-blocking
errors: accepted-count drops below 70%, retailer-listing drops below 70%, a
sufficiently covered record type disappears, a sufficiently covered category
drops below 60%, or rejection rate rises by more than 15 percentage points.
The first passing run remains a baseline-establishment run; it must pass its
structural checks but has no historical comparator. A failed candidate is
retained for investigation and cannot replace the passing baseline.

The following are still target source-promotion controls; any control not yet
implemented is not treated as a measured release gate:

- missingness for category-specific required fields;
- positive and plausible SGD price ranges;
- recognised currencies and new-condition policy;
- unexpected category changes;
- MPN/GTIN conflicts and mapping loss;
- parser extraction-rate changes by source and field;
- duplicate snapshot prevention;
- source freshness and last successful run;
- orphaned listing, benchmark, provenance, or embedding rows.

The implemented history thresholds are global defaults, rather than calibrated
per-source policies. They are deliberately conservative guardrails, not a
claim that a source is complete, fresh, rights-cleared, or accurate. A
threshold breach blocks promotion of that source snapshot but does not delete
the previous known-good version.

## Biases and limitations

Singapore retailer coverage can overrepresent large sellers, currently stocked models, gaming
hardware, English-language titles, and recent generations. Review evidence may overrepresent
extreme experiences. Benchmarks can favor particular games, drivers, operating systems, presets,
or vendor software. Missing smaller sellers can distort observed value and stock.

Mitigations include source-level slices, seller counts, temporal validation, workload-specific
benchmark contexts, brand/model-family error slices, source freshness, and UNKNOWN compatibility
results. These controls reduce but do not eliminate coverage bias.

## Privacy and retention

Catalog data is non-personal, but interaction events can become personal when tied to accounts.
Anonymous sessions are the default. Retention and deletion policy must be defined before public
accounts launch. Logs avoid raw free text and credential-bearing URLs. If public build sharing is
released, it must use an allow-listed projection rather than the full query or event record.

## Version and reproducibility contract

Promotion-eligible training/evaluation datasets must have a manifest containing schema version,
dataset name/version, row and group counts, file digests, synthetic-data declaration, split
metadata, and a content hash. Promotion-eligible evaluation artifacts must name that manifest
hash, sample counts, metrics, confidence intervals, run metadata, reportability, and their own
hash. Legacy engineering evidence is explicitly labelled where it does not yet satisfy this target
contract. A changed file, parser, label, split, or filter creates a new version; published artifacts
are immutable.

The processed-catalogue production path derives descriptive statistics only when every offer in
the loaded catalogue passes the production rights gate. It analyzes at most the newest 10,000
stored observations and exposes current delivered price, 30/90-day medians, 90-day
percentile/recent low/volatility, observed seller and stock trends, robust anomaly flags, history
sufficiency, and truncation state. Sparse history cannot expose percentile or volatility. This is
labelled `descriptive_observed_history`; it is not a forecast or a guarantee of a live/future
price. The current rights-blocked retailer batch yields no eligible production price-intelligence
claim and the processed-catalogue API suppresses current prices and raw price observations as
well as derived statistics. Seed and public-demo price intelligence is explicitly illustrative and
is not evidence of rights-cleared retailer history.

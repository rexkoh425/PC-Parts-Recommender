# Product requirements

Status: implementation baseline  
Product scope: new desktop PC components sold in Singapore  
Currency: SGD  
Last updated: 2026-07-22

## Product outcome

PC Build Recommender turns a structured budget, workload mix, retained hardware, physical
constraints, and preferences into three to five distinct complete builds. A returned build is
useful only when its selected listings are within budget and in stock, all known hard
compatibility rules pass, missing evidence is visible, and the component-level reasons can be
traced to a data, rule, or model version.

This is an applied search-and-recommendation product. Hybrid retrieval and learned ranking find
good candidates; versioned deterministic rules decide compatibility; CP-SAT assembles the final
build. An LLM is not an authority for compatibility, price, benchmark, or availability claims.

## Users and jobs

| User | Primary job |
| --- | --- |
| First-time builder | Obtain a complete, explainable build without learning every interface and clearance rule. |
| Gamer | Balance target resolution and frame rate against total cost, noise, and power. |
| Local-AI user | Prioritise usable GPU memory, inference performance, software support, memory, cooling, and power. |
| Developer | Prioritise compilation, single- and multicore performance, memory, and responsive storage. |
| Content creator | Balance rendering, encoding, GPU acceleration, memory, and storage capacity. |
| Upgrade user | Retain owned parts and optimise only the compatible remainder. |

## Authoritative request contract

Structured form values are authoritative. Natural-language input may populate the same schema,
but the user must be able to inspect and change the parsed values before generation.

Required input:

- total budget in SGD;
- one or more workload profiles whose weights sum to one;
- retained components, if any;
- minimum GPU memory, system memory, and storage requirements;
- case-size, Wi-Fi, stock, and brand constraints;
- noise, efficiency, upgradeability, and brand preferences; and
- requested alternative profiles, from one to five.

Hard requirements and soft preferences must never be conflated. A preference can affect rank;
a hard requirement removes a candidate or makes the optimisation infeasible.

## Functional requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| FR-01 | Retrieve CPU, GPU, motherboard, memory, storage, power-supply, cooler, and case candidates with BM25, vector similarity, reciprocal-rank fusion, and structured filters. | Frozen-query retrieval artifact comparing each stage. |
| FR-02 | Preserve one canonical product separately from seller-specific listings and price snapshots. | Database constraints and ingestion idempotency tests. |
| FR-03 | Match listing duplicates conservatively and queue uncertain pairs for review. | Labelled-pair precision, recall, F1, and threshold artifact. |
| FR-04 | Return PASS, FAIL, WARNING, or UNKNOWN for every applicable compatibility rule. | Unit, property, boundary, and generated-build tests by rule version. |
| FR-05 | Prefer observed benchmark evidence; label every model estimate as predicted with model version and confidence. | API contract tests and model evaluation artifacts. |
| FR-06 | Rank suitable components for the workload, price, quality, availability, and preferences. | Paired NDCG evaluation against BM25 on an identical frozen candidate set. |
| FR-07 | Select exactly one required component per category within budget and all hard constraints. | CP-SAT versus exhaustive enumeration on reduced catalogues. |
| FR-08 | Return three to five meaningful profiles where feasible, with later builds differing in at least two meaningful components. | Optimiser diversity tests and response contract tests. |
| FR-09 | Explain price, compatibility, performance basis, trade-offs, and component alternatives without inventing evidence. | Snapshot/API tests with source and version assertions. |
| FR-10 | Replace one component while holding requested components fixed, rechecking compatibility, and re-optimising the remainder. | Integration and browser flow tests. |
| FR-11 | Save, compare, share, and refresh builds at current prices. | API and browser tests; public links must not expose private session data. |
| FR-12 | Record product/build interactions with query, rank, data, model, and rule versions. | Event-schema and persistence tests. |
| FR-13 | Show ingestion failures, stale prices, unmatched listings, missing fields, and rule failures to administrators. | Dagster asset status and admin API/UI tests. |

## Safety and truthfulness requirements

- Missing socket, BIOS, dimensions, slot width, connector, or power data produces UNKNOWN, not
  PASS.
- No build with FAIL or unresolved hard UNKNOWN results may be presented as fully compatible.
- Retail price means the cheapest current permitted in-stock listing including known shipping;
  missing shipping must be disclosed.
- Review summaries may only use permitted evidence and must retain source URLs.
- Synthetic rows may support tests and demos but may not support reported model-quality claims.
- A metric is a target until a content-addressed dataset manifest and a verified evaluation
  artifact establish a measured result.
- The product does not automate purchasing and does not guarantee future prices, stock, BIOS
  state, assembly quality, thermals, or vendor warranty outcomes.

## Non-functional requirements

| ID | Requirement | Initial target, not a measured result |
| --- | --- | --- |
| NFR-01 | Search latency | p95 below 500 ms at the declared load profile. |
| NFR-02 | Build-generation latency | p95 below 2.5 s at the declared candidate caps and solver limit. |
| NFR-03 | Reproducibility | Every result names data, model, ranking-feature, rule, and optimiser versions. |
| NFR-04 | Availability behavior | Source or model failure degrades explicitly; stale evidence is timestamped. |
| NFR-05 | Privacy | Anonymous session IDs by default; no sensitive free text in ordinary logs. |
| NFR-06 | Operability | One default Compose command; optional Dagster and MLflow profiles. |
| NFR-07 | Maintainability | Modular monolith, typed contracts, migrations, unit/property/integration/browser tests. |

## Target scale and quality gates

These are goals, not current inventory or achieved metrics:

| Area | MVP target | Portfolio target |
| --- | ---: | ---: |
| Canonical products | 750 | 3,000 |
| Retailer listings | 3,000 across 3 sources | 10,000 across 5 sources |
| Benchmark observations | 3,000 | 8,000 or more |
| Labelled duplicate pairs | 1,000 | 2,500 |
| Judged search scenarios | 75 | 150 |
| Graded relevance labels | 1,000 | 2,000 or more |
| Compatibility test builds | 2,000 | 10,000 |
| Pilot/study users | 5 | 20 |

Model-quality targets are defined in [evaluation-report.md](evaluation-report.md). They may not
be copied into a résumé or demo as results until the report links a reportable artifact.

## Release slices

1. Foundation: typed schemas, PostgreSQL/pgvector migration, provenance, seed catalogue, API.
2. Ingestion: permitted sources, immutable snapshots, parsing, listings, prices, quality checks.
3. Entity resolution and compatibility: conservative matching and deterministic rules.
4. Retrieval and performance: BM25/vector/RRF plus observed and predicted workload scores.
5. Ranking and optimisation: query-grouped LambdaMART and exact constrained build selection.
6. Product experience: full build flow, replacement, saving, comparison, explanations, events.
7. Evidence: frozen datasets, load tests, controlled user study, data/model cards, public demo.

## Scope exclusions for the first release

Laptops, used parts, custom water cooling, purchasing, detailed thermal simulation, 3D assembly,
countries outside Singapore, neural price forecasting, and claims of personalisation before
sufficient per-user history are intentionally excluded.

## Definition of done

The product is portfolio-ready only after the complete browser flow works on current source data,
every returned build independently passes hard compatibility checks, the optimiser matches
exhaustive search on reduced catalogues, retrieval/ranking/model baselines use frozen leakage-safe
splits, latency is measured, provenance and versions are exposed, and every public metric can be
reproduced from saved artifacts. Passing software tests alone does not establish model quality or
user impact.


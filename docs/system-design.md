# System design

Status: implementation baseline; interfaces may evolve through migrations  
Last updated: 2026-07-23

## Design decisions

The system is a modular monolith: one Next.js application, one FastAPI process, one PostgreSQL
database with pgvector, one Dagster code location, and one installable Python core package. This
keeps transactions and versioning simple at the expected scale while maintaining explicit module
boundaries. Kafka, Kubernetes, separate model services, and a separate vector database are not
justified until measurements demonstrate a bottleneck.

Compatibility filtering runs before ranking and optimisation, and its result is never delegated
to a language model. Observed performance evidence has priority over model estimates. CP-SAT owns
final combinatorial selection; component rank scores are inputs, not permission to violate a hard
constraint.

## Context and containers

```mermaid
flowchart LR
    U["User"] --> W["Next.js web"]
    W --> A["FastAPI modular monolith"]
    A --> R["BM25 + pgvector retrieval"]
    A --> C["Versioned compatibility rules"]
    A --> M["LightGBM models"]
    A --> O["OR-Tools CP-SAT"]
    R --> P[("PostgreSQL + pgvector")]
    C --> P
    M --> P
    O --> P
    D["Dagster ingestion and evaluation"] --> P
    D --> F["Raw snapshots and artifacts"]
    D --> X["MLflow"]
    S["Permitted manufacturers, retailers, benchmarks, reviews"] --> D
```

Default Docker Compose services are PostgreSQL, API, and web. `pipeline` adds Dagster; `mlops`
adds MLflow. The optional profiles keep ordinary application startup light. The API applies
Alembic migrations before serving and waits for a healthy database. Container health checks are
service-local and do not depend on third-party connectivity.

The separate production Compose contract does not reuse those local defaults. It provisions
distinct PostgreSQL roles/databases, runs application and MLflow migrations as one-shot release
jobs, separates Dagster webserver/daemon/gRPC code processes, uses PostgreSQL Dagster metadata,
binds operator endpoints to loopback, enforces resource/PID/log limits and read-only application
filesystems, and adds optional Prometheus, blackbox, PostgreSQL-exporter, and recovery profiles.
An external TLS/OIDC ingress and off-host encrypted backups remain deployment responsibilities.
See `docs/deployment-runbook.md` for the complete trust and rollback boundary.

## Module boundaries

| Boundary | Responsibility | Must not decide |
| --- | --- | --- |
| `catalog` | Domain and persistence contracts, canonical products, listings, provenance | Relevance or compatibility |
| `entity_resolution` | Blocking, pair features, conservative duplicate probability, review queue | Build selection |
| `retrieval` | Release-bound BM25 online, embeddings, pgvector search, RRF, candidate recall; PostgreSQL FTS diagnostic fallback | Hard compatibility |
| `performance_models` | Workload-specific observed/predicted scores and confidence | Whether missing data is compatible |
| `ranking` | Query-grouped features and component ordering | Feasibility |
| `compatibility` | PASS/FAIL/WARNING/UNKNOWN versioned rules and evidence | Soft preference value |
| `optimizer` | Budget, exact selection, feasibility, profile objectives, diversity | Source truth |
| `pricing` | Current/median/percentile/volatility labels | Future price promises |
| `reviews` | Permitted aspect evidence with citations | Generic unsupported sentiment |
| `explanations` | Render already-computed reasons and evidence | Invent facts or override rules |
| `annotation` | OIDC-bound reviewer roles, blinded leases, immutable judgments/adjudication, frozen label exports | Infer a label or self-authorize data rights |
| `services/api` | Validation, orchestration, persistence, versioned responses | Hidden business logic in routes |
| `pipelines` | Immutable acquisition, parsing, quality, training, evaluation | Serving mutable unversioned models |

## Online recommendation sequence

1. FastAPI validates the structured request; natural language must already be represented in the
   same inspectable schema.
2. The production retrieval layer scores the immutable, manifest-validated search corpus with
   BM25 and searches pgvector in one read-only database snapshot, fuses ranks with RRF, then
   applies direct structured filters. The database determines the active, release-matched IDs that
   BM25 may score, so stock, price, brand, and specification constraints are never stale index
   metadata. PostgreSQL `ts_rank_cd` remains an explicit diagnostic fallback, not the production
   BM25 implementation.
3. Compatibility evaluates candidates against retained parts and known requirements. FAIL is
   removed. Hard UNKNOWN is retained only for an explicit infeasibility or warning path, not
   silently called compatible.
4. Observed benchmarks are selected from comparable benchmark contexts. When absent, a
   versioned workload regressor may provide a predicted score with confidence.
5. The ranker scores remaining products using retrieval, workload, price, availability, quality,
   freshness, and preference features.
6. CP-SAT selects one item in each required category subject to budget, stock, retained parts,
   feature requirements, pairwise compatibility, clearances, connectors, and power headroom.
7. Each objective profile is solved with integer-normalised scores. Diversity constraints require
   later solutions to differ in at least two meaningful components.
8. The response persists the request/build and emits data, ranking-model, rule, and optimiser
   status/version fields. Explanations render stored decisions and evidence.

Product-price responses separately summarize rights-eligible stored observations. The summary is
explicitly descriptive: current delivered price, 30/90-day medians, 90-day percentile/recent low/
volatility, observed seller/stock trends, history sufficiency, and robust anomaly flags. Sparse
history withholds percentile/volatility. Analysis is bounded to the newest 10,000 observations and
never forecasts or promises a live/future price.

Candidate caps are operational guardrails, not quality claims: 30 CPU, 30 GPU, 50 motherboard,
40 memory, 40 storage, 40 PSU, 30 cooler, and 30 case candidates by default. Recall and solver
latency must be measured before changing them.

## Data and persistence

The initial Alembic migration creates canonical products, retailer listings, price snapshots,
benchmarks, provenance, rules, review evidence, queries, builds, components, and interactions. It
also creates:

- a PostgreSQL full-text GIN index over the canonical search document for diagnostics and fallback;
- a 384-dimensional pgvector embedding table keyed by product and embedding model; and
- an HNSW cosine index for initial MiniLM retrieval.

The vector dimension is a schema contract. Adopting a different embedding dimension requires a
new table/migration or a coordinated re-embedding migration; vectors from incompatible models
must never share a fixed-dimension column.

Alembic revision `20260723_0006` adds the durable annotation control plane: OIDC-bound reviewers
and roles, projects/groups/items, leases, immutable judgments and adjudications, frozen exports,
and append-only audit events. Exactly two independent review assignments are required. Lease and
idempotency secrets are hashed, a reviewer cannot reclaim or adjudicate the same item, and database
triggers reject updates/deletes to human decisions and the audit ledger.

Raw source responses are immutable and content-addressed. Parsed rows carry source URL, source
type, retrieval time, raw hash, parser version, access/licence note, verification time, and
extraction confidence. Upserts use stable source keys, MPN/GTIN evidence, and canonical IDs.
Price snapshot uniqueness is listing plus observation time.

## Offline pipeline

Dagster is an optional dependency so the API and unit-test environment can import pipeline
modules without it. The initial code location schedules a 12-hour observability materialization;
source-specific jobs may run more frequently according to permission and rate limits. Every run
should follow:

```text
fetch -> immutable raw snapshot -> parse -> validate -> block/match -> upsert
      -> price snapshot -> search document -> conditional embedding -> quality report
```

Content hashes make ingestion idempotent. Embeddings regenerate only when their canonical input
or embedding-model version changes. Failed extraction-rate, category, price, MPN-conflict, mapping,
or duplicate-snapshot checks stop promotion while preserving raw evidence for diagnosis.

The Awin source adapter receives only an already-downloaded local CSV/gzip. Before acquisition
bytes enter the raw store, it verifies an exact Ed25519-signed policy against an independently
pinned trust-root digest, key/policy validity, Singapore use grants, category/host/currency rules,
and resource limits. It emits an authorization receipt and safe `awin://` identity; it has no
network/API-key surface. Feed access, contractual rights, signing-key custody, download, retention,
and deletion are external operator responsibilities. No real Awin feed or grant is bundled.

## Production release authority

One operator-pinned `pc-build-recommender.serving-release.v3` manifest is the release root. Both the
catalogue import job and API bootstrap verify the same exact catalogue, offers, reviewed mappings,
review-evidence JSONL, and entity-resolution artifacts. ER authority is derived only after
cross-checking a LightGBM model
and fitted calibrator, metadata, serving evidence, v2 human-labelled evaluation, threshold/catalogue
policy, active Singapore rights approval, and all component/binding SHA-256 values. Legacy
eligibility booleans and direct diagnostic paths cannot promote a release. Import and API consume
the same read-only release directory and digest; no promoted ER release currently ships.

Review evidence is a separate bounded, cited catalog input rather than a crawler or generic text
store. Each record names a canonical product and supported aspect, carries at most 500 characters of
evidence, has matching credential-free HTTPS data/provenance URLs, and carries active Singapore
display, cache, history, and derivation rights. The manifest pins its bytes; an explicit empty
artifact is required when no permitted evidence is available. The API then omits review claims
rather than deriving them from ungoverned text.

The same manifest binds the embedding index and a local semantic-encoder bundle by relative path,
SHA-256, file count, and byte count; operator settings must independently pin the exact path/digest.
The bounded validator rejects symlinks/junctions and mutation, Sentence-Transformers loads with
`local_files_only=true`, and startup warm-up must produce a finite, nonzero, normalized vector of
the stored dimension. Production is fail-closed because no encoder-weight bundle is committed.

LambdaMART features are committed before review in a separate non-reviewer snapshot. Each
annotation task preserves only opaque row hashes; after freeze, a materializer verifies those
hashes, the human release, qrels, and split before appending grades. Human training and evaluation
reject raw feature JSONL without the resulting dataset manifest.

LambdaMART publication is also immutable but occurs before a release manifest references it. A
publication-intent digest binds exact pre-label/labeled feature, human, qrels, split inputs and
training settings. Model,
metadata, and manifest are sealed and reloaded in a hidden sibling stage, then exposed with one
no-replace directory rename. Same-intent retries converge on committed bytes; different intents
cannot overwrite. This prevents partial serving but supplies no model-quality evidence by itself.

Training runs consume immutable dataset manifests. Grouped splits prevent product-family or
hardware-generation leakage. MLflow records parameters and metrics, while the repository's
evaluation-artifact contract records dataset digest, synthetic-data policy, sample counts,
confidence intervals, and a self-hash.

## Compatibility and power behavior

Every rule declares required fields and returns one of four states. Absence is UNKNOWN. Examples:

- exact CPU/motherboard socket plus chipset/BIOS support;
- DDR generation, total capacity, and module count;
- motherboard/case and PSU/case form factors;
- GPU length/slot width and cooler height/radiator clearance;
- CPU cooler mounting sockets;
- PSU GPU/EPS connectors and estimated-load headroom; and
- storage interface and available motherboard slots.

Power is a configurable estimate of CPU peak, GPU board power, motherboard, memory, storage,
cooling, and fans, followed by 20% to 30% headroom. It is not a measured wall-power guarantee.

## Version and cache contract

Online logs and responses carry request/query ID, data version, ranking-model version,
compatibility-rule version, candidate counts, filter counts, solver status, warnings, and elapsed
time. Cache keys include all versions that can change a result. Updating one of those versions
invalidates older recommendation-cache entries; saved builds remain historical records and can be
explicitly re-run at current versions.

## Failure behavior

| Failure | Product behavior |
| --- | --- |
| Retailer unavailable | Use timestamped recent data only within freshness policy; disclose staleness. |
| Vector model unavailable | Fall back to release-bound BM25 and structured retrieval; report retrieval mode. |
| Ranker unavailable or incompatible | Use a deterministic documented baseline; never invent a model version. |
| Required compatibility data absent | Return UNKNOWN or infeasible explanation. |
| Solver timeout with incumbent | Return only independently revalidated feasible incumbents and the status. |
| No feasible build | Return zero builds plus specific binding constraints and possible relaxations. |
| Review evidence unavailable | Omit the review claim; core recommendation continues. |

## Security, privacy, and source compliance

Only permitted APIs, feeds, documents, or controlled imports may be acquired. Raw snapshots retain
access notes and are not automatically redistributed. Credentials are injected through the
environment and never committed. PostgreSQL's published port is for local development; a public
deployment must use a private network, rotated secrets, TLS at ingress, least-privilege database
roles, request limits, and restrictive CORS.

Interaction logs use anonymous session IDs unless a user account exists. Avoid logging natural
language, URLs containing tokens, or other personal data. Public build links expose only an
allow-listed build representation.

## Local and GPU operation

`scripts/dev.ps1 -Build -Detach` starts the default stack. Add `-WithDagster` and `-WithMlflow`
for the optional services. `scripts/test.ps1` uses the existing environment by default.

The lockfile intentionally resolves the standard CPU PyTorch wheel. On the Windows host,
`uv sync --locked --extra embeddings` installs the semantic-indexing dependencies, after which
`scripts/setup-gpu.ps1` replaces the standard wheel with `torch==2.13.0+cu130` from the official
PyTorch index and verifies `torch.cuda`, the device name, and compute capability. This CUDA wheel
has been verified on the local RTX 5070 Ti. Running `uv sync` later restores the locked CPU wheel,
so the GPU setup script must be rerun before host GPU training. This override is operational state,
not a lockfile change. The production API image installs the `serving` extra so it can load the
pinned local encoder on CPU while forcing Hugging Face/Transformers offline mode. Dagster remains
separately optional. The model tree is a mounted release artifact, never fetched at API startup.

LightGBM can use the GPU only when the installed build supports that device mode; training code
must detect support and record the actual device. Sentence-Transformers should select CUDA only
after `torch.cuda.is_available()` succeeds. Device choice does not relax grouped-split or frozen-
test-set requirements.

## Scale path

At the portfolio target of 3,000 products and 10,000 listings, PostgreSQL, bounded retrieval, and
one API process can remain adequate. Scale changes require evidence: query plans, connection
saturation, p95/p99 latency, memory, job duration, and failure recovery. Replicas, queues, or
service extraction are later options, not baseline architecture.

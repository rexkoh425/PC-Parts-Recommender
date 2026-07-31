# Database schema and migrations

PostgreSQL 16 with pgvector is the integration and deployment database. SQLite remains a
convenient unit-test fallback, but it does not prove PostgreSQL full-text or vector behavior.

From the repository root:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://pcbr:pcbr_local_only@localhost:5432/pc_build_recommender"
uv run alembic -c db/alembic.ini upgrade head
uv run alembic -c db/alembic.ini current
```

The initial migration mirrors the SQLAlchemy catalog/recommendation records and adds a
384-dimensional pgvector table for the initial MiniLM embedding model. A future change in
embedding dimensionality requires a new migration and a versioned re-embedding run; vectors
from different dimensions must not be mixed.

The versioned catalog/vector loader validates the normalized catalog against every ID-map row,
content hash, artifact SHA-256, matrix dimension, and L2 norm before it connects to PostgreSQL.
Validation can run without a database:

```powershell
uv run --no-sync python scripts/import_vector_catalog.py `
  --catalog data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/portfolio-3000/records.jsonl `
  --artifact-dir artifacts/retrieval/buildcores-embeddings `
  --verify-only
```

Remove `--verify-only` to migrate and idempotently import the catalog, source provenance, search
documents, and vectors. The live rows retain the data version, index version, embedding-model
name, encoder fingerprint, dataset content hash, and both artifact hashes. Online vector queries
must select those exact versions through `PgVectorSearchBackend`; category and active-product
filters are applied in the SQL cosine query.

`PostgresHybridRetriever` executes PostgreSQL cover-density full-text ranking and the exact
versioned pgvector cosine query against the same structured predicates. It fuses rank positions
with deterministic RRF (`k=60` by default), then reloads and rechecks only the fused IDs before
returning typed product documents. Budget, stock, brand, memory, VRAM, Wi-Fi, form-factor, and
scalar attribute constraints are pushed down before either source's top-k cutoff. PostgreSQL
`ts_rank_cd` is described as BM25-like lexical ranking, not mislabeled as native BM25.
Lexical retrieval, vector retrieval, and fused-document hydration share one read-only,
repeatable-read transaction so a concurrent price refresh cannot mix snapshots in one response.
The online vector backend requires the exact encoder fingerprint and dataset content hash as well
as model, data, and index versions, preventing a partially mixed index from being served.
pgvector 0.8.0 or newer is required: filtered vector queries enable strict-order iterative HNSW
scans so category and availability predicates cannot starve the requested top-k result set.

The importer memory-maps the float32 matrix and constructs SQL upsert values one bounded batch at
a time. `--batch-size` controls the peak bind payload; no list of all 3,000 Python vector lists is
created. Migration `20260722_0003` converts filter attributes to JSONB and adds partial listing
price/stock and exact serving-release indexes alongside the existing full-text GIN and HNSW cosine
indexes. Generic JSONB GIN indexes are intentionally deferred because the current scalar `#>>`,
cast, and range predicates cannot use them; targeted expression indexes require live `EXPLAIN`
evidence first.

Downgrade removes application tables but deliberately leaves the `vector` extension installed
because extensions can be shared by other schemas in the same PostgreSQL database.

# Infrastructure

## Local development

The default Compose profile starts PostgreSQL with pgvector, the FastAPI service, and the
Next.js application. Dagster and MLflow are optional so the core product does not pay their
startup cost during ordinary development.

```powershell
./scripts/dev.ps1 -Build -Detach
./scripts/dev.ps1 -Build -Detach -WithDagster -WithMlflow
```

Service URLs:

- Website: <http://localhost:3000>
- API: <http://localhost:8000>
- Dagster (optional): <http://localhost:3001>
- MLflow (optional): <http://localhost:5000>

The API container applies Alembic migrations before accepting traffic. Persistent Docker
volumes hold PostgreSQL, Dagster state, and MLflow artifacts. `docker compose down` preserves
them; `docker compose down --volumes` deletes them and should only be used when a clean local
database is intentional.

Dagster exposes manually launched `buildcores_import`, `benchmark_import`, and
`retailer_feed_import` jobs. The twelve-hour data-observability schedule is defined but must be
enabled deliberately. The governed-web `governed_web_retention_hourly` schedule is different: it
is `RUNNING` by default whenever the optional pipeline profile and its Dagster daemon are running.
Each retention run strictly derives every concrete governed-web source from the image's
`/app/config/source_registry.yaml`, copied from `data/source_registry.yaml` at build time, then
preflights all raw and processed receipt trees before any deletion. Rebuild and redeploy the
Dagster image whenever that registry changes. The retailer job refuses to run without both a
controlled CSV path and an explicit JSON consent policy in `RETAILER_FEED_CSV` and
`RETAILER_FEED_POLICY_JSON`.

`scripts/maintain_web_retention.py` is only a diagnostic or emergency operator surface; its
explicit `--source-name` values cannot prove complete registry coverage. The scheduled asset is the
production orchestration path. `WEB_RETENTION_MAXIMUM_ENTRIES` is one global planning/RAM bound
across the registry-derived source set, with a bounded pre-delete revalidation pass; it is not a
per-source quota. Raw-page v1 or processed-retention v1 receipts intentionally stop the run and
require a controlled, evidenced migration or purge rather than in-place mutation.

The passwords in `.env.example` are development placeholders. A shared or public deployment
must inject secrets and must not expose PostgreSQL directly to the public network.

Semantic embedding generation is intentionally excluded from the API and Dagster images so they
do not carry a multi-gigabyte accelerator runtime. Prepare a host indexing environment with:

```powershell
uv sync --locked --extra embeddings
./scripts/setup-gpu.ps1
```

The sync installs the locked standard PyTorch wheel; the second command reapplies and verifies the
CUDA 13.0 host override. Run semantic indexing/training on the host, then load its versioned vector
artifact into PostgreSQL. Ordinary API serving consumes stored vectors and remains dependency-light.

## Hardened single-host deployment

`docker-compose.production.yml` is a separate fail-closed contract. It requires immutable image
digests, six secret files, reviewed serving artifacts, HTTPS origins, production versions, private
networks, loopback operator ports, resource/PID limits, read-only application filesystems, bounded
logs, one-shot migrations, PostgreSQL-backed Dagster metadata, an explicit MLflow schema job, and
optional Prometheus/backup profiles.

Start by copying `.env.production.example` to the ignored `.env.production` file and replacing
every placeholder. This validation does not start Docker:

```powershell
python scripts/validate_production_env.py --env-file .env.production
```

Operational instructions and safety boundaries are in:

- [deployment runbook](../docs/deployment-runbook.md)
- [backup and restore policy](../docs/backup-and-restore.md)
- [observability guide](../docs/observability.md)

The configuration is not evidence of completed recovery, load, authorization, failover, or alert
delivery drills. In particular, neither an externally routed Dagster failure alert nor an
independent dead-man check for the hourly retention success is configured here, so unattended
retention is not production-ready. Keep the local `docker-compose.yml` for development only.

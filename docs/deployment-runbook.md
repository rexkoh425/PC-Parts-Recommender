# Production deployment runbook

Status: hardened single-host contract; runtime drills are still required
Last updated: 2026-07-22

This runbook operates `docker-compose.production.yml`. It is intentionally separate from the
local developer stack. The contract is appropriate for a controlled single-host portfolio
deployment behind an operator-managed TLS ingress. It is not a substitute for managed HA
PostgreSQL, multi-host orchestration, an identity provider, or tested disaster recovery.

## Release gates

Do not deploy unless all of the following are true:

1. CI quality, migration, production-contract, dependency-audit, secret, IaC, Dockerfile, and
   runtime-image scans pass or have a documented, time-bounded exception.
2. API, web, Dagster, MLflow, PostgreSQL/pgvector, exporters, Prometheus, and utility images are
   referenced by immutable `image@sha256:<digest>` values.
3. The web image was built with `NEXT_PUBLIC_API_URL` equal to `PCBR_PUBLIC_API_URL`. Next.js
   public variables are compiled into the browser bundle and cannot be corrected by changing only
   the container environment.
4. The exact BuildCores products, governed retailer offers, reviewed mapping manifest,
   review-evidence JSONL, read-only catalogue-readiness report, local semantic-encoder bundle, and
   version-3 serving manifest were frozen together. The manifest must bind the ER model/calibrator, serving evidence,
   human-labelled v2 evaluation, matcher/catalogue policy, and operator-approved rights record.
   The report's data version equals `PCBR_API_DATA_VERSION`, records `production_ready=true` with
   no blockers, and every model/rule/solver version and content hash is in the release ticket.
   `demo`, `development`, `untrained`, and unknown versions are rejected by preflight.
   Review evidence must be cited and rights-checked for active Singapore display, cache, history,
   and derivation; use a pinned explicit empty JSONL when none is permitted. Research-only crawls
   and offers without the same active rights remain ineligible for this artifact.
5. A PostgreSQL and artifact backup completed before any schema or model change, and the most
   recent restore drill is within the operator's accepted window.
6. The current processed catalogue is capable of the intended product behavior. The repository's
   checked-in data snapshot currently has insufficient consented in-stock retailer coverage for a
   real complete build, even though the deployment machinery is hardened.

The manual `release-images` workflow builds the five project-owned images, compiles the selected
HTTPS API origin into the web bundle, pushes to GHCR, emits SBOM/provenance attestations, scans the
candidate before publication, and prints digest references for the production env file. Branch protection
must require the ordinary CI/security workflows before an operator approves a release. Third-party
PostgreSQL, Prometheus, blackbox, and utility image digests are selected and scanned separately.

## Trust boundary

- Only the web and API ports bind to loopback. Dagster, MLflow, and Prometheus also bind to
  loopback only when their profiles are enabled. PostgreSQL has no published host port.
- A separate reverse proxy or load balancer terminates TLS. It must authenticate Dagster, MLflow,
  and Prometheus with OIDC/SSO, rate-limit the API, set request-size limits, and deny public access
  to `/metrics`, `/docs`, `/redoc`, and `/openapi.json`.
- The API independently caps request bodies with `PCBR_API_MAX_REQUEST_BODY_BYTES`. Build
  generation admits at most `PCBR_API_BUILD_GENERATION_MAX_CONCURRENCY` executions and
  `PCBR_API_BUILD_GENERATION_MAX_QUEUE_SIZE` waiters per API process. A full queue returns `429`;
  an admitted waiter that exceeds `PCBR_API_BUILD_GENERATION_QUEUE_TIMEOUT_SECONDS` returns `503`.
- Public build links require the migrated durable database. `POST /v1/builds/{build_id}/shares`
  stores an immutable allow-listed snapshot (not the original request, listing URLs, ownership
  state, or internal IDs). Keep the returned revocation token outside logs and analytics; it is
  shown only at creation and is required by the revoke endpoint. Set
  `PCBR_API_BUILD_SHARE_TTL_HOURS` deliberately and run the `20260723_0007` migration before
  enabling the endpoint.
- The read-only operations route, `GET /v1/admin/operations`, reads its token only from the mounted
  `PCBR_API_ADMIN_TOKEN_FILE` secret and requires that value in `X-PCBR-Admin-Token`. The browser
  operations page holds the token only in page memory and is intentionally `noindex`; do not put
  the value in a `NEXT_PUBLIC_` setting, logs, analytics, URLs, or this repository. This endpoint
  reports aggregate mapping, price-freshness, missing-field, release-blocker, and bounded
  instrumented-pipeline evidence only. Pipeline user-code writes separate, aggregate-safe
  receipts to `PCBR_PIPELINE_OPERATIONS_DIR`; the API receives only that directory on a read-only
  mount. A missing or invalid mount is reported as unavailable, never as healthy. These receipts
  do not replace Dagster's authenticated run store: scheduler, queue, and worker-control failures
  remain there.
  Both overload responses include `Retry-After`. Keep ingress limits at least as strict where
  practical, and tune these values only from measured load-test and solver-duration evidence.
- The API container has only the application database credential. The migration job, Dagster,
  MLflow, monitoring, and PostgreSQL bootstrap each have separate credentials.
- The `pipeline` network permits outbound source access. Source adapters still enforce the source
  registry, consent, licence, provenance, and rate-limit contracts.
- Docker control sockets are never mounted into application or monitoring containers.

## Prepare the host

Create directories outside the Git checkout for serving data, pipeline state, pipeline-operation
receipts, artifacts, MLflow
artifacts, backups, and seven secret files. Use randomly generated values of at least 24
characters. Do not put a password on a command line or in shell history. On Windows, restrict each
file ACL to the deployment identity and administrators; on Linux, set mode `0600`.

Copy `.env.production.example` to the ignored `.env.production` file and replace every
`CHANGE_ME` value. `PCBR_SERVING_RELEASE_DIR` must contain the pinned manifest and every referenced
artifact; the semantic encoder must be in its content-addressed
`encoders/<PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256>` directory. Release images must use registry
digests, not tags. Build that encoder directory before assembling the serving release by running
`scripts/package_semantic_encoder_bundle.py` with the exact `model_name`, `model_revision`, and
embedding manifest that the release will pin. The command copies a verified local model snapshot,
records its Apache-2.0 provenance, and publishes an immutable digest-named directory; it never
downloads weights or uses a mutable model name at startup. Then run:

```powershell
python scripts/validate_production_env.py --env-file .env.production
docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile pipeline --profile mlops --profile observability `
  --profile operations --profile restore config --quiet
```

The first command checks required values, secret length and presence, file/directory existence,
release-artifact schemas and data-version parity, a positive readiness decision with no blockers,
loopback bindings, HTTPS origins, restrictive CORS, immutable versions, image digests, and obvious
clear-text credentials. The deployment job still recomputes all catalogue and rights evidence;
preflight never substitutes for that gate. On Windows it cannot prove that an ACL is private, so
that remains an operator check.

`PCBR_GOVERNED_OFFERS_FILE` is the canonical offer-artifact setting. The deprecated
`PCBR_DYNACORE_OFFERS_FILE` fallback remains accepted for migration, but the canonical value wins
when both are present.

The same fail-fast operations are available through `scripts/production.ps1`. It never creates a
production env file and deliberately has no restore action:

```powershell
./scripts/production.ps1 -Action Validate
./scripts/production.ps1 -Action Config
./scripts/production.ps1 -Action DeployCore
./scripts/production.ps1 -Action Status
```

The bind-mounted data and artifact directories must be writable by the documented container UIDs:
API `10001`, Dagster `10002`, MLflow `10003`, and PostgreSQL exporter `10004` where applicable.
Serving catalogue files remain read-only.

## Initial database bootstrap

The PostgreSQL initialization script runs only when `postgres-data` is empty. It creates separate
application, migration, Dagster, MLflow, and monitoring roles/databases; installs pgvector and
`pg_stat_statements`; and sets default application grants for objects created by the migration
role.

```powershell
docker compose --env-file .env.production -f docker-compose.production.yml up -d postgres
docker compose --env-file .env.production -f docker-compose.production.yml ps postgres
docker compose --env-file .env.production -f docker-compose.production.yml logs --tail 200 postgres
```

If a volume already exists, changing secret files does not rotate database passwords. Follow the
rotation procedure below. Never delete a named volume merely to replay initialization.

## Deploy a release

Run the schema task, then the one-shot catalogue release task, inspect both, and only then start
the serving processes:

```powershell
docker compose --env-file .env.production -f docker-compose.production.yml up migrate
docker compose --env-file .env.production -f docker-compose.production.yml ps -a migrate
docker compose --env-file .env.production -f docker-compose.production.yml logs migrate

docker compose --env-file .env.production -f docker-compose.production.yml up catalog-release
docker compose --env-file .env.production -f docker-compose.production.yml ps -a catalog-release
docker compose --env-file .env.production -f docker-compose.production.yml logs catalog-release

docker compose --env-file .env.production -f docker-compose.production.yml up -d api web
docker compose --env-file .env.production -f docker-compose.production.yml ps
```

`migrate` and `catalog-release` use the schema-owning role; the latter does not run migrations.
The catalogue task idempotently imports the exact mounted product, offer, reviewed-mapping, and
review-evidence artifacts plus the manifest-bound ER release. Before database mutation it validates
the manifest-pinned vector matrix, ID map, catalogue-derived search documents, and embedding
identity; recomputes production readiness and current data-use rights; and requires an exact match
to the frozen read-only report and configured data version. It then upserts the processed rows,
imports the pinned vectors/search-document hashes, and verifies the exact product/listing sets and
row hashes in PostgreSQL. Any stale canonical product, listing, or vector provenance fails closed;
the release task performs no stale-row deletion. Reconciliation requires a separate audited
operator workflow, which is not currently implemented. Any failure keeps the job unsuccessful and
Compose cannot start the API. The API independently loads
the same manifest and ER binding, verifies the local semantic-encoder bundle before an offline
warm-up, constructs a release-bound BM25 index from the exact validated embedding search
documents, and repeats readiness, schema, and database/file identity checks with its separate
non-DDL role. BM25 startup fails closed above 50,000 documents or 64 MiB of source text; request-
time database predicates still decide the active, price, stock, brand, and specification-eligible
IDs. Keep the API at one
worker per container: its lightweight Prometheus counters are process-local and the current
catalogue is loaded into each worker's memory. Scale with multiple containers only after an
ingress and per-instance metrics discovery have been tested.

Enable optional control planes deliberately:

```powershell
docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile pipeline up -d dagster-code dagster-webserver dagster-daemon

docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile mlops up mlflow-migrate
docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile mlops up -d mlflow

docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile observability up -d postgres-exporter blackbox-exporter prometheus
```

Dagster uses a PostgreSQL metadata store, one queued run by default, a separate gRPC user-code
process, a daemon, and a webserver. MLflow has its own database and an explicit schema upgrade job.
Both UIs require authentication at ingress; their loopback bindings are not authentication.

## Verify the release

From the deployment host:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
Invoke-RestMethod http://127.0.0.1:8000/v1/system/freshness
Invoke-WebRequest http://127.0.0.1:3000 -UseBasicParsing
```

For enabled profiles, verify Dagster `/server_info`, MLflow `/health`, and Prometheus `/-/ready`.
Then verify the public HTTPS origins through ingress. Confirm:

- health and application responses carry the expected data, model, rule, and solver versions;
- `/metrics` is scrapeable from Prometheus but not from the public internet;
- browser requests use the production API origin, not localhost;
- PostgreSQL exporter uses only `pg_monitor` privileges;
- the Dagster daemon is live and the code location loads; and
- a bounded synthetic request exercises the full route without saving or publishing a build that
  violates the real stock/data policy.

Record timestamps, image digests, database revision, data/model/rule versions, health output, and
the deployment ID. Passing health probes is engineering evidence, not model-quality evidence.

## Configuration and secret rotation

Generate a new secret file without replacing the old one. As PostgreSQL administrator, rotate one
role at a time with a parameterized `ALTER ROLE`, update the matching Compose secret path, and
restart only its consumers. Verify health before rotating the next role. Rotate the administrator
last. Never paste a secret into a checked-in SQL file, process argument, ticket, or log.

Changing the bootstrap secret file alone has no effect after initialization. Changing application
versions or catalogue paths requires rerunning `migrate` and `catalog-release` before restarting
the API. Changing `NEXT_PUBLIC_API_URL` requires a new web image.

## Upgrade and rollback

Before upgrade:

1. run both backups and copy them to encrypted off-host storage;
2. review Alembic and MLflow migrations for lock time and backward compatibility;
3. verify free disk, memory, and database connections;
4. record current image digests and schema revisions; and
5. rehearse on a restored copy of production data.

Application rollback is safe only while the old image supports the current schema. Restore the
previous immutable image digests, run the production preflight, and restart API/web. Never run an
Alembic downgrade automatically in production. For an incompatible or destructive migration,
stop writers and use the tested database restore procedure.

For performance or ranking models that are not catalogue-identity inputs,
rollback changes the reviewed model/version mapping to the last compatible
artifact, reruns production preflight, and restarts API containers. An
entity-resolution or catalogue-binding rollback additionally requires a newly
pinned full serving manifest, successful preflight, and a rerun of
`catalog-release` before the API restart. Keep the data, feature, embedding,
model, and compatibility-rule contract together; changing only a label is not
a rollback.

## Ranker publication-stage maintenance

LambdaMART training publishes `ranker-artifact` through an atomic directory rename. An abrupt
process exit before that rename can leave a hidden sibling stage. Inspect one explicit artifact
parent and bundle name first; the command is dry-run unless `--apply` is present:

```powershell
.\.venv\Scripts\python.exe scripts\maintain_ranker_publication_stages.py `
  --parent 'D:\pcbr\artifacts\ranking\ltr-v4' `
  --bundle-name ranker-artifact `
  --minimum-age-hours 24
```

Review every reported stage, then repeat the exact command with `--apply`. The maintenance command
refuses relative, root, home, link-traversing, or unbounded scopes; preserves new and actively
locked stages; and never selects the final bundle. Do not run filesystem cleanup tools against the
artifact parent as a substitute for this bounded command.

## Incident response

1. Preserve evidence: deployment ID, request IDs, image digests, versions, UTC timestamps, probe
   output, recent bounded logs, and the affected query/build identifiers.
2. Stop the unsafe edge path. Remove public ingress or stop API/web if compatibility, data
   provenance, authorization, or secret exposure is suspected.
3. Do not delete containers, volumes, raw snapshots, or MLflow artifacts while diagnosing.
4. Check Prometheus alerts, API JSON logs, PostgreSQL slow/error logs, Dagster run state, MLflow
   backend health, disk, memory, and secret expiry.
5. Roll back immutable images only when schema compatibility is known. Otherwise restore into a
   separate environment first.
6. Independently recheck any build returned during the affected rule/data interval.

Alert rules point to this section, but no Alertmanager destination is configured in the repository.
Connecting paging or ticket routing is a deployment-owner responsibility and a blocker for
unattended operation.

## Stop without destroying state

```powershell
docker compose --env-file .env.production -f docker-compose.production.yml `
  --profile pipeline --profile mlops --profile observability stop
```

Do not use `down --volumes` in production. Volume deletion and database restore are separate,
explicitly authorized disaster-recovery actions.

## Production-readiness evidence still required

Before calling the deployment production-ready, complete and retain: backup restore drills,
dependency-loss and restart tests, ingress authorization tests, secret rotation, load tests with
p50/p95/p99 and resource saturation, slow migration rehearsal, alert delivery, data freshness
failure tests, and a rollback drill. None has been inferred from static configuration.

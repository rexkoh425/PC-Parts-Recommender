# Observability and operational signals

Status: baseline instrumentation and alerts; a bounded Locust harness exists, but external routing
and qualifying production-load evidence are pending
Last updated: 2026-07-23

## Logs

The API writes one JSON object per request with UTC timestamp, severity, request ID, method, route,
status, duration, data version, ranker version, compatibility-rule version, and solver version.
Exception logs include the error type and stack. Do not add raw natural-language queries,
credentials, token-bearing URLs, or unrestricted request bodies.

PostgreSQL writes a stable key/value prefix containing time, process, user, database, application,
and client, plus slow statements above the configured threshold, lock waits, connections, and
disconnections. Dagster and MLflow write to stdout/stderr. Every production container uses bounded
Docker log rotation (`max-size` and `max-file`) and a service/container tag.

For retention and search, forward stdout/stderr to an operator-owned log platform over an
authenticated channel. The repository does not mount the Docker socket into a collector and does
not prescribe a vendor. Redact at source and grant operators least-privilege access.

## Metrics

The API endpoint `/metrics` exposes:

- `pcbr_http_requests_total` by method, route template, and status;
- `pcbr_http_request_duration_seconds` histogram by method and route template; and
- `pcbr_http_requests_in_progress`.

It also exposes bounded recommendation-domain counters and gauges:

- build-generation outcome, returned-build count, solver/profile outcome, validator rejection,
  and successful component-replacement counters;
- successful product-search result, empty-result, candidate-funnel (`retrieved`, category-filtered,
  brand-filtered, and ranked-before-pagination), and authoritative compatibility-filter counters;
- performance-signal provenance by observed/predicted/relative/insufficient-data basis, confidence,
  and a closed decision code; fallback counters distinguish an unpromoted model from an input outside
  its training contract without exposing raw feature values;
- compatibility request and individual rule-result counters, including `UNKNOWN` outcomes;
- accepted interaction events by the closed event-type enum; and
- freshness observations plus latest product, listing, release-blocker, and immutable-release
  startup-verification gauges; and
- latest authenticated admin-operations observations: explicit mapping/receipt availability,
  unresolved manual-review and mapping-outcome counts, aggregate missing critical-field values,
  and bounded instrumented-pipeline failure evidence.

These metrics intentionally omit raw query text, product/build/request identifiers, model/data
versions, retailer URLs, and human-readable failure messages. Their labels are closed enums rather
than client-controlled values, so they cannot create unbounded process-local time series.

Operations gauges are refreshed from the same validated aggregate snapshot used by authenticated
`GET /v1/admin/operations`; private `/metrics` scrapes refresh it internally without exposing the
admin response body. They deliberately disappear when the corresponding evidence is unavailable
instead of retaining stale values. They are not a replacement for Dagster's authenticated scheduler,
queue, and worker control plane.

Raw URL paths are never labels, preventing product/build/request IDs from creating unbounded
cardinality. The endpoint excludes its own scrape requests. Metrics are process-local, so the
single-host contract intentionally uses one API worker per container. A future multi-replica
deployment must discover/scrape every replica or adopt a multiprocess-capable telemetry backend.

The `observability` profile adds:

- PostgreSQL exporter using a dedicated `pg_monitor` role;
- blackbox probes for API liveness/readiness, web, and optional Dagster/MLflow endpoints; and
- Prometheus with 15-day local retention and checked-in alert rules.

Prometheus binds to loopback and is not an authentication boundary. Ingress must require operator
SSO. Optional Dagster/MLflow probes are visible but intentionally excluded from core availability
alerts when those profiles are disabled.

## Initial alerts

`infra/monitoring/alerts.yml` covers core endpoint failure, PostgreSQL exporter failure, API 5xx
ratio above 2%, build-generation p95 above the 2.5-second target, sustained concurrency, high
database connection use, deadlocks, unavailable mapping or pipeline receipt evidence, missing
critical catalogue fields, and bounded instrumented-pipeline failures or invalid/truncated
receipts. The zero-tolerance data-evidence alerts express release-safety conditions; the latency
and capacity thresholds are initial operating hypotheses, not measured SLO achievement.

No Alertmanager or notification destination is checked in because routing identities and secrets
belong to the deployment environment. Unattended operation is blocked until alert delivery,
deduplication, escalation, silence expiry, and a synthetic failure have been tested.

The default-running `governed_web_retention_hourly` and `wdc_research_retention_daily` Dagster
schedules log structured asset results and fail on receipt, confinement, work-budget, or deletion
errors. Their local run state is not an external page. A deployment must route Dagster run failures
and add independent dead-man checks when no successful materialization is observed within the
governed-web 60-minute interval or the WDC daily interval. Exercise both paths with a synthetic
failure and a paused schedule before authorizing unattended acquisition or retention.

## Health semantics

- Liveness answers whether a process can serve HTTP and must remain dependency-light.
- API readiness checks the loaded catalogue and returns immutable data/model/rule/solver versions.
- PostgreSQL health uses `pg_isready`; the exporter provides a separate monitoring-role view.
- Dagster webserver, daemon, and gRPC code location have separate probes.
- MLflow `/health` checks the server process. Release verification must additionally create/read a
  bounded test artifact in a non-production experiment to prove backend/artifact-store access.
- External synthetic checks should exercise a safe build request and verify versions and
  compatibility without publishing a result.

HTTP 200 alone cannot prove catalogue freshness, model correctness, rule safety, database
durability, or artifact recoverability. Those require explicit assertions and drills.

## Remaining high-value signals

The operations surface now records aggregate missing-critical-field values, entity-resolution
review-queue depth, and bounded instrumented pipeline failures; private metrics scrapes refresh
the aggregate snapshot and conservative evidence alerts are checked in. Before a serious public
launch, test delivery, escalation, silence expiry, and an intentionally failed source run, then
set a review-queue SLO only after measuring a representative operating baseline. Retain Dagster
control-plane dead-man checks for scheduler, queue, and worker failures. Release-artifact
verification is a startup verdict for the immutable serving release, not a claim of continuous
on-disk integrity after the process starts. The performance fallback decision is intentionally
coarse: it distinguishes a model-promotion gate from an input outside its training contract, but
never emits raw feature values. Add traces only after a privacy/cardinality review.

The checked-in [Locust harness](load-testing.md) now enforces a reviewed request profile, loopback
or explicitly acknowledged HTTPS target, and bounded endpoint labels. It can produce development
smoke evidence, but it does not make any latency target achieved: a qualifying run still needs a
pinned serving release and the declared profile, hardware, resource, database, cache, candidate,
and raw-output evidence. Measure p50/p95/p99 latency, error rate, throughput, CPU, memory,
database pool saturation, index latency, and optimiser duration before claiming latency targets.

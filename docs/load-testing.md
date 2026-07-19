# Load testing

The repository uses [Locust](https://locust.io/) for HTTP load generation. The checked-in
development profile exercises product search and build generation, retaining distinct endpoint
metrics rather than collapsing them into one average.

This harness is deliberately bounded:

- it accepts only a reviewed JSON request profile for the search and build endpoints, with finite
  JSON bodies, status expectations, and task weights;
- it permits plain HTTP only on loopback targets;
- a remote target must use HTTPS and set the exact `PCBR_LOAD_CONFIRM` acknowledgement;
- Locust metric names are fixed to the HTTP method and path, so query text, IDs, and request bodies
  never become metric labels; and
- the default profile declares `development_only`, so its measurements cannot support production
  latency or résumé claims.

## Development run

Use two PowerShell terminals. In the first, start the development-only API:

```powershell
$env:PCBR_API_ENVIRONMENT = "development"
uv run --no-sync uvicorn services.api.main:app --host 127.0.0.1 --port 8000
```

In the second, preflight the host memory cap, then run a small bounded Locust test. `uvx` keeps
Locust out of the project serving environment and pins the runner version for this command.

```powershell
.\scripts\check-memory-cap.ps1 -MaxUsedGb 55
$env:PCBR_LOAD_BASE_URL = "http://127.0.0.1:8000"
$env:PCBR_LOAD_PROFILE_FILE = "scripts/loadtest/development-profile.json"
uvx --from "locust==2.44.4" locust -f scripts/loadtest/locustfile.py `
  --headless --users 2 --spawn-rate 1 --run-time 30s `
  --csv artifacts/evaluation/load/development-demo-api-mix
```

Retain all four Locust CSV files (`*_stats.csv`, `*_stats_history.csv`, `*_failures.csv`, and
`*_exceptions.csv`) together with the exact profile SHA-256 printed at test start. Record host
hardware, OS, container limits, API mode, catalogue/listing counts, data/model/rule/solver
versions, cache state, database state, candidate caps, user count, spawn rate, warm-up, run
duration, and client-side resource use beside the CSV files.

While the target API is still running, turn that raw output into one immutable evidence record:

```powershell
uv run --no-sync python scripts/summarize_load_test.py `
  --profile scripts/loadtest/development-profile.json `
  --csv-prefix artifacts/evaluation/load/development-demo-api-mix/locust `
  --target-origin http://127.0.0.1:8000 `
  --output artifacts/evaluation/load/development-demo-api-mix/summary.json `
  --users 2 --spawn-rate 1 --run-time-seconds 30 --warmup-seconds 0 `
  --cache-state cold --database-state in_memory_demo
```

The summarizer reloads the reviewed profile, hashes all four Locust CSV files, requires one
non-zero result row for each declared endpoint, captures bounded `/health/ready` and
`/v1/system/freshness` release metadata from the same origin, records non-identifying host capacity,
and writes a no-overwrite content-addressed JSON record. It does not retain request bodies, query
text, IDs, credentials, hostnames, or raw API responses. A development-only profile remains a
development-only measurement even if its p95s are below the product targets.

## Remote and production-candidate runs

Load testing a remote environment is a deliberate operational action. Obtain the deployment
owner's approval, use a production-candidate profile tied to the immutable serving release, and
set the acknowledgement only for that approved execution:

```powershell
$env:PCBR_LOAD_BASE_URL = "https://approved.example.com"
$env:PCBR_LOAD_CONFIRM = "I_UNDERSTAND_THIS_GENERATES_LOAD"
```

Do not point the development profile at a public deployment. A measured production result still
requires rights-cleared market data, a pinned serving release, declared hardware and load profile,
and retained raw outputs before updating the evaluation report.

# Deployment: Next.js on Vercel, FastAPI on Render

Status: the controlled public demonstration is deployable. A real retailer
catalogue stays fail-closed until a qualifying authorised release exists.

## Why the services are split

The API cannot run as a Vercel serverless function. Its locked production
dependencies measure roughly **356 MB unzipped** against Vercel's **250 MB**
limit:

| Package | Size |
| --- | ---: |
| scipy | 93 MB |
| ortools | 80 MB |
| pandas | 43 MB |
| numpy (with `numpy.libs`) | 43 MB |

Lazy imports do not help: `services.api.main` loads ortools, scipy, pandas,
scikit-learn, lightgbm, numpy, and SQLAlchemy at application start, because the
optimiser and ranking stack is wired into the application boundary.

The web application therefore deploys to Vercel as an ordinary Next.js project
and the API runs as a container. This needs no Vercel beta capability.

## 1. Deploy the API (Render)

`render.yaml` is a blueprint for `infra/api.Dockerfile`.

1. Render → **New → Blueprint** → select this repository.
2. Render reads `render.yaml` and prompts for the two `sync: false` values:

| Variable | Value |
| --- | --- |
| `PCBR_API_CORS_ORIGINS` | `["https://YOUR-APP.vercel.app"]` — a JSON array of exact web origins |
| `PCBR_API_IMPRESSION_SIGNING_KEY` | a random secret of at least 32 bytes |

3. Deploy, then confirm `https://YOUR-API.onrender.com/health/ready` returns
   `{"status":"ready",...}`.

`PCBR_API_SERVICE_MODE=public_demo` is set in the blueprint. In that mode the
API serves a deterministic in-memory fixture, so the container needs no
database and the entrypoint skips migrations. Do not set `DATABASE_URL`,
retailer file paths, or source-policy keys for the demo.

Render's free instances sleep when idle and cold-start slowly. Use a paid
instance if the demo must respond immediately.

Any container host works. Fly.io (`fly launch --dockerfile
infra/api.Dockerfile`) or Railway are equivalent; only the CORS origin and the
web app's `NEXT_PUBLIC_API_URL` need to agree.

## 2. Deploy the web application (Vercel)

1. Vercel → **Import** this repository. Keep the project root at the repository
   root; `vercel.json` builds the `apps/web` workspace.
2. Framework preset: **Next.js**.
3. Environment variables:

| Variable | Value |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | the API's HTTPS origin, e.g. `https://YOUR-API.onrender.com` |
| `NEXT_PUBLIC_DATA_MODE` | `api` |
| `NEXT_PUBLIC_SITE_URL` | the deployment's own HTTPS origin |

4. Deploy a preview, exercise a build generation, then promote.

The browser calls the API cross-origin with credentials, so
`PCBR_API_CORS_ORIGINS` must name the exact Vercel origin. A wildcard is
rejected by the browser and by `validate_http_exposure`.

## Web-only fallback

Setting `NEXT_PUBLIC_DATA_MODE=demo` and omitting `NEXT_PUBLIC_API_URL` serves
`apps/web/lib/demo-api.ts` fixtures entirely in the browser, with no API at all.
Build generation and browsing work; re-optimisation, sharing, and the admin
views are unavailable.

## Real catalogue release

Switching to `PCBR_API_SERVICE_MODE=processed_catalog` is not a configuration
change. The API refuses to start until it receives a sealed serving release,
approved retailer source data, rights evidence, a semantic-encoder bundle, and
durable PostgreSQL storage. None of those authorised inputs exist in this
repository. That mode also requires the database wiring the demo path skips, so
`DATABASE_URL` or the `PCBR_DATABASE_*` variables become mandatory again.

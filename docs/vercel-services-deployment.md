# Vercel deployment

Status: controlled public-demo deployment is supported; a real retailer catalogue remains
fail-closed until a qualifying, authorised release exists.

This repository is configured for one Vercel project with two Services:

- `web` builds `apps/web` as Next.js and handles the site routes;
- `api` loads `services.api.main:app` as FastAPI; and
- Vercel routes `/v1/*` and `/health*` to the API while the browser uses the same origin.

The route arrangement is intentional. FastAPI already owns `/v1/*` and `/health*`; it does not
need to be changed to include an extra deployment-only prefix. The web build must use
`NEXT_PUBLIC_API_URL=/`, which resolves browser API requests to the same deployment origin.

## Create the Vercel project

1. The repository is published at `github.com/rexkoh425/PC-Parts-Recommender`, which is the remote
   Vercel imports from.
2. In Vercel, import the repository and set its **Framework Preset** to **Services**. Services is
   currently a Vercel beta capability; request access if it is not available in the selector.
3. Keep the project root at the repository root so Vercel can read `vercel.json` and both service
   directories.
4. Add the environment variables below for the relevant environments, then deploy a preview before
   promoting it.

For local parity, install the Vercel CLI and run `vercel dev -L` from the repository root. The
`-L` flag runs the services locally without first authenticating to Vercel.

## Controlled public demo

Use these Vercel environment variables to deploy the demonstrator. It exposes only deterministic
sample products/builds and must remain visibly labelled as illustrative. It is not a retailer-data
release and does not make price, stock, ranking-quality, or production-compatibility claims.

| Variable | Value |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | `/` |
| `NEXT_PUBLIC_DATA_MODE` | `api` |
| `NEXT_PUBLIC_SITE_URL` | the deployment's own HTTPS origin, e.g. `https://YOUR_DOMAIN` |
| `PCBR_API_ENVIRONMENT` | `production` |
| `PCBR_API_SERVICE_MODE` | `public_demo` |
| `PCBR_API_DOCS_ENABLED` | `false` |
| `PCBR_API_CORS_ORIGINS` | `["https://YOUR_DOMAIN"]` |
| `PCBR_API_IMPRESSION_SIGNING_KEY` | a distinct random secret of at least 32 bytes |

Use the actual Vercel production or custom domain in the CORS value. Same-origin browser calls do
not depend on CORS, but a restrictive value keeps the API configuration safe if an external caller
is introduced later. Do not configure `DATABASE_URL`, retailer file paths, source-policy keys, or
retailer credentials for this demo mode. Keep the impression-signing key in Vercel's encrypted
environment variables; it is required to prevent a browser from forging trusted interaction
attribution.

## Real catalogue release

Switching to `PCBR_API_SERVICE_MODE=processed_catalog` is not a configuration-only change. The API
will correctly refuse to start until it receives a sealed serving release, approved retailer source
data, rights evidence, a semantic-encoder bundle, durable PostgreSQL storage, and the remaining
production inputs. The current repository has none of those authorised retailer inputs.

Vercel Functions have ephemeral filesystems. Do not package a retailer feed, a signed policy, a
database dump, or a model bundle into Git or a public function deployment. The next production
phase must add a private release-artifact materialisation mechanism and a managed PostgreSQL/
pgvector database, then prove cold-start time, bundle size, and solver latency within Vercel's
limits before routing production traffic.

## If Services is unavailable

Deploy two Vercel projects from the same repository instead:

1. Create a **web** project with root directory `apps/web`.
2. Create an **api** project with root directory `.` and FastAPI entrypoint
   `services.api.main:app`.
3. Set the web project's `NEXT_PUBLIC_API_URL` to the API project's HTTPS URL and set
   `PCBR_API_CORS_ORIGINS` in the API project to the web URL.

This fallback is still entirely on Vercel, but the services use separate domains and require CORS.

# Windows development and evidence guide

Last verified: 2026-07-23
Supported host baseline: Windows 11, PowerShell 7, Python 3.12, Node.js 24

This guide describes the operational path from a clean checkout to local services, source
manifests, model artifacts, and evaluation evidence. It intentionally distinguishes an
engineering smoke test from an artifact that is eligible for promotion or a public claim.

## 1. Install and verify prerequisites

Install Python 3.12, `uv`, Node.js 24, npm, Git, and Docker Desktop. Confirm the versions from the
repository root:

```powershell
python --version
uv --version
node --version
npm --version
docker compose version
```

Python 3.13 is outside the current package contract. Node 24 is the repository baseline even if a
newer host version happens to build successfully.

Create the locked base environments:

```powershell
uv sync --locked --extra modeling
npm install
```

Before Docker, embedding, model, or large-data work, enforce the host RAM cap:

```powershell
./scripts/check-memory-cap.ps1 -MaxUsedGb 55
```

The script reports used, free, total, and cap values as JSON. It exits `1` when used memory is at
or above 55 GB. Treat a nonzero exit as a hard stop: do not start Docker services, source
materialization, model training, or another memory-intensive process until usage falls below the
cap.

Semantic indexing is optional for the API/test baseline. Install it before embedding work:

```powershell
uv sync --locked --extra embeddings
```

Do not commit `.env`, raw source files, processed source rows, MLflow state, or credentials.

## 2. Run the application

### Docker Compose

Start Docker Desktop's Linux engine, then run:

```powershell
./scripts/check-memory-cap.ps1 -MaxUsedGb 55
./scripts/dev.ps1 -Build -Detach
```

`dev.ps1` creates `.env` from `.env.example` only when the file is absent, so it does not replace
an existing local configuration. Review the generated development-only credentials.

The default stack is PostgreSQL/pgvector, FastAPI, and Next.js. Optional services:

```powershell
./scripts/dev.ps1 -Build -Detach -WithDagster -WithMlflow
```

Local ports are web 3000, API 8000, Dagster 3001, MLflow 5000, and PostgreSQL 5432. The defaults
in `.env.example` are development-only credentials and must be replaced before any shared
deployment.

Useful checks:

```powershell
docker compose --env-file .env ps
Invoke-RestMethod http://localhost:8000/health/ready
Invoke-RestMethod http://localhost:8000/v1/system/freshness
```

Stop the stack without deleting its named volumes:

```powershell
./scripts/dev.ps1 -Down
```

### Host processes

Use separate terminals:

```powershell
uv run --no-sync uvicorn services.api.main:app --reload --port 8000
```

```powershell
npm run dev:web
```

The API contract is visible at <http://localhost:8000/docs>. Responses expose request, data,
ranking, compatibility-rule, and solver version headers. Development fixtures and fallback
rankers must retain names such as `demo`, `baseline`, or `untrained`; they are not trained-model
evidence.

### Runtime data modes

`PCBR_API_SERVICE_MODE=demo` is the default development contract fixture. Startup rejects demo
mode in non-development environments. The processed-catalog mode requires explicit source paths:

```powershell
$env:PCBR_API_SERVICE_MODE = "processed_catalog"
$env:PCBR_API_BUILDCORES_CATALOG_PATH = (Resolve-Path "data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/portfolio-3000/records.jsonl").Path
$env:PCBR_API_GOVERNED_OFFERS_PATH = (Resolve-Path "data/processed/dynacore_controlled_pdf/6e243d7bf1cba090f529b09a9276fac03fedddcadb8c11cf9ce7ec1e674bb9ba/records.jsonl").Path
uv run --no-sync uvicorn services.api.main:app --reload --port 8000
```

`PCBR_API_REVIEWED_MAPPING_PATH` may point to a separately reviewed listing/product mapping
manifest. Without it, the conservative importer accepts only exact conflict-free mappings.

At the current snapshot, processed mode loads 3,000 BuildCores products and 485 controlled
Dynacore offers but maps only two offers, asserts zero in-stock listings, and therefore cannot
produce a complete priced build. Search remains useful with `in_stock_only=false`; generation
returns an explicit infeasibility result. Do not weaken stock or matching constraints to make a
demo appear feasible.

Non-development processed mode is a different, fail-closed contract. Import and API startup both
load the same operator-pinned `pc-build-recommender.serving-release.v4` manifest and exact
catalogue/offers/reviewed-mappings/review-evidence bytes. The review artifact must be explicitly
empty or contain bounded cited evidence with active Singapore display, cache, history, and
derivation rights. That manifest must bind a promoted LightGBM entity-resolution model with fitted
calibrator, human-labelled v2 evaluation, threshold/readiness policy, active Singapore rights
approval, and every identity digest. Direct model/evaluation CLI arguments or legacy eligibility
booleans cannot authorize production. No genuine promoted release is shipped.
Version 4 additionally pins and independently verifies the signed source-batch manifest, raw
snapshot, rejection stream, externally mounted current source registry, and separately configured
Ed25519 trust-root digest. Verification
uses the governed-offers path as the signed batch's accepted-record stream. Development processed
mode does not require this source-release envelope. V4 supports one Awin batch, not an aggregate
of multiple source batches.

## 3. Run verification

The normal local gate is:

```powershell
./scripts/test.ps1 -Suite all
npm run test:e2e
```

Run optional tests only when their dependencies are available:

```powershell
./scripts/test.ps1 -Suite python -IncludeIntegration -IncludeSlow
```

Check infrastructure independently:

```powershell
docker compose --env-file .env.example config --quiet
uv run --no-sync alembic -c db/alembic.ini upgrade head
```

The migration command needs a reachable PostgreSQL `DATABASE_URL`. A passing unit suite,
migration, or browser test is engineering evidence; none is a substitute for a frozen model
evaluation or user study.

## 4. Enable the NVIDIA GPU

The portable lockfile uses CPU PyTorch. Install the optional embedding dependencies, then apply
the local CUDA override:

```powershell
./scripts/check-memory-cap.ps1 -MaxUsedGb 55
uv sync --locked --extra embeddings
./scripts/setup-gpu.ps1
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The verified host used `torch==2.13.0+cu130` on an RTX 5070 Ti. This is host-specific operational
state, not a dependency-lock guarantee.

Important behavior:

- `uv sync` restores the CPU wheel in `uv.lock`.
- Use `uv run --no-sync ...` after the CUDA override.
- Rerun `setup-gpu.ps1` after any dependency sync.
- Sentence-Transformers can use CUDA when PyTorch detects it.
- The installed LightGBM wheel may lack a compatible GPU learner and fall back to CPU.
- Record the requested device, actual device, library versions, and fallback reason in every run.

GPU availability improves throughput; it does not make a split leakage-safe or a metric
reportable.

## 5. Acquire and materialize permitted data

The source policy and exact measured hashes are in [data-sources.md](data-sources.md). Fetch only
the registered open sources with:

```powershell
uv run --no-sync python scripts/fetch_open_data.py `
  --source buildcores --source blender --source mlperf `
  --buildcores-profile portfolio --blender-limit 3000
```

To reuse the source files already inspected on this host:

```powershell
uv run --no-sync python scripts/fetch_open_data.py --source buildcores `
  --buildcores-profile portfolio `
  --buildcores-archive tmp/source-inspection/buildcores-6a64ab14.zip

uv run --no-sync python scripts/fetch_open_data.py --source blender `
  --blender-archive tmp/source-inspection/blender-latest.zip `
  --blender-limit 3000 --blender-scan-limit 10000

uv run --no-sync python scripts/fetch_open_data.py --source mlperf `
  --mlperf-summary tmp/source-inspection/mlperf-v6-summary.json
```

Every materialization writes a raw content hash, retrieval metadata, license/access note,
deterministic records and rejections, a processed manifest, and a data-quality report. Reusing a
content-identical snapshot is idempotent. A distinct run is compared with the newest valid
`PASS` report for the same source and variant; material count, category, record-type, and
rejection-rate regressions cause a failed report and block the Dagster asset or CLI promotion.
The first passing run establishes the comparison baseline.

The Dynacore adapter is deliberately excluded from the open-source default. If its exact local
PDF is used for parser development, the command must be explicit:

```powershell
uv run --no-sync python scripts/fetch_open_data.py --source dynacore `
  --dynacore-pdf tmp/pdfs/dynacore-2026-07-17.pdf
```

Those records remain unmatched, stock `UNKNOWN`, development-only, training-ineligible,
claim-ineligible, and non-redistributable unless written permission changes the policy. A parser
quality `PASS` means the batch obeyed structural checks; it does not grant reuse rights or assert
that every offer is currently in stock.

For a consented retailer feed, review `RetailerFeedPolicy` in
`pipelines/sources/retailer_csv.py`. The retailer identity, feed ID, consent reference, source
URL, access note, and explicit use rights are mandatory.

For Awin, obtain and download the authorised feed outside this application; never pass or persist a
credential-bearing download URL. Then supply the local CSV/gzip, exact signed policy, detached
signature, local trust root, and an independently distributed trust-root digest:

```powershell
uv run --no-sync python scripts/fetch_open_data.py --source awin_feed `
  --awin-feed C:\secure\feeds\merchant.csv.gz `
  --awin-policy-json C:\secure\policies\merchant-policy.json `
  --awin-policy-signature C:\secure\policies\merchant-policy.sig.json `
  --awin-trust-root C:\secure\policies\trust-root.json `
  --awin-trust-root-sha256 REPLACE_WITH_64_HEX_SHA256
```

The importer verifies Ed25519 authority, validity windows, Singapore rights, signed category/host/
currency/resource rules, streams bounded input, and emits an authorization receipt. It does not
download feeds, manage keys, or execute retention. No real Awin feed or grant is included; see
[the Awin feed guide](awin-local-feed.md).

## 6. Build the semantic index

Use the normalized BuildCores envelope, not the raw archive:

```powershell
$PortfolioRecords = "data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/portfolio-3000/records.jsonl"

./scripts/check-memory-cap.ps1 -MaxUsedGb 55
uv run --no-sync python -m pc_build_recommender.retrieval.embedding_index `
  --input $PortfolioRecords `
  --output-dir artifacts/retrieval/buildcores-embeddings `
  --data-version buildcores-portfolio-3000-07e485e82333 `
  --encoder sentence-transformer `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --device cuda --batch-size 128
```

Validate `manifest.json` before loading the index. The measured artifact contains 3,000
L2-normalized float32 vectors with dimension 384 and records CUDA as the resolved encoder device.
It does not contain retrieval relevance metrics.

Production additionally requires a content-addressed local Sentence-Transformers bundle under
`encoders/<bundle-sha256>`. Serving-release v2 and operator settings must agree on path, digest,
file count, and bytes. The API image installs the `serving` extra, disables Hugging Face/
Transformers network access, loads with `local_files_only=true`, and runs a finite/nonzero/
normalized dimension-checking warm-up before readiness. No production weight bundle is bundled;
startup remains blocked until an operator mounts the pinned tree and complete serving release.

Use the packager only with a local snapshot whose model name and revision exactly match the index
manifest. It verifies both contracts before copying, adds deterministic provenance, validates the
published directory, and rejects symlinks/junctions in the source or final bundle tree:

```powershell
$EncoderSnapshot = "C:\secure\models\all-MiniLM-L6-v2\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
./scripts/check-memory-cap.ps1 -MaxUsedGb 55
uv run --no-sync python scripts/package_semantic_encoder_bundle.py `
  --source $EncoderSnapshot `
  --output-root artifacts/serving/encoders `
  --model-name sentence-transformers/all-MiniLM-L6-v2 `
  --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 `
  --licence Apache-2.0 `
  --expected-source-sha256 0f4856ff5afd30b5b9cc9b3864c48d4daf24cbe9124bf2c7e21ceab6de297bf0 `
  --embedding-manifest artifacts/retrieval/buildcores-full-embeddings-pinned/manifest.json
```

The generated `artifacts/serving/` directory is intentionally ignored by Git. Copy the resulting
digest-named directory into `PCBR_SERVING_RELEASE_DIR/encoders/`; do not rename it, edit it, or
replace a previously published bundle.

## 7. Exercise the model-training stack

### Synthetic smoke data

Generate deterministic fixtures:

```powershell
./scripts/check-memory-cap.ps1 -MaxUsedGb 55
uv run --no-sync python -m training.generate_starter_data
```

Train entity-resolution baselines and LightGBM:

```powershell
uv run --no-sync python -m training.train_entity_resolution `
  --input data/processed/starter/entity_resolution.synthetic.jsonl `
  --artifact-dir artifacts/models/entity-resolution/synthetic-smoke `
  --model all --device auto --allow-synthetic-diagnostics
```

Train median, Ridge, and LightGBM workload regressors:

```powershell
uv run --no-sync python -m training.train_performance `
  --input data/processed/starter/performance_gpu.synthetic.csv `
  --artifact-dir artifacts/models/performance/synthetic-smoke `
  --device auto --allow-synthetic-diagnostics
```

The `--allow-synthetic-diagnostics` switch is intentionally conspicuous. Artifacts produced from
these fixtures are permanently non-promotable, even when their scores appear strong.

### Entity-resolution transfer benchmark

The CC BY 4.0 Zenodo archive at <https://doi.org/10.5281/zenodo.8164151> is stored as
`tmp/er-benchmark/Dn7.zip` with SHA-256
`3e3fd6951ab4c4ed6aa741c2594d3ab496b63aeca6b41b8a1e639bc6d9895980`. Its deposited files
declare 43,418 candidate pairs with 763 positives across source tables of 2,554 and 22,074 rows.

The guarded materializer assigned source records, rather than pair rows, to disjoint splits and
dropped candidates that would cross those boundaries:

| Split | Pairs | Positives |
| --- | ---: | ---: |
| Train | 15,951 | 481 |
| Validation | 1,945 | 144 |
| Test | 1,802 | 138 |

At the validation-F1 threshold, CPU LightGBM produced held-out precision 0.81102, recall 0.74638,
F1 0.77736, and average precision 0.81666. At a validation threshold selected for at least 0.99
precision, test precision reached 1.0 but recall fell to 0.05072. This clearly misses the combined
precision, recall, and F1 requirements.

The report is `artifacts/models/er-transfer-dn7/transfer_benchmark_report.json`, SHA-256
`6f81289e73584f2894ea9ddf8380c4201140927b56584fa6203877981381a022`.
It is transfer-benchmark-only and cannot establish PC-component retailer or Singapore-production
quality. A separate PC-domain labelled test set remains mandatory. Before parsing the materialised
split files, the command reserves conservative host memory for JSON objects, typed pairs, optional
embeddings, and learner workspaces. It refuses a projected use at or above 55 GiB by default and
records the preflight in the report; `--max-host-used-gb` and
`--minimum-free-memory-mb` make that operational limit explicit.

The transfer command evaluates every configured baseline and model before it writes any model
artifact. A later evaluation failure therefore cannot leave an earlier baseline looking like a
complete comparable benchmark release. This guards diagnostic evidence only; it does not make the
transfer cohort promotable.

### Real entity-resolution data

Collect independent labels through the PostgreSQL-backed workflow rather than editing qrels or pair
files directly. Apply Alembic revision `20260723_0006`, then invoke
`scripts/manage_annotations.py` from a trusted wrapper. The wrapper must provide an identity JSON
whose OIDC issuer/subject was already verified upstream; the CLI deliberately does not validate a
JWT and production wrappers should use `--require-verified-identity-file`.

The operational sequence is `bootstrap-admin`, `provision-reviewer`, `create-project`,
`import-batch`, `open-project`, repeated `claim-review` / `submit-judgment`, independent
`claim-adjudication` / `submit-adjudication` where needed, then `freeze-project`. Claim output must
be written to a protected file because it contains the one-time lease secret; only a SHA-256 is
stored in PostgreSQL. The service enforces exactly two different reviewers, no reviewer reclaim,
hashed idempotency, immutable judgments/audit events, and an adjudicator who did not judge that
item. Relevance hard failures are structured codes separate from the 0-4 grade.

An administrator can run `project-status --project-id <id>` at any point to see aggregate item
states, judgment coverage, lease health, adjudication backlog, and coarse freeze blockers. The
command intentionally emits no evidence, labels, rationales, reviewer identities, or lease
secrets, and it uses grouped database counts rather than materialising the annotation corpus. Its
preflight is operational guidance only: `freeze-project` remains the strict integrity and release
gate.

Freezing emits `annotation-<release-sha256>/`. Relevance exports contain
`human-judgments.json`, `qrels.json`, `query-split.json`, and `evidence-snapshots.json`; entity-
resolution exports contain `human-labels.json`, `pairs.jsonl`, `listing-split.json`, and evidence.
The manifest binds every file. JSON and streaming JSONL batches run through one service transaction;
a mid-import parse or database failure rolls back every group, item, and audit row so a clean retry
is safe. No human release exists yet, so the commands provide a collection surface, not achieved
label counts or metrics.

### Preparing a relevance review batch

Do not import `frozen-candidates.json` from the silver diagnostic: it includes weak labels and is
not a valid human-collection input. First capture real, rights-cleared candidates in the strict
label-free schema at
[`evals/retrieval/relevance-annotation-candidates.template.json`](../evals/retrieval/relevance-annotation-candidates.template.json).
Each candidate needs a canonical product ID, reviewer-safe evidence, and one HTTPS source record
with the access/licence note and retrieval time. Each query must declare its actual intent-leakage
group; paraphrases of one intent share that group.

For a verified catalogue, use the bounded capture command instead of copying product rows by hand.
It verifies the catalogue `records.jsonl` against its manifest, retains only records that explicitly
allow training and published metrics, pools category-scoped candidates with the development hybrid
retriever, and removes retrieval rank, score, and source-membership fields before creating the
review input. It first retains every fused RRF candidate, then adds source-exclusive BM25 and
vector candidates round-robin until the explicit per-query cap. The supplied BuildCores starter
prompts are author-curated requests only; they are not qrels or human labels. The source licence
allows a future derived model to be served only if the carried attribution notice is displayed on
each public output that uses BuildCores evidence. It does not make the present development retrieval
stack, price data, or a model release production-ready.

```powershell
uv run --no-sync python scripts/capture_relevance_annotation_candidates.py `
  --query-set evals/retrieval/buildcores-portfolio-annotation-queries.v2.json `
  --catalog-records data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/portfolio-3000/records.jsonl `
  --catalog-manifest data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/portfolio-3000/manifest.json `
  --output-dir data/processed/annotation_candidate_capture/buildcores-portfolio-3000-author-curated-v3 `
  --top-k 20 `
  --per-source-top-k 20 `
  --max-candidates-per-query 30
```

The capture manifest binds the query set, catalogue file and manifest, source policy, candidate
IDs, candidate bytes, and aggregate per-query pooling counts. It records that development retrieval
was used only to construct the review pool; ranks, scores, and a candidate's source membership are
not exposed to reviewers and are not evaluation evidence. A separate
`prelabel-features.jsonl` is not reviewer input: it commits the exact raw candidate features and
the `RankingFeatureBuilder` matrix before labels exist. The manifest binds its bytes, per-query
row hashes, candidate universe, feature contract, retrieval contract, and explicit development-
vector promotion blocker. The default cap is deliberately modest:
reviewers see up to the 20-item fused set plus at most 10 source-exclusive additions, while the
manifest never maps an individual candidate to a discovery source.

Compile that capture before creating the project:

```powershell
uv run --no-sync python scripts/prepare_relevance_annotation_batch.py `
  --input data/processed/annotation_candidate_capture/buildcores-portfolio-3000-author-curated-v3/candidates.json `
  --capture-manifest data/processed/annotation_candidate_capture/buildcores-portfolio-3000-author-curated-v3/manifest.json `
  --output-dir data/processed/annotation_batch/buildcores-portfolio-3000-author-curated-v3
```

The compiler applies the same recursive blinding guard as the annotation service, rejects synthetic
rows, model/retrieval ranks and scores, relevance labels, unapproved source policy, and incomplete
provenance. It assigns the `train`, `validation`, and `test` splits by deterministic leakage group
before review and refuses to overwrite an existing output directory. The emitted `manifest.json`
binds the raw capture, policy, split assignment, and exact JSONL bytes. It does not label examples,
imply human agreement, or make a model promotable.

Use the generated files at the trusted service boundary:

```powershell
uv run --no-sync python scripts/manage_annotations.py `
  --verified-identity-file $env:ANNOTATION_VERIFIED_IDENTITY_FILE `
  --require-verified-identity-file `
  create-project --spec data/processed/annotation_batch/buildcores-portfolio-3000-author-curated-v3/project-spec.json

uv run --no-sync python scripts/manage_annotations.py `
  --verified-identity-file $env:ANNOTATION_VERIFIED_IDENTITY_FILE `
  --require-verified-identity-file `
  import-batch --project-id <project-id> `
  --input data/processed/annotation_batch/buildcores-portfolio-3000-author-curated-v3/groups.jsonl
```

Only then open the project and collect two independent reviews plus adjudication through the existing
workflow. Keep the capture and resulting immutable batch alongside the later frozen release so
ranking evidence can be reproduced.

The checked-in v3 starter capture contains 16 queries and 480 blinded query-product tasks. Its
deterministic split contains 10 train, 3 validation, and 3 test intent groups. This proves the
pre-label commitment path only: it is far below the approximately 150 independently reviewed
queries and 50 frozen test groups required by the promotion policy.

After `freeze-project` creates a content-addressed relevance release, materialize the only
human-training input accepted by the promotion path:

```powershell
uv run --no-sync python -m training.materialize_ranking_snapshot `
  --capture-dir data/processed/annotation_candidate_capture/buildcores-portfolio-3000-author-curated-v3 `
  --annotation-release-dir data/annotations/releases/annotation-<release-sha256> `
  --output-dir data/evaluation/ranking/human-v1
```

The materializer independently verifies every annotation file hash and size, release self-hash,
two reviewers per pair, adjudicated qrels, split lineage, and the pre-label hashes preserved in
`evidence-snapshots.json`. It rebuilds the feature matrix and refuses any post-label feature
change. The output `ranking.jsonl` differs from the committed pre-label rows only by
`relevance_grade`; `manifest.json` binds both phases.

For **entity resolution**, the JSONL input must contain listing/canonical pairs, binary labels,
row-level `is_synthetic`, and stable listing IDs. The trainer groups every listing's candidate set
into one split. Use a frozen external file for the final evaluation:

```powershell
uv run --no-sync python -m training.train_entity_resolution `
  --input data/evaluation/entity-resolution/train-labelled.jsonl `
  --artifact-dir artifacts/models/entity-resolution/candidate `
  --model all --device auto

uv run --no-sync python -m training.evaluate_entity_resolution `
  --input data/evaluation/entity-resolution/frozen-test.jsonl `
  --artifact-dir artifacts/models/entity-resolution/candidate/lightgbm
```

Do not enable automatic merging until the frozen hard-negative set supports the precision gate
and its confidence interval. A high score on generated title variants is not evidence of retailer
generalization.

### Real performance data

The CSV must provide numeric feature columns, a positive target, product ID, product-family or
generation leakage group, and explicit `is_synthetic`. Preserve benchmark context before forming
the target. Example shape:

```powershell
uv run --no-sync python -m training.train_performance `
  --input data/evaluation/performance/gpu-render-train.csv `
  --artifact-dir artifacts/models/performance/gpu-render-candidate `
  --dataset-manifest data/evaluation/performance/gpu-render-train.manifest.json `
  --category gpu --workload blender_render `
  --features "compute_units,vram_gb,memory_bandwidth_gbps,boost_clock_mhz,board_power_w" `
  --target-column target_score `
  --product-id-column product_id `
  --family-column product_family `
  --generation-column hardware_generation `
  --synthetic-column is_synthetic `
  --device auto

uv run --no-sync python -m training.evaluate_performance `
  --input data/evaluation/performance/gpu-render-frozen-test.csv `
  --artifact-dir artifacts/models/performance/gpu-render-candidate
```

The example paths are release inputs to create, not files currently promised by the repository.
The dataset manifest is mandatory for a promotion-eligible candidate; omitting
it permanently marks the output non-promotable. Never pool Blender
versions/scenes/backends/OS contexts or MLPerf systems with different node and
accelerator counts into a supposedly comparable target.

`--target-transform log1p` is available for a pre-declared experiment when a
positive benchmark target spans several orders of magnitude and relative error
is the decision metric. It changes only the learner's internal target scale:
validation/test metrics, intervals, and served values are inverse-transformed
back to the original benchmark unit. It is not a replacement for a frozen
holdout, comparable benchmark coverage, or the promotion gates.

Training evidence records repository-relative paths where possible. Operator
inputs outside the repository are represented as `<external>/<filename>` with
their SHA-256 digest, so published evidence remains portable without exposing
machine-specific directory layouts.

`training.prepare_blender_performance` reads both normalized JSONL inputs once. It keeps the
catalogue-family index in memory, but writes the compact, conservatively matched Blender rows to a
temporary disk-backed SQLite database with an 8 MiB SQLite page-cache budget. Cohort selection and
exact median/MAD aggregation then read one selected hardware family at a time; raw benchmark
envelopes are not accumulated in Python. The CLI rejects a source JSONL row above 1,000,000 bytes
before decoding it and applies a conservative admission check by default: projected host use must
stay below 55 GiB with at least 1 GiB free. It records the line limit, SQLite cache budget, memory
reservation, and preflight snapshot in the dataset manifest. Ensure the system temporary drive has
room for the intermediate store; `--maximum-record-bytes`, `--max-host-used-gb`,
`--minimum-free-memory-mb`, `--catalog-memory-expansion-factor`, and
`--preparation-runtime-memory-mb` are explicit operating controls, not quality claims.

The repository also retains a completed GPU engineering diagnostic, not a candidate release. It
uses the local Blender 4.5.0 / `junkshop` / OPTIX / Windows pilot with its existing manifest and
is intentionally pinned to CPU, one training thread, 128 MiB model budget, and the 55 GiB host
admission cap:

```powershell
./scripts/check-memory-cap.ps1 -MaxUsedGb 55
uv run --no-sync python -m training.train_performance `
  --input data/processed/model_training/blender_gpu_performance_250k_full/blender_performance.csv `
  --dataset-manifest data/processed/model_training/blender_gpu_performance_250k_full/manifest.json `
  --artifact-dir artifacts/ml/performance/blender_gpu_content_creation_optix_pilot_v1 `
  --category gpu --workload blender_4_5_0_junkshop_optix_windows `
  --features base_clock_mhz,boost_clock_mhz,vram_gb,board_power_watts `
  --device cpu --max-training-memory-mb 128 --max-host-used-gb 55 `
  --minimum-free-memory-mb 1024 --max-cpu-threads 1 --bootstrap-resamples 500
```

This command must be directed at a new artifact path if rerun: sealed artifact directories are
immutable. The current 50-row / 36-family pilot fails every quality gate and is not a routeable
model; see the model card rather than treating it as a GPU accuracy result.

The trainer publishes an evidence-sealed performance artifact: its v2 manifest
hash-binds the model, metadata, training evidence, and training report, plus
an exact copied dataset manifest when one was supplied. A promotable artifact
cannot be loaded without all three evidence records and their matching split,
frame, model, and dataset identities. Older v1 artifacts remain loadable only
when their own metadata is non-promotable.

Before model fitting, the performance trainer also samples physical host memory
and reserves its conservative model-allocation estimate. It refuses a run that
would reach the default 55 GiB used-RAM cap or leave less than 1 GiB free;
`--max-host-used-gb` and `--minimum-free-memory-mb` make that operating policy
explicit and preserve the sampled preflight in the training report. This is an
admission check, not a claim that other processes cannot allocate memory later.
The ranking and entity-resolution trainers apply the same policy before
materializing their JSON inputs. Their reports record input bytes, the explicit
12x JSON/object/feature expansion factor, a 512 MiB runtime allowance, and the
host-memory admission snapshot; those values can be tightened for a smaller
deployment with the matching CLI options.

### Ranking

LambdaMART needs query-grouped candidates and relevance grades from 0 to 4. Before training:

1. freeze realistic query scenarios and structured constraints;
2. have two reviewers grade a shared candidate pool;
3. resolve material disagreements;
4. split by query ID, never by query-product row; and
5. compare BM25 and LambdaMART on identical candidates and qrels.

No silver or rule-generated label should be described as human relevance. Such a pilot can debug
features and evaluation code but remains non-reportable.

Train only from the materialized, manifest-bound human snapshot:

```powershell
uv run --no-sync python -m training.train_ranking `
  --input data/evaluation/ranking/human-v1/ranking.jsonl `
  --dataset-manifest data/evaluation/ranking/human-v1/manifest.json `
  --human-judgments data/annotations/releases/annotation-<release-sha256>/human-judgments.json `
  --qrels data/annotations/releases/annotation-<release-sha256>/qrels.json `
  --frozen-query-split data/annotations/releases/annotation-<release-sha256>/query-split.json `
  --candidate-set-version <qrels-version> `
  --label-provenance human `
  --artifact-dir artifacts/models/ranking/candidate-v1
```

The trainer reports validation diagnostics only. It does not score, log, or make a promotion
decision from the frozen test split. Before any test access, preregister exactly one model,
evaluation policy, bootstrap configuration, and frozen cohort:

```powershell
uv run --no-sync python -m training.register_ranking_evaluation `
  --feature-snapshot data/evaluation/ranking/human-v1/ranking.jsonl `
  --dataset-manifest data/evaluation/ranking/human-v1/manifest.json `
  --human-judgments data/annotations/releases/annotation-<release-sha256>/human-judgments.json `
  --qrels data/annotations/releases/annotation-<release-sha256>/qrels.json `
  --frozen-query-split data/annotations/releases/annotation-<release-sha256>/query-split.json `
  --ranker-model artifacts/models/ranking/candidate-v1/ranker-artifact/ranker.txt `
  --intent-root artifacts/evaluation/ranking/intents `
  --ledger-dir artifacts/evaluation/ranking/test-access-ledger
```

Then evaluate the persisted artifact using only the generated content-addressed intent. The
evaluator never accepts caller-supplied challenger rankings or policy overrides:

```powershell
uv run --no-sync python -m training.evaluate_ranking `
  --feature-snapshot data/evaluation/ranking/human-v1/ranking.jsonl `
  --dataset-manifest data/evaluation/ranking/human-v1/manifest.json `
  --human-judgments data/annotations/releases/annotation-<release-sha256>/human-judgments.json `
  --qrels data/annotations/releases/annotation-<release-sha256>/qrels.json `
  --frozen-query-split data/annotations/releases/annotation-<release-sha256>/query-split.json `
  --ranker-model artifacts/models/ranking/candidate-v1/ranker-artifact/ranker.txt `
  --evaluation-intent artifacts/evaluation/ranking/intents/<intent-sha256>.json `
  --ledger-dir artifacts/evaluation/ranking/test-access-ledger `
  --output-dir artifacts/evaluation/ranking/sealed
```

Registration claims the cohort for one intent. Exact retries are idempotent; any different model,
policy, or bootstrap configuration for that cohort fails closed. The evaluator records test access
before re-adjudicating human judgments, computes BM25, RRF, and challenger rankings, and commits a
content-addressed no-replace bundle under `sealed/<intent-sha256>/`. The evidence binds the exact
model, metadata, manifest, feature matrix, candidate set, finite scores, rankings, access record,
and promotion decision. These example paths are future release inputs; no qualifying
human-labelled ranker artifact is currently included.

The training command publishes `artifact-dir/ranker-artifact/` as one immutable bundle. It seals
and reloads model/metadata/manifest files in a hidden sibling stage, fsyncs them, and commits with a
single no-replace directory rename. `publication_intent_sha256` binds the exact feature snapshot,
human judgments, qrels, query split, candidate/data versions, parameters, seed, and early stopping.
Same-intent crash/concurrent retries adopt the committed exact bytes; a different intent fails.
Stale hidden stages are never served and can be inspected with
`scripts/maintain_ranker_publication_stages.py` (dry-run by default; deletion requires `--apply`).
This integrity control does not mean a qualifying ranker has been trained.

### Current retrieval diagnostic

The checked silver pilot used 3,000 real catalogue records and 32 deterministic
specification-predicate queries. It contains 12,000 category-candidate rows and 1,549 positive
weak labels, but no human judgments. Its purpose is to exercise retrieval and expose weaknesses:

| System | Recall@20 | Recall@50 | MRR | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.299656 | 0.534842 | 0.522279 | 0.273296 |
| Vector only | 0.200046 | 0.345173 | 0.570899 | 0.273354 |
| BM25 + vector RRF | 0.310128 | 0.534015 | 0.644085 | 0.350134 |
| Stable-ID negative control | 0.065153 | 0.164753 | 0.303975 | 0.106372 |

RRF's diagnostic NDCG@10 delta over BM25 is `+0.076838` (silver-label relative delta
`+28.115%`), but Recall@50 is effectively unchanged. These values must not be described as the
project's ranking target achievement: the labels are generated from the same declared predicates,
there is no reviewer agreement, and LambdaMART was intentionally not trained because doing so
would measure reconstruction of the weak-label rule.

Evidence:

- `artifacts/evaluation/retrieval-silver-pilot-v1/metrics.json`, file SHA-256
  `9fa91b01fd385a147f56ee7267930b4f106aa38c6225de6bdd862cffdaf5c737`;
- `artifacts/evaluation/retrieval-silver-pilot-v1/frozen-candidates.json`, SHA-256
  `96bc2d029a9f7046f465990a216d9abe2c8702123a374739c10780ca51fa5386`;
- `evals/retrieval/real_catalog_silver_queries.v1.json`, SHA-256
  `fac11c46a59f1cfe88908fb2b7f0188e018c956665cd885f7ceceff4d8469968`.

### Current performance diagnostic and temporal audit

The corrected Blender/BuildCores dataset selects Blender 4.0.0 / `junkshop` / CPU / Windows and
preserves native higher-is-better `samples_per_minute`. Its 1,625 joined observations aggregate to
172 hardware rows across 109 families. The development LightGBM internal split reports R-squared
0.9408 and MAPE 22.23%, but this is already-observed, adaptively explored evidence; MAPE exceeds the
12% gate and the artifact is relative-only/non-promotable. Dataset CSV SHA-256:
`a4922e0d51a7981f40a257a363cb25649ce5cc27cf32a7517b5e560652acfdf9`.

The isolated v3 CPU experiment selected features only from 113 development rows, then evaluated a
separate 30-row / 15-family holdout. It improved development out-of-fold MAPE 12.04% relatively,
but the held-out diagnostic remains R-squared 0.8763 and MAPE 18.98%, with grouped-bootstrap lower
R-squared 0.6279 and upper MAPE 24.02%. Its schema is permanently non-loadable by production and
the already-explored dataset still requires a new external frozen cohort before promotion.

The 2026-07-23 temporal evaluator compared raw snapshots `c0f9d35c...af4833` and
`67582ebc...93f5`. It isolated 113 novel submissions / 339 observations, but zero met the full
frozen cohort including build hash, benchmark script, and scene checksum. It retained 0 rows and 0
families, attempted no model inference, computed no external metric, pooled no rows, and left
promotion disabled. Report semantic digest:
`957139150dbc1bbdd6deab1810b966ea59a33d9768b206e5033287c88a3f044b`; exact file SHA-256:
`e675a0d1de5f0556aedeebb41c93f19f8130f3af22b3a010b7dc9c789fb458cf`; protocol SHA-256:
`6952cd7a220920fc9be882211577a67a67268a055e702e4b64955ada5123154a`.

The separate GPU OPTIX pilot has a correctly oriented render-throughput target but only 50 rows
across 36 product-family groups. Its sealed 6-row internal test is far below the confidence gates
and fails them (LightGBM R-squared -1.8538, MAPE 203.52%). It is retained to prevent accidental
GPU-model promotion from an inadequate cohort, not as accuracy evidence.

### Optional MLflow tracking

Tracking is off unless a training CLI receives `--track-mlflow`. For a portable CPU environment:

```powershell
uv pip install 'mlflow>=2.19,<4'
```

If the verified CUDA override is already installed, avoid a synchronizing command that would
restore the CPU wheel. Install only MLflow into the existing environment:

```powershell
uv pip install "mlflow>=2.19,<4"
```

Then use either the local artifact-backed store or the optional Compose server:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
# Append --track-mlflow to one of the complete training commands above.
```

Tracking records dataset hashes, grouped splits, feature/model versions, metrics, devices,
fallbacks, promotion blockers, and artifact checksums. It is fail-open for experiment logging:
training can continue when MLflow is absent or unreachable, but the failure is recorded. See
[mlflow-tracking.md](mlflow-tracking.md) for all three training CLIs.

## 8. Evidence and promotion contract

An ingestion manifest is not automatically an evaluation dataset manifest. A promotable run must
retain:

- immutable input hashes and access/licensing notes;
- row and group counts plus split assignments and seed;
- row-level synthetic provenance;
- leakage checks and point-in-time feature cutoffs;
- baseline and candidate predictions on the same eligible rows;
- point estimates, sample counts, confidence intervals, and failure slices;
- model, feature, data, rule, and library versions;
- requested and actual compute device; and
- artifact hashes and serving compatibility checks.

Use [evaluation-report.md](evaluation-report.md) as the public claim registry. A value belongs in
its measured column only after a qualifying artifact is independently checked. Targets, console
scores, synthetic diagnostics, silver-label pilots, and counts of unvalidated generated cases do
not qualify.

## 9. Portfolio evidence roadmap

Prioritize the following work before adding infrastructure:

| Gate | Evidence required |
| --- | --- |
| Retail coverage | At least three consented Singapore feeds with stable listing IDs, explicit rights, price/stock freshness, and unmatched-listing review. |
| Entity resolution | 2,500 independently labelled pairs, hard variant negatives, frozen thresholds, confidence intervals, and source/family slices. |
| Retrieval and ranking | Approximately 150 queries, 2,000+ two-reviewer grades, frozen qrels, paired BM25/vector/RRF/LambdaMART comparison. |
| Performance | Comparable workload cohorts, family/generation grouping, observed-first serving, OOD and confidence reporting. |
| Compatibility | 10,000 saved cases with expected rule outcomes and independent rechecks under a named rule version. |
| Optimisation | Exhaustive parity on reduced catalogues, feasible-incumbent revalidation, diversity, stability, and infeasibility explanations. |
| Operations | Declared load profile, p50/p95/p99 latency, error rate, throughput, and subsystem timing. |
| User impact | Counterbalanced study, participant-level paired results, median time, mistakes, confidence, and blinded build quality. |

This evidence path supports strong model metrics without optimizing toward a test set or turning
unlicensed market data into an accidental dependency.

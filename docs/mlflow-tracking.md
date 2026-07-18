# Optional MLflow experiment tracking

Entity-resolution, workload-performance, and LambdaMART training can record reproducible
MLflow runs. Tracking is deliberately opt-in: no CLI contacts a tracking server or creates
a local run store unless `--track-mlflow` is passed.

For a portable CPU environment, install the optional dependency without changing model
code:

```powershell
uv pip install 'mlflow>=2.19,<4'
```

On an already CUDA-configured workstation, preserve the verified PyTorch wheel by using
`uv pip install "mlflow>=2.19,<4"` instead of syncing. Then follow
[`scripts/setup-gpu.ps1`](../scripts/setup-gpu.ps1) and use `uv run --no-sync`; a regular
sync may restore the lockfile's portable CPU PyTorch wheel.

With no URI configured, the run store is `artifacts/mlruns`:

```powershell
uv run --no-sync python -m training.train_performance `
  --input data/processed/model_training/performance.csv `
  --artifact-dir artifacts/models/performance/gpu-gaming-v1 `
  --category gpu `
  --workload gaming_1440p `
  --features architecture_generation,vram_gb,memory_bandwidth_gbps,board_power_w `
  --track-mlflow
```

To use the Compose MLflow server or another HTTP backend, set the standard environment
variable and still pass the explicit opt-in flag:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
uv run --no-sync python -m training.train_entity_resolution `
  --input data/processed/labels/entity-pairs.jsonl `
  --artifact-dir artifacts/models/entity-resolution/v1 `
  --track-mlflow
```

All three CLIs also accept `--mlflow-experiment` and `--mlflow-run-name`. If the optional
MLflow dependency is absent or the backend cannot be reached, model training and native
artifact persistence continue. The training report records `dependency_missing` or
`tracking_failed`; it never silently claims that tracking succeeded.

## Recorded evidence

Each run records:

- source file SHA-256 and dataset/content version;
- grouped split counts and leakage unit;
- feature and model versions;
- model parameters and finite held-out metrics;
- requested and actual learner device, plus fallback reason when applicable;
- promotion eligibility and every blocker;
- a deterministic SHA-256 manifest of every native artifact file.

Artifacts remain native JSON, JSONL, CSV, and LightGBM text. The integration does not call
MLflow model flavours, and `.pkl`, `.pickle`, and `.joblib` files are refused by the
artifact manifest. This keeps the serving contract inspectable and avoids pickle-based
model loading.

## Ranking evidence gate

`training.train_ranking` requires query-grouped candidates with a `relevance_grade` from
0 to 4 and explicit `--label-provenance`. Human labels additionally require the immutable
`--dataset-manifest` emitted by `training.materialize_ranking_snapshot`; it binds the feature
commitment made before review to the annotation release, qrels, and split. Silver and synthetic
labels require
`--allow-non-human-labels` and remain permanently non-promotable, regardless of their
diagnostic NDCG. A promotable evaluation additionally requires a frozen dataset, two or
more independent human reviewers, the target query/grade counts, and held-out improvement
over BM25 on the same candidate set.

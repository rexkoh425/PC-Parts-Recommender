"""Optional, failure-isolated MLflow tracking for reproducible training commands.

The training CLIs own and persist their native, inspectable artifacts.  This module only
copies those files into MLflow and records their SHA-256 manifest; it deliberately does
not use MLflow's model-flavour serializers (and therefore never creates pickle models).

Tracking is opt-in at the CLI.  When enabled, ``MLFLOW_TRACKING_URI`` may point at an
HTTP server.  If it is unset, runs use a repository-local file store under
``artifacts/mlruns``.  The optional ``mlops`` dependency may be absent in serving and
developer environments; an explicitly requested run is then reported as
``dependency_missing`` without affecting model training.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit, urlunsplit

from training._common import sha256_file

ARTIFACT_MANIFEST_SCHEMA = "pc-build-recommender.mlflow-artifacts.v1"
UNSAFE_MODEL_SUFFIXES = frozenset({".joblib", ".pickle", ".pkl"})


def _default_tracking_uri() -> str:
    return (Path.cwd() / "artifacts" / "mlruns").resolve().as_uri()


def _redact_uri(uri: str) -> str:
    """Remove credentials before a tracking URI enters a report or run tag."""

    try:
        parsed = urlsplit(uri)
        if parsed.username is None and parsed.password is None:
            return uri
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except ValueError:
        return "<invalid-tracking-uri>"


@dataclass(frozen=True, slots=True)
class MLflowTrackingConfig:
    """Explicit MLflow run settings resolved by a training CLI."""

    enabled: bool
    experiment_name: str
    run_name: str | None = None
    tracking_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("MLflow experiment name must not be empty")
        if self.enabled and self.run_name is not None and not self.run_name.strip():
            raise ValueError("MLflow run name must not be empty")

    @property
    def resolved_tracking_uri(self) -> str:
        return self.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or _default_tracking_uri()

    @property
    def safe_tracking_uri(self) -> str:
        return _redact_uri(self.resolved_tracking_uri)


def add_mlflow_arguments(parser: Any, *, default_experiment: str) -> None:
    """Add consistent, opt-in MLflow arguments to an ``argparse`` parser."""

    parser.add_argument(
        "--track-mlflow",
        action="store_true",
        help=(
            "log this run to MLflow; remains disabled unless this flag is passed "
            "(install the 'mlops' extra to enable it)"
        ),
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help=(
            "MLflow file or HTTP URI; defaults to MLFLOW_TRACKING_URI or "
            "artifacts/mlruns"
        ),
    )
    parser.add_argument("--mlflow-experiment", default=default_experiment)
    parser.add_argument("--mlflow-run-name")


def tracking_config_from_args(args: Any) -> MLflowTrackingConfig:
    return MLflowTrackingConfig(
        enabled=bool(args.track_mlflow),
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=str(args.mlflow_experiment),
        run_name=args.mlflow_run_name,
    )


def _manifest_entries(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"artifact tree must not contain symlinks: {path}")
        if not path.is_file():
            continue
        if path.suffix.casefold() in UNSAFE_MODEL_SUFFIXES:
            raise ValueError(f"pickle-style artifacts are not permitted: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ValueError(f"artifact tree contains no files: {root}")
    return entries


def build_artifact_manifest(root: str | Path) -> dict[str, Any]:
    """Build a deterministic, content-addressed manifest for a native artifact tree."""

    path = Path(root).resolve()
    entries = _manifest_entries(path)
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "root_name": path.name,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": entries,
    }


def _param_value(value: Any) -> str | int | float | bool:
    if value is None:
        return "null"
    if isinstance(value, bool | int | float | str):
        text: str | int | float | bool = value
    else:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if isinstance(text, str) and len(text) > 500:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
    return text


def flatten_parameters(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, str | int | float | bool]:
    """Flatten nested mappings into stable MLflow parameter keys."""

    flattened: dict[str, str | int | float | bool] = {}
    for key in sorted(values):
        value = values[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_parameters(value, prefix=name))
        else:
            flattened[name] = _param_value(value)
    return flattened


def finite_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    """Keep only finite numeric metrics; booleans are not measurements."""

    result: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            result[str(name)] = numeric
    return result


@dataclass(slots=True)
class OptionalMLflowRun:
    """A best-effort MLflow run that cannot make model training fail."""

    config: MLflowTrackingConfig
    status: str = field(init=False)
    run_id: str | None = field(default=None, init=False)
    artifact_uri: str | None = field(default=None, init=False)
    errors: list[str] = field(default_factory=list, init=False)
    _mlflow: Any = field(default=None, init=False, repr=False)
    _active_run: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.status = "pending" if self.config.enabled else "disabled"

    def __enter__(self) -> Self:
        if not self.config.enabled:
            return self
        try:
            self._mlflow = importlib.import_module("mlflow")
        except (ImportError, ModuleNotFoundError) as error:
            self.status = "dependency_missing"
            self.errors.append(f"{type(error).__name__}: install the 'mlops' extra")
            return self
        try:
            self._mlflow.set_tracking_uri(self.config.resolved_tracking_uri)
            self._mlflow.set_experiment(self.config.experiment_name)
            self._active_run = self._mlflow.start_run(run_name=self.config.run_name)
            info = self._active_run.info
            self.run_id = str(info.run_id)
            artifact_uri = getattr(info, "artifact_uri", None)
            self.artifact_uri = str(artifact_uri) if artifact_uri is not None else None
            self.status = "active"
            self.log_tags(
                {
                    "tracking.backend": self.config.safe_tracking_uri,
                    "artifact.serialization": "native_text_json_no_pickle",
                }
            )
        except Exception as error:  # pragma: no cover - backend-specific failures
            self.status = "tracking_failed"
            self.errors.append(f"{type(error).__name__}: {error}"[:1000])
        return self

    @property
    def active(self) -> bool:
        return self.status == "active" and self._mlflow is not None

    def _attempt(self, operation: str, callback: Any) -> None:
        if not self.active:
            return
        try:
            callback()
        except Exception as error:  # pragma: no cover - backend-specific failures
            self.errors.append(f"{operation}: {type(error).__name__}: {error}"[:1000])

    def log_params(self, values: Mapping[str, Any]) -> None:
        params = flatten_parameters(values)
        if params:
            self._attempt("log_params", lambda: self._mlflow.log_params(params))

    def log_metrics(self, values: Mapping[str, Any]) -> None:
        metrics = finite_metrics(values)
        if metrics:
            self._attempt("log_metrics", lambda: self._mlflow.log_metrics(metrics))

    def log_tags(self, values: Mapping[str, Any]) -> None:
        tags = {str(key): str(value)[:5000] for key, value in values.items() if value is not None}
        if tags:
            self._attempt("set_tags", lambda: self._mlflow.set_tags(tags))

    def log_dict(self, payload: Mapping[str, Any], artifact_file: str) -> None:
        self._attempt("log_dict", lambda: self._mlflow.log_dict(dict(payload), artifact_file))

    def log_native_artifacts(
        self,
        root: str | Path,
        *,
        artifact_path: str = "model",
    ) -> dict[str, Any]:
        """Log an audited native artifact tree and return its content manifest."""

        path = Path(root).resolve()
        try:
            manifest = build_artifact_manifest(path)
        except (FileNotFoundError, OSError, ValueError) as error:
            self.errors.append(f"artifact_manifest: {type(error).__name__}: {error}"[:1000])
            return {
                "schema_version": ARTIFACT_MANIFEST_SCHEMA,
                "root_name": path.name,
                "error": f"{type(error).__name__}: {error}",
            }
        self.log_tags({f"{artifact_path}.content_sha256": manifest["content_sha256"]})
        self.log_metrics(
            {
                f"{artifact_path}.file_count": manifest["file_count"],
                f"{artifact_path}.total_bytes": manifest["total_bytes"],
            }
        )
        self.log_dict(manifest, f"manifests/{artifact_path}-manifest.json")
        self._attempt(
            "log_artifacts",
            lambda: self._mlflow.log_artifacts(str(path), artifact_path=artifact_path),
        )
        return manifest

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "status": self.status,
            "tracking_uri": self.config.safe_tracking_uri if self.config.enabled else None,
            "experiment_name": self.config.experiment_name if self.config.enabled else None,
            "run_id": self.run_id,
            "artifact_uri": _redact_uri(self.artifact_uri) if self.artifact_uri else None,
            "errors": list(self.errors),
        }

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        if self._active_run is None or self._mlflow is None:
            return
        try:
            self._mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")
            if exc_type is None and self.status == "active":
                self.status = "completed"
        except Exception as error:  # pragma: no cover - backend-specific failures
            self.status = "tracking_failed"
            self.errors.append(f"end_run: {type(error).__name__}: {error}"[:1000])


def promotion_blocker_tags(blockers: Sequence[str]) -> dict[str, str]:
    """Encode blockers in bounded tags while preserving full text as an artifact."""

    return {
        "promotion.eligible": str(not blockers).lower(),
        "promotion.blocker_count": str(len(blockers)),
        "promotion.blockers_sha256": hashlib.sha256(
            json.dumps(list(blockers), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }

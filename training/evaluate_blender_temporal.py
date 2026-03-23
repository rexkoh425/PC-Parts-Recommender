"""Run a bounded, fail-closed temporal audit of a Blender performance artifact.

The command compares immutable Blender Open Data snapshots by exact submission ID,
retains only a pre-registered complete benchmark execution contract, removes every
development leakage family, and computes model metrics only after fixed size gates
pass.  An insufficient cohort is a successful audit outcome, not an invitation to
pool incomparable observations or reuse development rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipelines.sources.base import RawSnapshot
from pipelines.sources.blender_temporal import (
    BlenderCohortContract,
    diff_blender_snapshots,
    load_blender_raw_snapshot,
)

from training._common import print_json, sha256_file
from training.prepare_blender_performance import (
    BenchmarkExecutionIdentity,
    prepare_blender_performance,
)

PROTOCOL_SCHEMA_VERSION = "pc-build-recommender.blender-temporal-evaluation-protocol.v1"
REPORT_SCHEMA_VERSION = "pc-build-recommender.blender-temporal-evaluation.v1"
REPORT_ENVELOPE_SCHEMA_VERSION = "pc-build-recommender.blender-temporal-evaluation-envelope.v1"
DEFAULT_PROTOCOL = Path("evals/performance/blender_cpu_content_creation_v4_external_protocol.json")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/evaluation/performance-temporal-v4")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return dict(value)


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _repository_root(protocol_path: Path) -> Path:
    for candidate in (protocol_path.parent, *protocol_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"cannot locate repository root above {protocol_path}")


def _resolve_input(root: Path, entry: Mapping[str, Any], *, label: str) -> Path:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label}.path must be a non-empty string")
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}.path escapes the repository root") from exc
    return path


def _verify_file(path: Path, expected_sha256: object, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{label} expected SHA-256 is invalid")
    observed = sha256_file(path)
    if observed != expected_sha256.casefold():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def _load_protocol(path: Path) -> tuple[dict[str, Any], str, Path]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    protocol = _mapping(payload, label="protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported Blender temporal evaluation protocol")
    return protocol, sha256_file(resolved), _repository_root(resolved)


def _stable_family_hashes(families: Sequence[str]) -> tuple[tuple[str, ...], str]:
    hashes = tuple(sorted(hashlib.sha256(value.encode("utf-8")).hexdigest() for value in families))
    digest = hashlib.sha256("".join(f"{value}\n" for value in hashes).encode("utf-8")).hexdigest()
    return hashes, digest


def _development_families(
    path: Path,
    *,
    expected_rows: int,
    expected_family_count: int,
    expected_family_hashes_sha256: str,
    maximum_rows: int = 10_000,
) -> tuple[set[str], tuple[str, ...]]:
    families: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "product_family" not in reader.fieldnames:
            raise ValueError("development CSV is missing product_family")
        for row_count, row in enumerate(reader, start=1):
            if row_count > maximum_rows:
                raise MemoryError(f"development CSV exceeds bounded limit {maximum_rows}")
            family = str(row.get("product_family") or "").strip()
            if not family:
                raise ValueError(f"development CSV row {row_count} has an empty product_family")
            families.add(family)
    if row_count != expected_rows:
        raise ValueError(
            f"development row count mismatch: expected {expected_rows}, observed {row_count}"
        )
    if len(families) != expected_family_count:
        raise ValueError(
            "development family count mismatch: "
            f"expected {expected_family_count}, observed {len(families)}"
        )
    family_hashes, family_hashes_sha256 = _stable_family_hashes(sorted(families))
    if family_hashes_sha256 != expected_family_hashes_sha256:
        raise ValueError("development leakage-family digest does not match protocol")
    return families, family_hashes


def _read_cpu_catalog(
    path: Path,
    *,
    expected_total_records: int,
    maximum_total_records: int,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total += 1
            if total > maximum_total_records:
                raise MemoryError(f"BuildCores input exceeds bounded limit {maximum_total_records}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            data = value.get("data")
            if isinstance(data, dict) and data.get("category") == "cpu":
                records.append(value)
    if total != expected_total_records:
        raise ValueError(
            f"BuildCores record count mismatch: expected {expected_total_records}, observed {total}"
        )
    return records, total


def _snapshot_lineage(snapshot: RawSnapshot, *, repository_root: Path) -> dict[str, object]:
    with zipfile.ZipFile(snapshot.path) as archive:
        members = [item for item in archive.infolist() if item.filename.lower().endswith(".jsonl")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one Blender JSONL member, found {len(members)}")
        member = members[0]
    return {
        "path": snapshot.path.relative_to(repository_root).as_posix(),
        "sha256": snapshot.content_sha256,
        "size_bytes": snapshot.byte_count,
        "metadata_path": snapshot.metadata_path.relative_to(repository_root).as_posix(),
        "metadata_sha256": sha256_file(snapshot.metadata_path),
        "retrieved_at": snapshot.retrieved_at.isoformat(),
        "parser_version": snapshot.parser_version,
        "jsonl_member": member.filename,
        "jsonl_uncompressed_bytes": member.file_size,
        "jsonl_compressed_bytes": member.compress_size,
    }


def _external_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    development_families: set[str],
    minimum_rows: int,
    minimum_families: int,
    strict_superset: bool,
) -> tuple[list[dict[str, Any]], dict[str, object], list[str]]:
    overlap_families = sorted(
        {
            str(row.get("product_family") or "")
            for row in rows
            if str(row.get("product_family") or "") in development_families
        }
    )
    clean_rows = [
        dict(row)
        for row in rows
        if str(row.get("product_family") or "") not in development_families
    ]
    clean_families = {str(row.get("product_family") or "") for row in clean_rows}
    remaining_overlap = sorted(clean_families & development_families)
    blockers: list[str] = []
    if not strict_superset:
        blockers.append("new snapshot is not a strict submission-ID superset of the old snapshot")
    if len(clean_rows) < minimum_rows:
        blockers.append(f"external cohort has {len(clean_rows)} rows; minimum is {minimum_rows}")
    if len(clean_families) < minimum_families:
        blockers.append(
            "external cohort has "
            f"{len(clean_families)} leakage families; minimum is {minimum_families}"
        )
    if remaining_overlap:
        blockers.append("development leakage-family overlap remains after exclusion")
    gate = {
        "candidate_row_count": len(rows),
        "candidate_leakage_family_count": len(
            {str(row.get("product_family") or "") for row in rows}
        ),
        "excluded_development_overlap_row_count": len(rows) - len(clean_rows),
        "excluded_development_overlap_family_count": len(overlap_families),
        "excluded_development_family_hashes": [
            hashlib.sha256(value.encode("utf-8")).hexdigest() for value in overlap_families
        ],
        "eligible_row_count": len(clean_rows),
        "eligible_leakage_family_count": len(clean_families),
        "remaining_development_overlap_count": len(remaining_overlap),
        "minimum_external_rows": minimum_rows,
        "minimum_external_leakage_families": minimum_families,
        "metric_evaluation_eligible": not blockers,
    }
    return clean_rows, gate, blockers


def _evaluate_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_directory: Path,
    expected_model_version: str,
    expected_features: tuple[str, ...],
    expected_development_group_hashes: tuple[str, ...],
    bootstrap_resamples: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    import numpy as np
    import pandas as pd  # type: ignore[import-untyped]

    from pc_build_recommender.performance_models import (
        calculate_regression_metrics,
        grouped_bootstrap_uncertainty,
        load_performance_artifact,
    )

    artifact = load_performance_artifact(artifact_directory)
    if artifact.model_version != expected_model_version:
        raise ValueError("loaded model version does not match protocol")
    if tuple(artifact.config.feature_columns) != expected_features:
        raise ValueError("loaded model features do not match protocol")
    if tuple(sorted(artifact.development_group_hashes)) != expected_development_group_hashes:
        raise ValueError("artifact development-group hashes do not match the frozen CSV")
    if artifact.config.category != "cpu" or artifact.config.workload != "content_creation":
        raise ValueError("artifact route does not match the temporal protocol")
    frame = pd.DataFrame.from_records(rows)
    predictions = np.asarray(
        artifact.booster.predict(
            frame.loc[:, expected_features], num_iteration=artifact.best_iteration
        ),
        dtype=float,
    )
    actual = frame[artifact.config.target_column].to_numpy(dtype=float)
    groups = frame[artifact.config.family_column].astype(str).tolist()
    metrics = calculate_regression_metrics(actual, predictions)
    uncertainty = grouped_bootstrap_uncertainty(
        actual,
        predictions,
        groups,
        confidence_level=bootstrap_confidence_level,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "model_version": artifact.model_version,
        "metrics": metrics.to_dict(),
        "grouped_bootstrap": uncertainty.to_dict(),
    }


def _write_content_addressed_report(
    report: Mapping[str, Any], output_directory: Path
) -> tuple[Path, str]:
    report_payload = dict(report)
    report_sha256 = _canonical_sha256(report_payload)
    envelope = {
        "schema_version": REPORT_ENVELOPE_SCHEMA_VERSION,
        "report_sha256": report_sha256,
        "report": report_payload,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{report_sha256}.json"
    temporary = output_path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    verify_temporal_report(output_path)
    return output_path, report_sha256


def verify_temporal_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    envelope = _mapping(payload, label="report envelope")
    if envelope.get("schema_version") != REPORT_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("unsupported temporal report envelope")
    report = _mapping(envelope.get("report"), label="report")
    observed = _canonical_sha256(report)
    if envelope.get("report_sha256") != observed:
        raise ValueError("temporal report digest does not match its payload")
    if report_path.stem != observed:
        raise ValueError("temporal report filename does not match its payload digest")
    return report


def run_temporal_evaluation(
    protocol_path: str | Path,
    *,
    output_directory: str | Path,
) -> tuple[Path, dict[str, Any]]:
    protocol, protocol_sha256, repository_root = _load_protocol(Path(protocol_path))
    inputs = _mapping(protocol.get("inputs"), label="protocol.inputs")
    development = _mapping(
        protocol.get("development_contract"), label="protocol.development_contract"
    )
    policy = _mapping(protocol.get("evaluation_policy"), label="protocol.evaluation_policy")
    limits = _mapping(protocol.get("resource_limits"), label="protocol.resource_limits")
    cohort_payload = _mapping(protocol.get("cohort_contract"), label="protocol.cohort_contract")

    old_entry = _mapping(inputs.get("old_snapshot"), label="inputs.old_snapshot")
    new_entry = _mapping(inputs.get("new_snapshot"), label="inputs.new_snapshot")
    development_csv_entry = _mapping(inputs.get("development_csv"), label="inputs.development_csv")
    development_manifest_entry = _mapping(
        inputs.get("development_manifest"), label="inputs.development_manifest"
    )
    catalog_entry = _mapping(inputs.get("buildcores_records"), label="inputs.buildcores_records")
    artifact_entry = _mapping(inputs.get("model_artifact"), label="inputs.model_artifact")

    old_path = _resolve_input(repository_root, old_entry, label="old_snapshot")
    new_path = _resolve_input(repository_root, new_entry, label="new_snapshot")
    development_csv_path = _resolve_input(
        repository_root, development_csv_entry, label="development_csv"
    )
    development_manifest_path = _resolve_input(
        repository_root, development_manifest_entry, label="development_manifest"
    )
    catalog_path = _resolve_input(repository_root, catalog_entry, label="buildcores_records")
    artifact_directory = _resolve_input(repository_root, artifact_entry, label="model_artifact")

    _verify_file(old_path, old_entry.get("sha256"), label="old snapshot")
    _verify_file(new_path, new_entry.get("sha256"), label="new snapshot")
    _verify_file(
        old_path.with_suffix(".zip.metadata.json"),
        old_entry.get("metadata_sha256"),
        label="old snapshot metadata",
    )
    _verify_file(
        new_path.with_suffix(".zip.metadata.json"),
        new_entry.get("metadata_sha256"),
        label="new snapshot metadata",
    )
    _verify_file(
        development_csv_path,
        development_csv_entry.get("sha256"),
        label="development CSV",
    )
    _verify_file(
        development_manifest_path,
        development_manifest_entry.get("sha256"),
        label="development manifest",
    )
    _verify_file(catalog_path, catalog_entry.get("sha256"), label="BuildCores records")
    _verify_file(
        artifact_directory / "artifact_manifest.json",
        artifact_entry.get("artifact_manifest_sha256"),
        label="artifact manifest",
    )
    _verify_file(
        artifact_directory / "metadata.json",
        artifact_entry.get("metadata_sha256"),
        label="artifact metadata",
    )
    _verify_file(
        artifact_directory / "model.txt",
        artifact_entry.get("model_sha256"),
        label="artifact model",
    )

    development_families, development_family_hashes = _development_families(
        development_csv_path,
        expected_rows=_positive_integer(development.get("row_count"), label="row_count"),
        expected_family_count=_positive_integer(
            development.get("leakage_family_count"), label="leakage_family_count"
        ),
        expected_family_hashes_sha256=str(development["leakage_family_hashes_sha256"]),
    )
    artifact_metadata = _mapping(
        json.loads((artifact_directory / "metadata.json").read_text(encoding="utf-8")),
        label="artifact metadata",
    )
    artifact_group_hashes = tuple(
        sorted(str(value) for value in artifact_metadata.get("development_group_hashes", []))
    )
    if artifact_group_hashes != development_family_hashes:
        raise ValueError("artifact development groups do not match the frozen development CSV")

    old_snapshot = load_blender_raw_snapshot(old_path)
    new_snapshot = load_blender_raw_snapshot(new_path)
    cohort_contract = BlenderCohortContract.from_dict(cohort_payload)
    temporal = diff_blender_snapshots(
        old_snapshot,
        new_snapshot,
        contract=cohort_contract,
        max_submissions=_positive_integer(
            limits.get("maximum_submissions_per_snapshot"),
            label="maximum_submissions_per_snapshot",
        ),
        max_novel_submissions=_positive_integer(
            limits.get("maximum_novel_submissions"), label="maximum_novel_submissions"
        ),
        max_novel_observations=_positive_integer(
            limits.get("maximum_novel_observations"), label="maximum_novel_observations"
        ),
        max_retained_records=_positive_integer(
            limits.get("maximum_retained_cohort_records"),
            label="maximum_retained_cohort_records",
        ),
    )

    prepared_rows: list[dict[str, Any]] = []
    preparation_error: str | None = None
    catalog_cpu_records_loaded = 0
    catalog_total_records_scanned = 0
    if len(temporal.retained_cohort_records) >= 3:
        catalog_records, catalog_total_records_scanned = _read_cpu_catalog(
            catalog_path,
            expected_total_records=_positive_integer(
                catalog_entry.get("expected_total_records"), label="expected_total_records"
            ),
            maximum_total_records=_positive_integer(
                catalog_entry.get("maximum_total_records"), label="maximum_total_records"
            ),
        )
        catalog_cpu_records_loaded = len(catalog_records)
        try:
            prepared_rows, _ = prepare_blender_performance(
                list(temporal.retained_cohort_records),
                catalog_records,
                category_filter="cpu",
                pinned_cohort=(
                    cohort_contract.benchmark_version,
                    cohort_contract.scene,
                    cohort_contract.backend,
                    cohort_contract.operating_system,
                ),
                pinned_execution_identity=BenchmarkExecutionIdentity(
                    cohort_contract.blender_build_hash,
                    cohort_contract.benchmark_script,
                    cohort_contract.scene_checksum,
                ),
                minimum_pilot_products=3,
                minimum_credible_products=_positive_integer(
                    policy.get("minimum_external_leakage_families"),
                    label="minimum_external_leakage_families",
                ),
                source_blockers=[],
                workload=str(development["workload"]),
            )
        except ValueError as exc:
            preparation_error = f"{type(exc).__name__}: {exc}"
    elif temporal.retained_cohort_records:
        preparation_error = "fewer than three exact-cohort observations; aggregation not attempted"

    if len(prepared_rows) > _positive_integer(
        limits.get("maximum_external_rows"), label="maximum_external_rows"
    ):
        raise MemoryError("prepared external rows exceed the protocol limit")
    temporal_summary = temporal.summary()
    strict_superset = bool(temporal_summary["new_snapshot_is_strict_superset"])
    clean_rows, gate, blockers = _external_gate(
        prepared_rows,
        development_families=development_families,
        minimum_rows=_positive_integer(
            policy.get("minimum_external_rows"), label="minimum_external_rows"
        ),
        minimum_families=_positive_integer(
            policy.get("minimum_external_leakage_families"),
            label="minimum_external_leakage_families",
        ),
        strict_superset=strict_superset,
    )
    if not temporal.retained_cohort_records:
        blockers.append("no novel observations match the complete frozen cohort contract")
    if preparation_error is not None:
        blockers.append(preparation_error)
    blockers = sorted(set(blockers))

    evaluation: dict[str, object] | None = None
    model_inference_attempted = False
    if bool(gate["metric_evaluation_eligible"]) and not blockers:
        model_inference_attempted = True
        evaluation = _evaluate_model(
            clean_rows,
            artifact_directory=artifact_directory,
            expected_model_version=str(development["model_version"]),
            expected_features=tuple(str(value) for value in development["feature_columns"]),
            expected_development_group_hashes=development_family_hashes,
            bootstrap_resamples=_positive_integer(
                policy.get("bootstrap_resamples"), label="bootstrap_resamples"
            ),
            bootstrap_confidence_level=float(policy["bootstrap_confidence_level"]),
            bootstrap_seed=int(policy["bootstrap_seed"]),
        )

    status = "metrics_computed" if evaluation is not None else "insufficient_external_cohort"
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task": "blender_cpu_content_creation_temporal_external_audit",
        "effective_at": new_snapshot.retrieved_at.isoformat(),
        "protocol": {
            "path": Path(protocol_path).resolve().relative_to(repository_root).as_posix(),
            "sha256": protocol_sha256,
            "protocol_id": protocol.get("protocol_id"),
            "registration_status": protocol.get("registration_status"),
        },
        "lineage": {
            "old_snapshot": _snapshot_lineage(old_snapshot, repository_root=repository_root),
            "new_snapshot": _snapshot_lineage(new_snapshot, repository_root=repository_root),
            "development_csv": {
                "path": development_csv_path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(development_csv_path),
                "row_count": int(development["row_count"]),
                "leakage_family_count": len(development_families),
                "leakage_family_hashes_sha256": development["leakage_family_hashes_sha256"],
            },
            "development_manifest_sha256": sha256_file(development_manifest_path),
            "buildcores_records_sha256": sha256_file(catalog_path),
            "artifact_manifest_sha256": sha256_file(artifact_directory / "artifact_manifest.json"),
            "artifact_metadata_sha256": sha256_file(artifact_directory / "metadata.json"),
            "artifact_model_sha256": sha256_file(artifact_directory / "model.txt"),
        },
        "cohort_contract": cohort_contract.to_dict(),
        "temporal_diff": temporal_summary,
        "preparation": {
            "catalog_total_records_scanned": catalog_total_records_scanned,
            "catalog_cpu_records_retained": catalog_cpu_records_loaded,
            "prepared_candidate_rows": len(prepared_rows),
            "error": preparation_error,
        },
        "external_gate": gate,
        "status": status,
        "model_inference_attempted": model_inference_attempted,
        "evaluation": evaluation,
        "block_reasons": blockers,
        "claim_boundary": {
            "external_performance_metrics_available": evaluation is not None,
            "supports_model_promotion": False,
            "external_rows_pooled_into_development": False,
            "content_creation_claim_scope": "Blender CPU junkshop rendering only",
        },
        "bounded_memory": {
            "design": "SQLite disk index plus bounded novel/cohort/catalog collections",
            "limits": limits,
        },
    }
    output_path, report_sha256 = _write_content_addressed_report(
        report, Path(output_directory).resolve()
    )
    return output_path, {
        "report_path": str(output_path),
        "report_sha256": report_sha256,
        "status": status,
        "metric_evaluation_eligible": bool(gate["metric_evaluation_eligible"]),
        "model_inference_attempted": model_inference_attempted,
        "retained_cohort_record_count": len(temporal.retained_cohort_records),
        "block_reasons": blockers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, summary = run_temporal_evaluation(
        args.protocol,
        output_directory=args.output_directory,
    )
    print_json(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

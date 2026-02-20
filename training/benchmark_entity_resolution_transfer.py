"""Reproduce the licensed Dn7 entity-resolution transfer benchmark.

This command fetches a pinned Zenodo snapshot, creates strict source-record-disjoint
splits, optionally computes frozen sentence embeddings on CUDA, trains exact/logistic/
LightGBM models, and evaluates once on the held-out test split.  Every emitted report is
explicitly scoped to transfer benchmarking rather than Singapore PC-retailer quality.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pipelines.sources.zenodo_er_benchmark import (
    ZENODO_CREATOR,
    ZENODO_DATASET_NAME,
    ZENODO_DOI,
    ZENODO_LICENSE,
    ZENODO_RECORD_URL,
    ZenodoEntityMatchingDn7Adapter,
    materialize_transfer_pairs,
)
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    precision_recall_curve,
)

from pc_build_recommender.entity_resolution import (
    BaseEntityResolver,
    CanonicalProductRecord,
    ExactMatchBaseline,
    LightGBMEntityResolver,
    ListingRow,
    LogisticMatchBaseline,
    MatchThresholds,
    LabelledPair,
    pair_example_from_dict,
)
from training._common import (
    estimate_materialized_file_memory_mib,
    portable_path_reference,
    print_json,
    read_json_lines,
    require_host_memory_headroom,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_SEED = 20260722
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_materialized_splits(path: Path) -> dict[str, tuple[LabelledPair, ...]]:
    result: dict[str, tuple[LabelledPair, ...]] = {}
    for split_name in ("train", "validation", "test"):
        rows = read_json_lines(path / f"pairs.{split_name}.jsonl")
        result[split_name] = tuple(pair_example_from_dict(row) for row in rows)
    return result


def _embedding_text(record: object) -> str:
    text = getattr(record, "text", "")
    attributes = getattr(record, "attributes", {})
    model_number = getattr(record, "manufacturer_part_number", None)
    dimensions = attributes.get("source_dimensions") if isinstance(attributes, Mapping) else None
    return " | ".join(
        str(value) for value in (text, model_number, dimensions) if value not in (None, "")
    )


def add_frozen_embeddings(
    splits: Mapping[str, Sequence[LabelledPair]],
    *,
    model_name: str,
    revision: str,
    device: str,
    batch_size: int,
) -> tuple[dict[str, tuple[LabelledPair, ...]], dict[str, Any]]:
    """Encode unique records once and reuse immutable record objects across pairs."""

    from sentence_transformers import SentenceTransformer

    listing_records = {
        pair.listing.listing_id: pair.listing for rows in splits.values() for pair in rows
    }
    product_records = {
        pair.product.product_id: pair.product for rows in splits.values() for pair in rows
    }
    record_keys = [
        *(("listing", key) for key in sorted(listing_records)),
        *(("product", key) for key in sorted(product_records)),
    ]
    records = [
        listing_records[key] if side == "listing" else product_records[key]
        for side, key in record_keys
    ]
    model = SentenceTransformer(model_name, revision=revision, device=device)
    matrix = np.asarray(
        model.encode(
            [_embedding_text(record) for record in records],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        ),
        dtype=np.float32,
    )
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise RuntimeError("sentence encoder returned an invalid matrix")

    embedded_listings: dict[str, ListingRow] = {}
    embedded_products: dict[str, CanonicalProductRecord] = {}
    for (side, key), record, embedding in zip(record_keys, records, matrix, strict=True):
        if side == "listing":
            if not isinstance(record, ListingRow):
                raise TypeError("listing embedding key resolved to a non-listing record")
            embedded_listings[key] = replace(
                record, embedding=tuple(float(value) for value in embedding)
            )
        else:
            if not isinstance(record, CanonicalProductRecord):
                raise TypeError("product embedding key resolved to a non-product record")
            embedded_products[key] = replace(
                record, embedding=tuple(float(value) for value in embedding)
            )

    enriched: dict[str, tuple[LabelledPair, ...]] = {}
    for split_name, rows in splits.items():
        enriched[split_name] = tuple(
            replace(
                pair,
                listing=embedded_listings[pair.listing.listing_id],
                product=embedded_products[pair.product.product_id],
            )
            for pair in rows
        )
    return enriched, {
        "enabled": True,
        "model": model_name,
        "revision": revision,
        "model_license": "Apache-2.0 (per the upstream model card)",
        "device": str(model.device),
        "dimension": int(matrix.shape[1]),
        "unique_records_encoded": int(matrix.shape[0]),
        "normalised": True,
    }


def _threshold_candidates(
    labels: NDArray[np.int64], scores: NDArray[np.float64]
) -> list[dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    candidates: list[dict[str, float]] = []
    for index, threshold in enumerate(thresholds):
        p = float(precision[index])
        r = float(recall[index])
        f1 = 0.0 if p + r == 0.0 else 2.0 * p * r / (p + r)
        candidates.append({"threshold": float(threshold), "precision": p, "recall": r, "f1": f1})
    return candidates


def select_validation_thresholds(
    labels: Sequence[int] | NDArray[np.int64],
    scores: Sequence[float] | NDArray[np.float64],
) -> dict[str, dict[str, float] | None]:
    """Select thresholds on validation only; never inspect test labels here."""

    targets = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    candidates = _threshold_candidates(targets, probabilities)
    if not candidates:
        raise ValueError("validation threshold selection requires at least two score values")
    best_f1 = max(candidates, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    precision_candidates = [row for row in candidates if row["precision"] >= 0.99]
    precision_99 = (
        max(precision_candidates, key=lambda row: (row["recall"], row["precision"]))
        if precision_candidates
        else None
    )
    return {"f1_optimised": best_f1, "precision_at_least_0_99": precision_99}


def _lightgbm_candidates(positive_weight: float) -> tuple[dict[str, Any], ...]:
    common = {
        "scale_pos_weight": positive_weight,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    return (
        {
            **common,
            "n_estimators": 350,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 4,
            "min_child_samples": 8,
        },
        {
            **common,
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 8,
        },
        {
            **common,
            "n_estimators": 450,
            "learning_rate": 0.035,
            "num_leaves": 15,
            "max_depth": 6,
            "min_child_samples": 15,
            "reg_alpha": 0.1,
            "reg_lambda": 0.4,
        },
        {
            **common,
            "n_estimators": 550,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "max_depth": 7,
            "min_child_samples": 12,
            "reg_alpha": 0.15,
            "reg_lambda": 0.6,
        },
    )


def tune_lightgbm(
    train: Sequence[LabelledPair],
    validation: Sequence[LabelledPair],
    *,
    seed: int,
    device: str,
    thresholds: MatchThresholds,
) -> tuple[LightGBMEntityResolver, list[dict[str, Any]]]:
    positives = sum(pair.label for pair in train)
    negatives = len(train) - positives
    if not positives or not negatives:
        raise ValueError("LightGBM training requires both labels")
    positive_weight = negatives / positives
    labels = np.asarray([pair.label for pair in validation], dtype=np.int64)
    trials: list[dict[str, Any]] = []
    best_model: LightGBMEntityResolver | None = None
    best_key = (-1.0, -1.0)
    for parameters in _lightgbm_candidates(positive_weight):
        model = LightGBMEntityResolver(
            device=device,
            random_state=seed,
            parameters=parameters,
            thresholds=thresholds,
        ).fit(train, calibrate=False)
        scores = model.predict_proba(validation)
        average_precision = float(average_precision_score(labels, scores))
        selected = select_validation_thresholds(labels, scores)["f1_optimised"]
        assert selected is not None
        trial = {
            "parameters": parameters,
            "validation_average_precision": average_precision,
            "validation_best_f1": selected["f1"],
            "actual_device": model.actual_device,
            "fallback_reason": model.fallback_reason,
        }
        trials.append(trial)
        key = (average_precision, selected["f1"])
        if key > best_key:
            best_key = key
            best_model = model
    assert best_model is not None
    best_model.fit_calibrator(validation)
    return best_model, trials


def _fit_models(
    splits: Mapping[str, Sequence[LabelledPair]],
    *,
    seed: int,
    device: str,
) -> tuple[dict[str, BaseEntityResolver], dict[str, Any]]:
    thresholds = MatchThresholds(auto_match=0.98, manual_review=0.80)
    train = splits["train"]
    validation = splits["validation"]
    exact = ExactMatchBaseline(thresholds=thresholds).fit(train, calibrate=False)
    exact.fit_calibrator(validation)
    logistic = LogisticMatchBaseline(thresholds=thresholds).fit(train, calibrate=False)
    logistic.fit_calibrator(validation)
    lightgbm, trials = tune_lightgbm(
        train,
        validation,
        seed=seed,
        device=device,
        thresholds=thresholds,
    )
    return {"exact": exact, "logistic": logistic, "lightgbm": lightgbm}, {"lightgbm_trials": trials}


def _evaluate_and_save(
    models: Mapping[str, BaseEntityResolver],
    splits: Mapping[str, Sequence[LabelledPair]],
    *,
    artifact_dir: Path,
    dataset_manifest: Path,
) -> dict[str, Any]:
    validation_labels = [pair.label for pair in splits["validation"]]
    evaluated: list[tuple[str, BaseEntityResolver, dict[str, Any]]] = []
    for model_name, model in models.items():
        validation_scores = model.predict_proba(splits["validation"])
        selected = select_validation_thresholds(validation_labels, validation_scores)
        f1_selection = selected["f1_optimised"]
        assert f1_selection is not None
        f1_evaluation = model.evaluate(
            splits["test"], classification_threshold=f1_selection["threshold"]
        )
        precision_selection = selected["precision_at_least_0_99"]
        precision_evaluation = (
            model.evaluate(
                splits["test"], classification_threshold=precision_selection["threshold"]
            )
            if precision_selection is not None
            else None
        )
        report: dict[str, Any] = {
            "validation_selected_thresholds": selected,
            "test_at_validation_f1_threshold": f1_evaluation.to_dict(),
            "test_at_validation_precision_0_99_threshold": (
                precision_evaluation.to_dict() if precision_evaluation is not None else None
            ),
            "claim_scope": "transfer_benchmark_only",
            "pc_retailer_production_claim_eligible": False,
        }
        if isinstance(model, LightGBMEntityResolver):
            report["device"] = {
                "requested": model.requested_device,
                "actual": model.actual_device,
                "fallback_reason": model.fallback_reason,
            }
            report["parameters"] = model.parameters
        evaluated.append((model_name, model, report))

    # Evaluate every model before persisting any artifact. A later model failure (for
    # example an unavailable accelerator learner) must not leave earlier baselines looking
    # like a complete comparable benchmark release.
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for model_name, model, report in evaluated:
        model_path = model.save_artifact(artifact_dir / model_name)
        evidence = {
            "claim_scope": "transfer_benchmark_only",
            "pc_retailer_production_claim_eligible": False,
            "dataset_manifest": portable_path_reference(
                dataset_manifest,
                workspace_root=REPOSITORY_ROOT,
            ),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "leakage_units": ["left_source_record_id", "right_source_record_id"],
            "validation_selected_thresholds": report["validation_selected_thresholds"],
            "test_access_policy": "test labels used once after model and threshold selection",
        }
        write_json(model_path / "transfer_benchmark_evidence.json", evidence)
        report["artifact_path"] = portable_path_reference(
            model_path,
            workspace_root=REPOSITORY_ROOT,
        )
        reports[model_name] = report
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="optional pre-downloaded official Dn7.zip")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/models/er-transfer-dn7")
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tree-device", choices=("cpu", "auto", "gpu"), default="cpu")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_EMBEDDING_REVISION)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--max-host-used-gb",
        type=float,
        default=55.0,
        help="refuse the benchmark when conservative projected host RAM reaches this cap",
    )
    parser.add_argument(
        "--minimum-free-memory-mb",
        type=float,
        default=1024.0,
        help="minimum host RAM that must remain after the conservative allocation",
    )
    parser.add_argument(
        "--materialization-memory-expansion-factor",
        type=float,
        default=16.0,
        help="conservative multiplier for parsed pairs, embeddings, and typed feature objects",
    )
    parser.add_argument(
        "--materialization-runtime-memory-mb",
        type=float,
        default=1024.0,
        help="fixed encoder/learner runtime allowance added to materialized split estimates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.embedding_batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    adapter = ZenodoEntityMatchingDn7Adapter(raw_root=args.raw_root)
    snapshot = adapter.fetch(archive_path=args.archive)
    source_dataset = adapter.parse(snapshot)
    materialized = materialize_transfer_pairs(
        source_dataset,
        snapshot,
        processed_root=args.processed_root,
        seed=args.seed,
    )
    dataset_manifest = materialized / "dataset_manifest.json"
    split_paths = [
        materialized / f"pairs.{split_name}.jsonl" for split_name in ("train", "validation", "test")
    ]
    estimated_materialization_mib = estimate_materialized_file_memory_mib(
        split_paths,
        expansion_factor=args.materialization_memory_expansion_factor,
        runtime_allowance_mib=args.materialization_runtime_memory_mb,
    )
    host_memory_preflight = require_host_memory_headroom(
        max_used_gib=args.max_host_used_gb,
        estimated_additional_mib=estimated_materialization_mib,
        minimum_free_mib=args.minimum_free_memory_mb,
    )
    resource_evidence = {
        "split_input_bytes": sum(path.stat().st_size for path in split_paths),
        "materialization_memory_expansion_factor": args.materialization_memory_expansion_factor,
        "materialization_runtime_memory_mib": args.materialization_runtime_memory_mb,
        "estimated_materialization_mib": estimated_materialization_mib,
        "host_memory_preflight": host_memory_preflight.to_dict(),
    }
    splits = load_materialized_splits(materialized)
    embedding_evidence: dict[str, Any]
    if args.skip_embeddings:
        embedding_evidence = {"enabled": False}
    else:
        splits, embedding_evidence = add_frozen_embeddings(
            splits,
            model_name=args.embedding_model,
            revision=args.embedding_revision,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )

    models, tuning = _fit_models(splits, seed=args.seed, device=args.tree_device)
    model_reports = _evaluate_and_save(
        models,
        splits,
        artifact_dir=args.artifact_dir,
        dataset_manifest=dataset_manifest,
    )
    report = {
        "schema_version": "pc-build-recommender.er-transfer-report.v1",
        "created_at": utc_now_iso(),
        "claim_scope": "transfer_benchmark_only",
        "pc_retailer_production_claim_eligible": False,
        "pc_retailer_production_claim_block_reason": (
            "Dn7 is an external consumer-product benchmark, not labelled Singapore PC listings."
        ),
        "source": {
            "dataset": ZENODO_DATASET_NAME,
            "creator": ZENODO_CREATOR,
            "doi": ZENODO_DOI,
            "record_url": ZENODO_RECORD_URL,
            "license": ZENODO_LICENSE,
            "raw_sha256": snapshot.content_sha256,
            "dataset_manifest": portable_path_reference(
                dataset_manifest,
                workspace_root=REPOSITORY_ROOT,
            ),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
        },
        "resources": resource_evidence,
        "split": {
            split_name: {
                "rows": len(rows),
                "positives": sum(pair.label for pair in rows),
                "left_entities": len({pair.listing.listing_id for pair in rows}),
                "right_entities": len({pair.product.product_id for pair in rows}),
                "left_group_hashes": sorted(
                    {sha256_text(pair.listing.listing_id) for pair in rows}
                ),
                "right_group_hashes": sorted(
                    {sha256_text(pair.product.product_id) for pair in rows}
                ),
            }
            for split_name, rows in splits.items()
        },
        "embeddings": embedding_evidence,
        "tuning": tuning,
        "models": model_reports,
    }
    report_path = args.report or args.artifact_dir / "transfer_benchmark_report.json"
    write_json(report_path, report)
    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

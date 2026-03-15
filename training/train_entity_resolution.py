"""Train and compare exact, logistic, and LightGBM entity resolvers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pc_build_recommender.entity_resolution import (
    ARTIFACT_FORMAT_VERSION,
    FEATURE_NAMES,
    ExactMatchBaseline,
    LightGBMEntityResolver,
    LogisticMatchBaseline,
    MatchThresholds,
    LabelledPair,
    pair_example_from_dict,
)
from pc_build_recommender.evaluation.splits import deterministic_group_split
from training._common import (
    estimate_materialized_file_memory_mib,
    print_json,
    read_json_lines,
    require_host_memory_headroom,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
)
from training.mlflow_tracking import (
    OptionalMLflowRun,
    add_mlflow_arguments,
    promotion_blocker_tags,
    tracking_config_from_args,
)


def _load_pairs(path: Path) -> tuple[LabelledPair, ...]:
    return tuple(pair_example_from_dict(row) for row in read_json_lines(path))


def _split_pairs(pairs: Sequence[LabelledPair], *, seed: int) -> dict[str, tuple[LabelledPair, ...]]:
    """Keep every candidate set for one listing in the same leakage-safe split."""

    group_ids = [pair.listing.listing_id for pair in pairs]
    split = deterministic_group_split(
        group_ids,
        weights={"train": 0.6, "validation": 0.2, "test": 0.2},
        seed=seed,
    )
    result: dict[str, list[LabelledPair]] = {name: [] for name in split.weights}
    for pair, split_name in zip(pairs, split.row_assignments(group_ids), strict=True):
        result[split_name].append(pair)
    for split_name, rows in result.items():
        if len(rows) < 2 or {row.label for row in rows} != {0, 1}:
            raise ValueError(
                f"{split_name} split needs at least two rows and both labels; "
                "add more independently labelled listings"
            )
    return {name: tuple(rows) for name, rows in result.items()}


def _resolver_factories(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = MatchThresholds(
        auto_match=args.auto_match_threshold,
        manual_review=args.manual_review_threshold,
    )
    all_factories = {
        "exact": lambda: ExactMatchBaseline(thresholds=thresholds),
        "logistic": lambda: LogisticMatchBaseline(thresholds=thresholds),
        "lightgbm": lambda: LightGBMEntityResolver(
            thresholds=thresholds,
            device=args.device,
            random_state=args.seed,
        ),
    }
    if args.model == "all":
        return all_factories
    return {args.model: all_factories[args.model]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="nested or flat JSONL pairs")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", choices=("all", "exact", "logistic", "lightgbm"), default="all")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--auto-match-threshold", type=float, default=0.98)
    parser.add_argument("--manual-review-threshold", type=float, default=0.80)
    parser.add_argument(
        "--max-host-used-gb",
        type=float,
        default=55.0,
        help="refuse training when conservative projected host RAM reaches this cap",
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
        default=12.0,
        help="conservative in-memory multiplier for JSON/typed/feature representations",
    )
    parser.add_argument(
        "--materialization-runtime-memory-mb",
        type=float,
        default=512.0,
        help="fixed learner/runtime allowance added to materialized input estimates",
    )
    parser.add_argument(
        "--allow-synthetic-diagnostics",
        action="store_true",
        help="train permanently non-promotable smoke-test artifacts on synthetic pairs",
    )
    add_mlflow_arguments(parser, default_experiment="pcbr-entity-resolution")
    return parser


def _track_training(
    args: argparse.Namespace,
    report_payload: dict[str, Any],
    *,
    report_path: Path,
) -> None:
    blockers = [
        f"{name}: target metrics or evidence gate not met"
        for name, report in report_payload["models"].items()
        if not report["promotion_eligible"]
    ]
    if report_payload["promotion_note"]:
        blockers.append(str(report_payload["promotion_note"]))
    feature_contract = sha256_text("\n".join(FEATURE_NAMES))
    with OptionalMLflowRun(tracking_config_from_args(args)) as tracking:
        tracking.log_params(
            {
                "task": report_payload["task"],
                "dataset": report_payload["input"],
                "split": report_payload["split"],
                "seed": args.seed,
                "thresholds": report_payload["thresholds"],
                "model_selection": args.model,
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "feature_contract_sha256": feature_contract,
                "feature_count": len(FEATURE_NAMES),
            }
        )
        metrics: dict[str, Any] = {}
        for model_name, model_report in report_payload["models"].items():
            metrics.update(
                {
                    f"{model_name}.test.{name}": value
                    for name, value in model_report["metrics"].items()
                }
            )
            device = model_report.get("device")
            if device:
                tracking.log_params({f"{model_name}.device": device})
                tracking.log_tags({f"{model_name}.device_fallback": device.get("fallback_reason")})
        tracking.log_metrics(metrics)
        tracking.log_tags(
            {
                "task": report_payload["task"],
                "feature.version": f"sha256:{feature_contract}",
                **promotion_blocker_tags(blockers),
            }
        )
        tracking.log_dict(
            {"eligible": not blockers, "block_reasons": blockers},
            "evidence/promotion-gate.json",
        )
        report_payload["mlflow_tracking"] = tracking.describe()
        write_json(report_path, report_payload)
        tracking.log_native_artifacts(args.artifact_dir, artifact_path="entity-resolution")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    estimated_materialization_mib = estimate_materialized_file_memory_mib(
        [args.input],
        expansion_factor=args.materialization_memory_expansion_factor,
        runtime_allowance_mib=args.materialization_runtime_memory_mb,
    )
    host_memory_preflight = require_host_memory_headroom(
        max_used_gib=args.max_host_used_gb,
        estimated_additional_mib=estimated_materialization_mib,
        minimum_free_mib=args.minimum_free_memory_mb,
    )
    pairs = _load_pairs(args.input)
    synthetic_count = sum(pair.is_synthetic for pair in pairs)
    if synthetic_count and not args.allow_synthetic_diagnostics:
        raise ValueError(
            f"input contains {synthetic_count} synthetic pairs; pass "
            "--allow-synthetic-diagnostics only for a non-promotable smoke test"
        )
    split = _split_pairs(pairs, seed=args.seed)
    include_synthetic = bool(synthetic_count)

    model_reports: dict[str, dict[str, Any]] = {}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in _resolver_factories(args).items():
        resolver = factory()
        resolver.fit(split["train"], calibrate=False)
        resolver.fit_calibrator(split["validation"])
        evaluation = resolver.evaluate(split["test"], include_synthetic=include_synthetic)
        artifact_path = resolver.save_artifact(args.artifact_dir / name)
        write_json(
            artifact_path / "training_evidence.json",
            {
                "source_sha256": sha256_file(args.input),
                "leakage_unit": "listing_id",
                "training_group_hashes": sorted(
                    {sha256_text(pair.listing.listing_id) for pair in split["train"]}
                ),
                "validation_group_hashes": sorted(
                    {sha256_text(pair.listing.listing_id) for pair in split["validation"]}
                ),
            },
        )
        target_metrics_met = (
            evaluation.precision >= 0.99 and evaluation.recall >= 0.94 and evaluation.f1 >= 0.96
        )
        report: dict[str, Any] = {
            "artifact_path": str(artifact_path.resolve()),
            "metrics": evaluation.to_dict(),
            "target_metrics_met": target_metrics_met,
            "promotion_eligible": evaluation.eligible_for_promotion and target_metrics_met,
        }
        if isinstance(resolver, LightGBMEntityResolver):
            report["device"] = {
                "requested": resolver.requested_device,
                "actual": resolver.actual_device,
                "fallback_reason": resolver.fallback_reason,
            }
        model_reports[name] = report

    report_payload = {
        "created_at": utc_now_iso(),
        "task": "entity_resolution_binary_classification",
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "rows": len(pairs),
            "synthetic_rows": synthetic_count,
        },
        "split": {
            name: {
                "rows": len(rows),
                "listings": len({pair.listing.listing_id for pair in rows}),
                "positives": sum(pair.label for pair in rows),
            }
            for name, rows in split.items()
        },
        "thresholds": {
            "auto_match": args.auto_match_threshold,
            "manual_review": args.manual_review_threshold,
        },
        "resources": {
            "materialized_input_file_bytes": args.input.stat().st_size,
            "materialization_memory_expansion_factor": args.materialization_memory_expansion_factor,
            "materialization_runtime_memory_mb": args.materialization_runtime_memory_mb,
            "estimated_materialization_memory_mb": estimated_materialization_mib,
            "host_memory_preflight": host_memory_preflight.to_dict(),
        },
        "models": model_reports,
        "promotion_note": (
            "synthetic diagnostics cannot support promotion or portfolio claims"
            if synthetic_count
            else None
        ),
    }
    report_path = args.report or args.artifact_dir / "training_report.json"
    write_json(report_path, report_payload)
    _track_training(args, report_payload, report_path=report_path)
    print_json(report_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

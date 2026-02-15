"""Train ER models only from attributable, source-eligible human review queues."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pc_build_recommender.entity_resolution import (
    BaseEntityResolver,
    ExactMatchBaseline,
    LightGBMEntityResolver,
    LogisticMatchBaseline,
    MatchThresholds,
    ReviewQueue,
    build_entity_resolution_serving_evidence,
    calibrate_and_select_from_human_reviews,
    entity_resolution_model_version,
    evaluate_grouped_human_reviews,
    split_human_review_queue,
)
from training._common import (
    estimate_materialized_file_memory_mib,
    print_json,
    require_host_memory_headroom,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
)


def _require_both_labels(name: str, items: Sequence[Any]) -> None:
    labels = {item.to_pair_example().label for item in items}
    if labels != {0, 1}:
        raise ValueError(f"{name} split must contain both human match labels")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", choices=("all", "exact", "logistic", "lightgbm"), default="all")
    parser.add_argument("--device", choices=("cpu", "auto", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--minimum-precision", type=float, default=0.99)
    parser.add_argument("--minimum-predicted-matches", type=int, default=25)
    parser.add_argument("--require-precision-ci-lower-bound", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
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
        help="conservative in-memory multiplier for reviewed JSONL and model feature objects",
    )
    parser.add_argument(
        "--materialization-runtime-memory-mb",
        type=float,
        default=512.0,
        help="fixed learner/runtime allowance added to the reviewed queue estimate",
    )
    return parser


def _factories(args: argparse.Namespace) -> dict[str, Callable[[], BaseEntityResolver]]:
    factories: dict[str, Callable[[], BaseEntityResolver]] = {
        "exact": ExactMatchBaseline,
        "logistic": LogisticMatchBaseline,
        "lightgbm": lambda: LightGBMEntityResolver(
            device=args.device,
            random_state=args.seed,
        ),
    }
    return factories if args.model == "all" else {args.model: factories[args.model]}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.review_queue.is_file():
        raise FileNotFoundError(args.review_queue)
    estimated_materialization_mib = estimate_materialized_file_memory_mib(
        [args.review_queue],
        expansion_factor=args.materialization_memory_expansion_factor,
        runtime_allowance_mib=args.materialization_runtime_memory_mb,
    )
    host_memory_preflight = require_host_memory_headroom(
        max_used_gib=args.max_host_used_gb,
        estimated_additional_mib=estimated_materialization_mib,
        minimum_free_mib=args.minimum_free_memory_mb,
    )
    resource_evidence = {
        "review_queue_bytes": args.review_queue.stat().st_size,
        "materialization_memory_expansion_factor": args.materialization_memory_expansion_factor,
        "materialization_runtime_memory_mib": args.materialization_runtime_memory_mb,
        "estimated_materialization_mib": estimated_materialization_mib,
        "host_memory_preflight": host_memory_preflight.to_dict(),
    }
    queue = ReviewQueue.import_jsonl(args.review_queue)
    if not queue.source_policy.training_eligible:
        raise PermissionError("review queue source policy forbids model training")
    splits = split_human_review_queue(queue, seed=args.seed)
    for split_name, items in splits.items():
        _require_both_labels(split_name, items)

    train_pairs = tuple(item.to_pair_example() for item in splits["train"])
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_reports: dict[str, Any] = {}
    for model_name, factory in _factories(args).items():
        resolver = factory().fit(train_pairs, calibrate=False)
        operating_point = calibrate_and_select_from_human_reviews(
            resolver,
            splits["calibration"],
            splits["threshold"],
            minimum_precision=args.minimum_precision,
            minimum_predicted_matches=args.minimum_predicted_matches,
            require_precision_ci_lower_bound=args.require_precision_ci_lower_bound,
        )
        test_evaluation = None
        test_precision_gate_met = False
        if operating_point.deployment_eligible:
            threshold = operating_point.require_deployable_threshold()
            resolver.thresholds = MatchThresholds(
                auto_match=threshold,
                manual_review=min(0.80, threshold),
            )
            probabilities = resolver.predict_proba(
                tuple(item.to_pair_example() for item in splits["test"])
            )
            test_evaluation = evaluate_grouped_human_reviews(
                splits["test"],
                probabilities,
                threshold=threshold,
                source_policy_reportable=queue.source_policy.published_metrics_eligible,
                evidence_scope=queue.source_policy.scope_note,
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            precision_metric = test_evaluation.evaluation.metric("entity.precision")
            precision_evidence = (
                precision_metric.ci_lower
                if args.require_precision_ci_lower_bound
                else precision_metric.value
            )
            test_precision_gate_met = bool(
                precision_evidence is not None
                and precision_evidence >= args.minimum_precision
                and (precision_metric.denominator or 0) >= args.minimum_predicted_matches
            )
        pairwise_gate_eligible = operating_point.deployment_eligible and test_precision_gate_met
        # Pairwise scores do not evaluate blocking, winner selection, ambiguity margin, or
        # deterministic anchors. They cannot authorize the deployed listing-level matcher.
        deployment_eligible = False
        release_class = "diagnostic"
        artifact_path = resolver.save_artifact(args.artifact_dir / release_class / model_name)
        evidence = {
            "label_source": "attributable_human_reviews",
            "review_queue_sha256": sha256_file(args.review_queue),
            "source_policy": queue.source_policy.to_dict(),
            "training_listing_group_hashes": sorted(
                {sha256_text(item.listing.listing_id) for item in splits["train"]}
            ),
            "calibration_listing_group_hashes": sorted(
                {sha256_text(item.listing.listing_id) for item in splits["calibration"]}
            ),
            "threshold_listing_group_hashes": sorted(
                {sha256_text(item.listing.listing_id) for item in splits["threshold"]}
            ),
            "operating_point": operating_point.to_dict(),
            "test_precision_gate_met": test_precision_gate_met,
            "deployment_eligible": deployment_eligible,
            "pairwise_gate_eligible": pairwise_gate_eligible,
            "resources": resource_evidence,
        }
        write_json(artifact_path / "human_training_evidence.json", evidence)
        model_report: dict[str, Any] = {
            "artifact_path": str(artifact_path.resolve()),
            "operating_point": operating_point.to_dict(),
            "test_evaluation": test_evaluation.to_dict() if test_evaluation else None,
            "test_precision_gate_met": test_precision_gate_met,
            "deployment_eligible": deployment_eligible,
            "pairwise_gate_eligible": pairwise_gate_eligible,
            "catalog_serving_blocker": (
                "pairwise evaluation does not evaluate the deployed listing-level matcher"
            ),
            "published_metrics_eligible": bool(test_evaluation and test_evaluation.reportable),
        }
        if isinstance(resolver, LightGBMEntityResolver):
            frozen_test_groups_sha256 = sha256_text(
                "\n".join(sorted({item.listing.listing_id for item in splits["test"]}))
            )
            serving_evidence = build_entity_resolution_serving_evidence(
                artifact_path,
                dataset_version=queue.source_policy.data_version,
                source_policy=queue.source_policy.to_dict(),
                deployment_eligible=False,
                review_queue_sha256=sha256_file(args.review_queue),
                frozen_test_groups_sha256=frozen_test_groups_sha256,
            )
            write_json(artifact_path / "serving_evidence.json", serving_evidence)
            model_report["model_version"] = entity_resolution_model_version(artifact_path)
            model_report["device"] = {
                "requested": resolver.requested_device,
                "actual": resolver.actual_device,
                "fallback_reason": resolver.fallback_reason,
            }
        model_reports[model_name] = model_report

    report = {
        "schema_version": "pc-build-recommender.er-human-training-report.v1",
        "created_at": utc_now_iso(),
        "review_queue": str(args.review_queue.resolve()),
        "review_queue_sha256": sha256_file(args.review_queue),
        "label_source": "attributable_human_reviews",
        "source_policy": queue.source_policy.to_dict(),
        "resources": resource_evidence,
        "split": {
            name: {
                "rows": len(items),
                "listing_groups": len({item.listing.listing_id for item in items}),
                "positives": sum(item.to_pair_example().label for item in items),
            }
            for name, items in splits.items()
        },
        "models": model_reports,
    }
    report_path = args.report or args.artifact_dir / "training_report.json"
    write_json(report_path, report)
    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

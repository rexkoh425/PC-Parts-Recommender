"""Build immutable listing-level ER diagnostic or promotion-candidate evidence.

This command never creates labels and never trains a model.  It replays the deployed
catalogue matcher on a frozen, independently reviewed PC-domain label set.  A small or
rights-ineligible fixture is written as an explicit diagnostic; only evidence satisfying the
production policy can produce a promotion candidate, and operator rights approval remains a
separate required artifact.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pc_build_recommender.catalog.entity_resolution_evaluation import (
    ER_LISTING_EVALUATION_EVIDENCE_SCHEMA_VERSION,
    evaluate_listing_matcher,
    frozen_listing_groups_sha256,
)
from pc_build_recommender.entity_resolution import (
    ER_EVALUATION_SCHEMA_VERSION_V4,
    ER_LISTING_REVIEW_PROTOCOL,
    EntityResolutionProductionEvaluation,
    EntityResolutionServingEvidence,
    SourceUsePolicy,
    build_entity_resolution_serving_evidence,
    entity_resolution_artifact_sha256,
    entity_resolution_model_version,
    entity_resolution_release_sha256,
    load_entity_resolution_evaluation,
    load_entity_resolution_policy,
    load_entity_resolution_runtime,
    load_frozen_canonical_catalogue,
    load_frozen_listing_label_set,
)
from pc_build_recommender.evaluation.manifest import sha256_json
from training._common import (
    estimate_materialized_file_memory_mib,
    print_json,
    require_host_memory_headroom,
    sha256_file,
    utc_now_iso,
    write_json,
    write_json_lines,
)

ER_LISTING_RELEASE_CANDIDATE_MANIFEST_SCHEMA_VERSION = (
    "pc-build-recommender.er-listing-release-candidate-manifest.v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--canonical-catalogue", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluated-at",
        help="timezone-aware ISO-8601 timestamp; defaults to the current UTC time",
    )
    parser.add_argument("--max-host-used-gb", type=float, default=55.0)
    parser.add_argument("--minimum-free-memory-mb", type=float, default=1024.0)
    parser.add_argument("--materialization-memory-expansion-factor", type=float, default=12.0)
    parser.add_argument("--materialization-runtime-memory-mb", type=float, default=512.0)
    return parser


def _evaluated_at(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("--evaluated-at must be ISO-8601") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("--evaluated-at must include a timezone")
    return result.astimezone(UTC)


def _prepare_destination(path: Path) -> tuple[Path, Path]:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(destination)
        if any(destination.iterdir()):
            raise FileExistsError(
                f"immutable ER evaluation destination is not empty: {destination}"
            )
        destination.rmdir()
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    ).resolve()
    return destination, temporary


def _source_policy_matches_model(
    labels_policy: SourceUsePolicy,
    runtime: EntityResolutionServingEvidence,
) -> bool:
    # Kept out of the core evaluator's public report so the exact comparison stays a
    # release-packaging invariant rather than a caller-selectable metric.
    return bool(
        labels_policy.listing_source == runtime.listing_source
        and labels_policy.catalogue_source == runtime.catalogue_source
        and labels_policy.training_eligible == runtime.source_training_eligible
        and labels_policy.published_metrics_eligible
        == runtime.source_published_metrics_eligible
        and labels_policy.model_serving_eligible == runtime.source_model_serving_eligible
    )


def _manifest_payload(
    *,
    release_class: str,
    created_at: str,
    evaluated_core_sha256: str,
    model_release_sha256: str,
    model_version: str,
    evaluation_summary: dict[str, object],
    files: dict[str, dict[str, object]],
    external_evidence_blockers: list[str],
) -> dict[str, object]:
    return {
        "schema_version": ER_LISTING_RELEASE_CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "created_at": created_at,
        "release_class": release_class,
        "evaluated_artifact_core_sha256": evaluated_core_sha256,
        "model_release_sha256": model_release_sha256,
        "model_version": model_version,
        "evaluation_evidence_schema_version": (
            ER_LISTING_EVALUATION_EVIDENCE_SCHEMA_VERSION
        ),
        "evaluation_summary": evaluation_summary,
        "files": files,
        "external_evidence_blockers": external_evidence_blockers,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for source in (
        args.model_artifact,
        args.labels,
        args.canonical_catalogue,
        args.policy,
    ):
        if not source.exists():
            raise FileNotFoundError(source)
    estimated_materialization_mib = estimate_materialized_file_memory_mib(
        [args.labels, args.canonical_catalogue],
        expansion_factor=args.materialization_memory_expansion_factor,
        runtime_allowance_mib=args.materialization_runtime_memory_mb,
    )
    memory_preflight = require_host_memory_headroom(
        max_used_gib=args.max_host_used_gb,
        estimated_additional_mib=estimated_materialization_mib,
        minimum_free_mib=args.minimum_free_memory_mb,
    )
    labels = load_frozen_listing_label_set(args.labels)
    canonical_catalogue = load_frozen_canonical_catalogue(args.canonical_catalogue)
    canonical_catalogue_file_sha256 = sha256_file(args.canonical_catalogue)
    if labels.canonical_catalogue_file_sha256 != canonical_catalogue_file_sha256:
        raise ValueError("labels do not bind the exact --canonical-catalogue bytes")
    if labels.canonical_catalogue_sha256 != canonical_catalogue.catalogue_sha256:
        raise ValueError("labels do not bind the canonical catalogue semantic release")
    if labels.canonical_catalogue_version != canonical_catalogue.catalogue_version:
        raise ValueError("labels canonical catalogue version does not match release")
    if labels.products != canonical_catalogue.products:
        raise ValueError("labels must contain the complete --canonical-catalogue release")
    policy = load_entity_resolution_policy(args.policy)
    source_runtime = load_entity_resolution_runtime(
        args.model_artifact,
        allow_unpromoted_human_diagnostic=True,
    )

    destination, prepared_temporary = _prepare_destination(args.output_dir)
    temporary: Path | None = prepared_temporary
    try:
        assert temporary is not None
        model_dir = temporary / "model"
        shutil.copytree(args.model_artifact.resolve(), model_dir)
        catalogue_path = temporary / "canonical-catalogue.json"
        shutil.copy2(args.canonical_catalogue.resolve(), catalogue_path)
        labels_path = temporary / "labels.json"
        shutil.copy2(args.labels.resolve(), labels_path)
        policy_path = temporary / "policy.json"
        shutil.copy2(args.policy.resolve(), policy_path)
        serving_evidence = build_entity_resolution_serving_evidence(
            model_dir,
            dataset_version=labels.dataset_version,
            source_policy=labels.source_policy.to_dict(),
            # This legacy field is intentionally non-authoritative. The immutable
            # evaluation, policy, authenticated rights, and replay derive authority.
            deployment_eligible=False,
            review_queue_sha256=labels.source_review_queue_sha256,
            frozen_test_groups_sha256=frozen_listing_groups_sha256(labels),
            end_to_end_matcher_evaluated=True,
        )
        write_json(model_dir / "serving_evidence.json", serving_evidence)
        runtime = load_entity_resolution_runtime(
            model_dir,
            allow_unpromoted_human_diagnostic=True,
        )
        evaluation = evaluate_listing_matcher(labels, runtime, policy)
        blockers = list(evaluation.promotion_blockers)
        if not _source_policy_matches_model(labels.source_policy, source_runtime.evidence):
            blockers.append("label source policy does not match the trained model source policy")
        promotion_eligible = not blockers
        release_class = "promotion_candidate" if promotion_eligible else "diagnostic"
        external_blockers = [
            "operator-signed ER rights approval is not generated by this command",
        ]
        if not promotion_eligible:
            external_blockers.append(
                "listing-level metrics or source permissions do not satisfy production policy"
            )
        decisions_path = temporary / "decisions.jsonl"
        write_json_lines(decisions_path, evaluation.decision_rows)
        release_sha256 = entity_resolution_release_sha256(model_dir)
        model_version = entity_resolution_model_version(model_dir)
        evaluated_at = _evaluated_at(args.evaluated_at)
        evaluation_record = EntityResolutionProductionEvaluation(
            schema_version=ER_EVALUATION_SCHEMA_VERSION_V4,
            evaluation_id=f"er-listing-{labels.dataset_sha256[:20]}",
            dataset_version=labels.dataset_version,
            model_version=model_version,
            label_source="human_reviewed",
            synthetic=False,
            precision=evaluation.auto_match_precision,
            labelled_pair_count=labels.labelled_pair_count,
            evaluated_at=evaluated_at,
            artifact_sha256=release_sha256,
            review_queue_sha256=labels.source_review_queue_sha256,
            frozen_test_groups_sha256=evaluation.frozen_listing_groups_sha256,
            auto_match_threshold=policy.auto_match_threshold,
            precision_numerator=evaluation.auto_match_correct,
            precision_denominator=evaluation.auto_match_count,
            precision_ci_lower=evaluation.auto_match_precision_ci_lower,
            precision_ci_upper=evaluation.auto_match_precision_ci_upper,
            recall=evaluation.auto_match_recall,
            f1=evaluation.auto_match_f1,
            reportable=promotion_eligible,
            deployment_eligible=promotion_eligible,
            label_dataset_sha256=labels.dataset_sha256,
            label_dataset_file_sha256=sha256_file(labels_path),
            decision_rows_sha256=sha256_file(decisions_path),
            matcher_decision_version=evaluation.matcher_decision_version,
            review_protocol=ER_LISTING_REVIEW_PROTOCOL,
            max_candidates=policy.max_candidates,
            minimum_text_score=policy.minimum_text_score,
            minimum_auto_margin=policy.minimum_auto_margin,
            listing_count=evaluation.listing_count,
            independent_reviewer_count=evaluation.independent_reviewer_count,
            candidate_blocking_hits=evaluation.candidate_blocking_hits,
            candidate_blocking_denominator=evaluation.candidate_blocking_denominator,
            candidate_blocking_recall=evaluation.candidate_blocking_recall,
            winner_selection_correct=evaluation.winner_selection_correct,
            winner_selection_denominator=evaluation.winner_selection_denominator,
            winner_selection_accuracy=evaluation.winner_selection_accuracy,
            ambiguity_case_count=evaluation.ambiguity_case_count,
            ambiguity_deferred_count=evaluation.ambiguity_deferred_count,
            ambiguity_false_auto_match_count=(
                evaluation.ambiguity_false_auto_match_count
            ),
            canonical_catalogue_version=evaluation.canonical_catalogue_version,
            canonical_catalogue_sha256=evaluation.canonical_catalogue_sha256,
            canonical_catalogue_file_sha256=(
                evaluation.canonical_catalogue_file_sha256
            ),
            canonical_catalogue_product_count=(
                evaluation.canonical_catalogue_product_count
            ),
            in_catalogue_listing_count=evaluation.in_catalogue_listing_count,
            unmatched_listing_count=evaluation.unmatched_listing_count,
            anchor_auto_match_count=evaluation.anchor_auto_match_count,
            model_route_listing_count=evaluation.model_route_listing_count,
            model_in_catalogue_listing_count=(
                evaluation.model_in_catalogue_listing_count
            ),
            model_unmatched_listing_count=evaluation.model_unmatched_listing_count,
            model_hard_negative_pair_count=evaluation.model_hard_negative_pair_count,
            model_hard_negative_listing_count=(
                evaluation.model_hard_negative_listing_count
            ),
            model_auto_match_correct=evaluation.model_auto_match_correct,
            model_auto_match_count=evaluation.model_auto_match_count,
            model_auto_match_precision=evaluation.model_auto_match_precision,
            model_auto_match_precision_ci_lower=(
                evaluation.model_auto_match_precision_ci_lower
            ),
            model_auto_match_precision_ci_upper=(
                evaluation.model_auto_match_precision_ci_upper
            ),
            model_auto_match_recall=evaluation.model_auto_match_recall,
            model_auto_match_f1=evaluation.model_auto_match_f1,
        )
        evaluation_path = temporary / "evaluation.json"
        write_json(evaluation_path, evaluation_record.to_dict())
        # Parse the exact persisted bytes before publishing the directory.
        load_entity_resolution_evaluation(evaluation_path)

        listing_evaluation_summary = evaluation.summary()
        listing_evaluation_summary.update(
            {
                "model_release_sha256": release_sha256,
                "model_version": model_version,
            }
        )
        evidence_summary = {
            **listing_evaluation_summary,
            "promotion_eligible": promotion_eligible,
            "promotion_blockers": blockers,
            "resource_preflight": memory_preflight.to_dict(),
        }
        files = {
            relative: {
                "sha256": sha256_file(temporary / relative),
                "size_bytes": (temporary / relative).stat().st_size,
            }
            for relative in (
                "canonical-catalogue.json",
                "labels.json",
                "decisions.jsonl",
                "evaluation.json",
                "policy.json",
                "model/metadata.json",
                "model/model.txt",
                "model/serving_evidence.json",
            )
        }
        manifest = _manifest_payload(
            release_class=release_class,
            created_at=utc_now_iso(),
            evaluated_core_sha256=entity_resolution_artifact_sha256(model_dir),
            model_release_sha256=release_sha256,
            model_version=model_version,
            evaluation_summary=evidence_summary,
            files=files,
            external_evidence_blockers=external_blockers,
        )
        manifest["manifest_sha256"] = sha256_json(manifest)
        write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

    printed_evaluation = evaluation.summary()
    printed_evaluation.update(
        {"model_release_sha256": release_sha256, "model_version": model_version}
    )
    print_json(
        {
            "output_dir": str(destination),
            "release_class": release_class,
            "promotion_eligible": promotion_eligible,
            "promotion_blockers": blockers,
            "external_evidence_blockers": external_blockers,
            "model_version": model_version,
            "evaluation": printed_evaluation,
            "resource_preflight": memory_preflight.to_dict(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

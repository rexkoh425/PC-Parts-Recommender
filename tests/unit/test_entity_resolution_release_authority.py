from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pc_build_recommender.catalog import CatalogEntityMatcher, EntityResolutionEvaluation
from pc_build_recommender.catalog.entity_resolution_evaluation import (
    evaluate_listing_matcher,
)
from pc_build_recommender.entity_resolution import (
    ER_CANONICAL_CATALOGUE_SCHEMA_VERSION,
    ER_CATALOG_MATCHER_DECISION_VERSION,
    ER_EVALUATION_SCHEMA_VERSION,
    ER_EVALUATION_SCHEMA_VERSION_V4,
    ER_LISTING_LABEL_DOMAIN,
    ER_LISTING_LABEL_SET_SCHEMA_VERSION,
    ER_LISTING_LABEL_SOURCE,
    ER_LISTING_LABEL_TERRITORY,
    ER_LISTING_REVIEW_PROTOCOL,
    ER_POLICY_SCHEMA_VERSION,
    ER_PRODUCTION_CLAIM_SCOPE,
    ER_REQUIRED_APPROVED_USES,
    ER_RIGHTS_APPROVAL_SCHEMA_VERSION,
    ER_SERVING_PROJECTION_VERSION,
    CanonicalProductRecord,
    EntityResolutionArtifactError,
    EntityResolutionContractError,
    LightGBMEntityResolver,
    build_entity_resolution_serving_evidence,
    canonical_catalogue_sha256,
    entity_resolution_file_sha256,
    entity_resolution_release_sha256,
    load_entity_resolution_policy,
    load_entity_resolution_release,
    load_entity_resolution_runtime,
    load_frozen_listing_label_set,
    seal_entity_resolution_policy,
    seal_entity_resolution_rights_approval,
    synthetic_pairs,
)
from pc_build_recommender.evaluation.manifest import sha256_json

_REVIEW_SHA = "a" * 64
_LISTING_IDS = tuple(f"listing-{index:03d}" for index in range(100))
_TEST_GROUP_SHA = hashlib.sha256("\n".join(_LISTING_IDS).encode("utf-8")).hexdigest()
_AS_OF = datetime(2026, 7, 23, 12, tzinfo=UTC)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _policy_payload(**updates: object) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ER_POLICY_SCHEMA_VERSION,
        "policy_id": "er-production-policy-fixture-v1",
        "claim_scope": ER_PRODUCTION_CLAIM_SCOPE,
        "territory": "SG",
        "required_label_source": "human_reviewed",
        "required_model_type": LightGBMEntityResolver.model_type,
        "required_matcher_decision_version": ER_CATALOG_MATCHER_DECISION_VERSION,
        "required_serving_projection_version": ER_SERVING_PROJECTION_VERSION,
        "minimum_precision": 0.99,
        "minimum_labelled_pairs": 2500,
        "minimum_auto_matches": 100,
        "minimum_recall": 0.94,
        "minimum_f1": 0.96,
        "auto_match_threshold": 0.98,
        "manual_review_threshold": 0.80,
        "max_candidates": 50,
        "minimum_text_score": 0.12,
        "minimum_auto_margin": 0.02,
        "evidence_candidate_limit": 5,
        "minimum_products": 750,
        "minimum_products_per_category": 1,
        "minimum_mapping_rate": 0.80,
        "minimum_critical_field_rate": 0.90,
        "require_complete_priced_coverage": True,
        "require_complete_in_stock_coverage": True,
        "require_complete_product_provenance": True,
        "require_complete_offer_provenance": True,
        "require_explicit_offer_rights": True,
        "require_production_offer_rights": True,
        "require_complete_listing_provenance": True,
        "require_promoted_entity_resolution_model": True,
    }
    payload.update(updates)
    return seal_entity_resolution_policy(payload)


def _create_listing_evidence(root: Path) -> tuple[str, str, str]:
    products = [
        {
            "product_id": f"gpu-{index:02d}",
            "category": "gpu",
            "brand": "Aster",
            "model": f"Nova {index:02d}",
            "canonical_name": f"Aster Nova {index:02d} 16GB",
            "manufacturer_part_number": f"GPU-{index:02d}",
            "gtin": None,
            "attributes": {"vram_gb": 16},
            "price_sgd": None,
            "embedding": None,
            "is_synthetic": False,
        }
        for index in range(25)
    ]
    canonical_catalogue_version = "pc-er-canonical-fixture-v1"
    canonical_products = tuple(
        CanonicalProductRecord.from_dict(product) for product in products
    )
    canonical_catalogue_digest = canonical_catalogue_sha256(
        canonical_catalogue_version, canonical_products
    )
    canonical_catalogue_path = root / "canonical-catalogue.json"
    _write_json(
        canonical_catalogue_path,
        {
            "schema_version": ER_CANONICAL_CATALOGUE_SCHEMA_VERSION,
            "catalogue_version": canonical_catalogue_version,
            "products": products,
            "catalogue_sha256": canonical_catalogue_digest,
        },
    )
    listing_groups: list[dict[str, Any]] = []
    for listing_index, listing_id in enumerate(_LISTING_IDS):
        gold_product_id = f"gpu-{listing_index % 25:02d}"
        pair_labels = []
        for product in products:
            product_id = str(product["product_id"])
            label = "MATCH" if product_id == gold_product_id else "NON_MATCH"
            pair_labels.append(
                {
                    "product_id": product_id,
                    "judgments": [
                        {
                            "reviewer_id": "reviewer-a",
                            "assignment_id": f"{listing_id}-{product_id}-a",
                            "label": label,
                            "reviewed_at": "2026-07-20T12:00:00+00:00",
                            "evidence_reference": f"fixture://{listing_id}/{product_id}/a",
                        },
                        {
                            "reviewer_id": "reviewer-b",
                            "assignment_id": f"{listing_id}-{product_id}-b",
                            "label": label,
                            "reviewed_at": "2026-07-20T12:01:00+00:00",
                            "evidence_reference": f"fixture://{listing_id}/{product_id}/b",
                        },
                    ],
                    "adjudication": None,
                    "resolved_label": label,
                }
            )
        listing_groups.append(
            {
                "listing": {
                    "listing_id": listing_id,
                    "title": f"Aster Nova {listing_index % 25:02d} 16GB",
                    "category": "gpu",
                    "brand": "Aster",
                    "manufacturer_part_number": f"GPU-{listing_index % 25:02d}",
                    "gtin": None,
                    "attributes": {},
                    "current_price_sgd": 799.0,
                    "embedding": None,
                    "retailer": "Approved Retailer",
                    "is_synthetic": False,
                },
                "match_disposition": "in_catalogue_match",
                "pair_labels": pair_labels,
            }
        )
    content: dict[str, Any] = {
        "schema_version": ER_LISTING_LABEL_SET_SCHEMA_VERSION,
        "dataset_version": "human-sg-pc-er-v1",
        "territory": ER_LISTING_LABEL_TERRITORY,
        "domain": ER_LISTING_LABEL_DOMAIN,
        "label_source": ER_LISTING_LABEL_SOURCE,
        "review_protocol": ER_LISTING_REVIEW_PROTOCOL,
        "created_at": "2026-07-20T12:00:00+00:00",
        "source_review_queue_sha256": _REVIEW_SHA,
        "canonical_catalogue_version": canonical_catalogue_version,
        "canonical_catalogue_sha256": canonical_catalogue_digest,
        "canonical_catalogue_file_sha256": entity_resolution_file_sha256(
            canonical_catalogue_path
        ),
        "source_policy": {
            "listing_source": "operator-approved-retailer-corpus",
            "catalogue_source": "operator-approved-manufacturer-corpus",
            "data_version": "human-sg-pc-er-v1",
            "training_eligible": True,
            "published_metrics_eligible": True,
            "model_serving_eligible": True,
            "scope_note": "test fixture whose approval is pinned separately",
        },
        "products": products,
        "listing_groups": listing_groups,
    }
    dataset_sha256 = sha256_json(content)
    labels_path = root / "labels.json"
    _write_json(labels_path, {**content, "dataset_sha256": dataset_sha256})
    return dataset_sha256, entity_resolution_file_sha256(labels_path)


def _create_release(root: Path) -> tuple[Path, Path, Path, Path]:
    label_dataset_sha, label_file_sha = _create_listing_evidence(root)
    model_dir = root / "model"
    resolver = LightGBMEntityResolver(device="cpu", random_state=17)
    resolver.fit(synthetic_pairs(seed=17, product_count=12), calibrate=True)
    resolver.save_artifact(model_dir)
    evidence = build_entity_resolution_serving_evidence(
        model_dir,
        dataset_version="human-sg-pc-er-v1",
        source_policy={
            "listing_source": "operator-approved-retailer-corpus",
            "catalogue_source": "operator-approved-manufacturer-corpus",
            "data_version": "human-sg-pc-er-v1",
            "training_eligible": True,
            "published_metrics_eligible": True,
            "model_serving_eligible": True,
            "scope_note": "test fixture whose approval is pinned separately",
        },
        deployment_eligible=False,
        review_queue_sha256=_REVIEW_SHA,
        frozen_test_groups_sha256=_TEST_GROUP_SHA,
    )
    _write_json(model_dir / "serving_evidence.json", evidence)

    release_sha = entity_resolution_release_sha256(model_dir)
    model_version = f"er-lightgbm-{release_sha[:16]}"

    # Production authority is derived from a deterministic replay of the pinned
    # matcher, so the fixture records what that replay actually produces instead
    # of hand-maintaining a parallel copy that silently drifts from the schema.
    policy_path = root / "policy.json"
    policy = _policy_payload()
    _write_json(policy_path, policy)
    replay = evaluate_listing_matcher(
        load_frozen_listing_label_set(root / "labels.json"),
        load_entity_resolution_runtime(model_dir, allow_unpromoted_human_diagnostic=True),
        load_entity_resolution_policy(policy_path),
    )
    decisions_path = root / "decisions.jsonl"
    decisions_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in replay.decision_rows
        ),
        encoding="utf-8",
    )
    decision_rows_sha = entity_resolution_file_sha256(decisions_path)

    evaluation_path = root / "evaluation.json"
    _write_json(
        evaluation_path,
        {
            "schema_version": ER_EVALUATION_SCHEMA_VERSION_V4,
            "evaluation_id": "er-held-out-human-eval-v1",
            "dataset_version": "human-sg-pc-er-v1",
            "model_version": model_version,
            "label_source": "human_reviewed",
            "synthetic": False,
            "precision": replay.auto_match_precision,
            "labelled_pair_count": replay.labelled_pair_count,
            "evaluated_at": "2026-07-21T12:00:00+00:00",
            "artifact_sha256": release_sha,
            "review_queue_sha256": _REVIEW_SHA,
            "frozen_test_groups_sha256": _TEST_GROUP_SHA,
            "auto_match_threshold": 0.98,
            "precision_numerator": replay.auto_match_correct,
            "precision_denominator": replay.auto_match_count,
            "precision_ci_lower": replay.auto_match_precision_ci_lower,
            "precision_ci_upper": replay.auto_match_precision_ci_upper,
            "recall": replay.auto_match_recall,
            "f1": replay.auto_match_f1,
            # Legacy flags are intentionally false: policy + rights derive authority.
            "reportable": False,
            "deployment_eligible": False,
            "label_dataset_sha256": label_dataset_sha,
            "label_dataset_file_sha256": label_file_sha,
            "decision_rows_sha256": decision_rows_sha,
            "matcher_decision_version": ER_CATALOG_MATCHER_DECISION_VERSION,
            "review_protocol": ER_LISTING_REVIEW_PROTOCOL,
            "max_candidates": 50,
            "minimum_text_score": 0.12,
            "minimum_auto_margin": 0.02,
            "listing_count": replay.listing_count,
            "independent_reviewer_count": replay.independent_reviewer_count,
            "candidate_blocking_hits": replay.candidate_blocking_hits,
            "candidate_blocking_denominator": replay.candidate_blocking_denominator,
            "candidate_blocking_recall": replay.candidate_blocking_recall,
            "winner_selection_correct": replay.winner_selection_correct,
            "winner_selection_denominator": replay.winner_selection_denominator,
            "winner_selection_accuracy": replay.winner_selection_accuracy,
            "ambiguity_case_count": replay.ambiguity_case_count,
            "ambiguity_deferred_count": replay.ambiguity_deferred_count,
            "ambiguity_false_auto_match_count": replay.ambiguity_false_auto_match_count,
            "canonical_catalogue_version": replay.canonical_catalogue_version,
            "canonical_catalogue_sha256": replay.canonical_catalogue_sha256,
            "canonical_catalogue_file_sha256": replay.canonical_catalogue_file_sha256,
            "canonical_catalogue_product_count": replay.canonical_catalogue_product_count,
            "in_catalogue_listing_count": replay.in_catalogue_listing_count,
            "unmatched_listing_count": replay.unmatched_listing_count,
            "anchor_auto_match_count": replay.anchor_auto_match_count,
            "model_route_listing_count": replay.model_route_listing_count,
            "model_in_catalogue_listing_count": replay.model_in_catalogue_listing_count,
            "model_unmatched_listing_count": replay.model_unmatched_listing_count,
            "model_hard_negative_pair_count": replay.model_hard_negative_pair_count,
            "model_hard_negative_listing_count": replay.model_hard_negative_listing_count,
            "model_auto_match_correct": replay.model_auto_match_correct,
            "model_auto_match_count": replay.model_auto_match_count,
            "model_auto_match_precision": replay.model_auto_match_precision,
            "model_auto_match_precision_ci_lower": replay.model_auto_match_precision_ci_lower,
            "model_auto_match_precision_ci_upper": replay.model_auto_match_precision_ci_upper,
            "model_auto_match_recall": replay.model_auto_match_recall,
            "model_auto_match_f1": replay.model_auto_match_f1,
        },
    )
    rights_path = root / "rights.json"
    rights = seal_entity_resolution_rights_approval(
        {
            "schema_version": ER_RIGHTS_APPROVAL_SCHEMA_VERSION,
            "approval_id": "er-rights-approval-fixture-v1",
            "decision": "approved",
            "dataset_version": "human-sg-pc-er-v1",
            "model_version": model_version,
            "model_release_sha256": release_sha,
            "evaluation_sha256": entity_resolution_file_sha256(evaluation_path),
            "policy_sha256": policy["policy_sha256"],
            "review_queue_sha256": _REVIEW_SHA,
            "frozen_test_groups_sha256": _TEST_GROUP_SHA,
            "approved_uses": sorted(ER_REQUIRED_APPROVED_USES),
            "territories": ["SG"],
            "approved_by": "fixture-compliance-owner",
            "approved_at": "2026-07-22T12:00:00+00:00",
            "expires_at": "2027-07-22T12:00:00+00:00",
            "source_contract_sha256s": ["c" * 64],
            "evidence_references": ["contract://fixture/approved-source-v1"],
        }
    )
    _write_json(rights_path, rights)
    return model_dir, evaluation_path, policy_path, rights_path


@pytest.fixture
def release_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    return _create_release(tmp_path)


def _copy_release(
    release_paths: tuple[Path, Path, Path, Path], target: Path
) -> tuple[Path, Path, Path, Path]:
    model_dir, evaluation, policy, rights = release_paths
    copied_model = target / "model"
    shutil.copytree(model_dir, copied_model)
    copied_evaluation = target / "evaluation.json"
    copied_labels = target / "labels.json"
    copied_catalogue = target / "canonical-catalogue.json"
    copied_decisions = target / "decisions.jsonl"
    copied_policy = target / "policy.json"
    copied_rights = target / "rights.json"
    shutil.copy2(evaluation, copied_evaluation)
    shutil.copy2(evaluation.parent / "labels.json", copied_labels)
    shutil.copy2(evaluation.parent / "canonical-catalogue.json", copied_catalogue)
    shutil.copy2(evaluation.parent / "decisions.jsonl", copied_decisions)
    shutil.copy2(policy, copied_policy)
    shutil.copy2(rights, copied_rights)
    return copied_model, copied_evaluation, copied_policy, copied_rights


def test_complete_release_derives_authority_without_eligibility_booleans(
    release_paths: tuple[Path, Path, Path, Path],
) -> None:
    release = load_entity_resolution_release(*release_paths, as_of=_AS_OF)

    assert release.runtime.production_authorized is True
    assert release.runtime.release_identity == release.identity
    assert release.evaluation.reportable is False
    assert release.evaluation.deployment_eligible is False
    assert release.identity.model_release_sha256 == entity_resolution_release_sha256(
        release_paths[0]
    )
    assert release.identity.calibrator_sha256 != release.identity.model_file_sha256
    assert release.policy.matcher_kwargs() == {
        "max_candidates": 50,
        "minimum_text_score": 0.12,
        "minimum_auto_margin": 0.02,
        "evidence_candidate_limit": 5,
    }
    assert release.policy.production_catalog_policy_kwargs()["minimum_er_precision"] == 0.99
    assert (
        release.runtime.authorize_for_production(
            release.evaluation,
            minimum_precision=release.policy.minimum_precision,
            minimum_labelled_pairs=release.policy.minimum_labelled_pairs,
            minimum_auto_matches=release.policy.minimum_auto_matches,
            minimum_recall=release.policy.minimum_recall,
            minimum_f1=release.policy.minimum_f1,
        )
        is release.runtime
    )
    CatalogEntityMatcher((), runtime=release.runtime, **release.policy.matcher_kwargs())
    with pytest.raises(ValueError, match="settings do not match"):
        CatalogEntityMatcher(
            (),
            runtime=release.runtime,
            max_candidates=release.policy.max_candidates,
            minimum_text_score=release.policy.minimum_text_score,
            minimum_auto_margin=0.03,
            evidence_candidate_limit=release.policy.evidence_candidate_limit,
        )


def test_direct_evaluation_object_cannot_promote_diagnostic_runtime(
    release_paths: tuple[Path, Path, Path, Path],
) -> None:
    runtime = load_entity_resolution_runtime(
        release_paths[0], allow_unpromoted_human_diagnostic=True
    )
    evaluation_payload = json.loads(release_paths[1].read_text(encoding="utf-8"))
    evaluation = EntityResolutionEvaluation(
        **{
            **evaluation_payload,
            "evaluated_at": datetime.fromisoformat(evaluation_payload["evaluated_at"]),
        }
    )

    with pytest.raises(EntityResolutionArtifactError, match="direct.*authorization is disabled"):
        runtime.authorize_for_production(
            evaluation,
            minimum_precision=0.99,
            minimum_labelled_pairs=2500,
        )


def test_review_queue_and_frozen_groups_must_match_all_three_artifacts(
    release_paths: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    model, evaluation_path, policy, rights_path = _copy_release(
        release_paths, tmp_path / "mismatch"
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["review_queue_sha256"] = "d" * 64
    _write_json(evaluation_path, evaluation)
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    rights["review_queue_sha256"] = "d" * 64
    rights["evaluation_sha256"] = entity_resolution_file_sha256(evaluation_path)
    rights = seal_entity_resolution_rights_approval(rights)
    _write_json(rights_path, rights)

    with pytest.raises(EntityResolutionArtifactError, match="review_queue_sha256"):
        load_entity_resolution_release(model, evaluation_path, policy, rights_path, as_of=_AS_OF)


def test_policy_and_rights_content_hashes_reject_tampering(
    release_paths: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    model, evaluation, policy_path, rights = _copy_release(
        release_paths, tmp_path / "policy-tamper"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["minimum_precision"] = 0.50
    _write_json(policy_path, policy)

    with pytest.raises(EntityResolutionContractError, match="policy SHA-256"):
        load_entity_resolution_release(model, evaluation, policy_path, rights, as_of=_AS_OF)

    model, evaluation, policy_path, rights_path = _copy_release(
        release_paths, tmp_path / "rights-tamper"
    )
    rights_payload = json.loads(rights_path.read_text(encoding="utf-8"))
    rights_payload["approved_by"] = "unreviewed-caller"
    _write_json(rights_path, rights_payload)

    with pytest.raises(EntityResolutionContractError, match="rights approval SHA-256"):
        load_entity_resolution_release(model, evaluation, policy_path, rights_path, as_of=_AS_OF)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("minimum_precision", 0.98),
        ("minimum_labelled_pairs", 2499),
        ("minimum_auto_matches", 99),
        ("minimum_recall", 0.93),
        ("minimum_f1", 0.95),
        ("auto_match_threshold", 0.97),
        ("minimum_products", 749),
        ("minimum_products_per_category", 0),
        ("minimum_mapping_rate", 0.79),
        ("minimum_critical_field_rate", 0.89),
        ("require_complete_offer_provenance", False),
        ("require_production_offer_rights", False),
        ("require_promoted_entity_resolution_model", False),
        ("territory", "US"),
    ],
)
def test_content_hashing_cannot_self_attest_a_weakened_production_policy(
    field: str,
    unsafe_value: object,
) -> None:
    with pytest.raises(EntityResolutionContractError, match="non-negotiable floors"):
        _policy_payload(**{field: unsafe_value})


def test_rights_must_bind_exact_evaluation_and_required_uses(
    release_paths: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    model, evaluation, policy, rights_path = _copy_release(
        release_paths, tmp_path / "wrong-evaluation"
    )
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    rights["evaluation_sha256"] = "e" * 64
    rights = seal_entity_resolution_rights_approval(rights)
    _write_json(rights_path, rights)

    with pytest.raises(EntityResolutionArtifactError, match="evaluation digest"):
        load_entity_resolution_release(model, evaluation, policy, rights_path, as_of=_AS_OF)

    model, evaluation, policy, rights_path = _copy_release(release_paths, tmp_path / "missing-use")
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    rights["approved_uses"].remove("serve_derived_model")
    rights = seal_entity_resolution_rights_approval(rights)
    _write_json(rights_path, rights)

    with pytest.raises(EntityResolutionContractError, match="missing required uses"):
        load_entity_resolution_release(model, evaluation, policy, rights_path, as_of=_AS_OF)


def test_model_or_embedded_calibrator_tamper_fails_closed(
    release_paths: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    model, evaluation, policy, rights = _copy_release(release_paths, tmp_path / "calibrator-tamper")
    metadata_path = model / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["calibrator"]["intercept"] += 0.001
    _write_json(metadata_path, metadata)

    with pytest.raises(EntityResolutionArtifactError, match="does not match artifact bytes"):
        load_entity_resolution_release(model, evaluation, policy, rights, as_of=_AS_OF)


def test_listing_decisions_are_recomputed_not_only_hashed(
    release_paths: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    model, evaluation_path, policy, rights_path = _copy_release(
        release_paths, tmp_path / "decision-tamper"
    )
    decisions_path = evaluation_path.parent / "decisions.jsonl"
    rows = decisions_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["winner_correct"] = False
    rows[0] = json.dumps(first, sort_keys=True)
    decisions_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["decision_rows_sha256"] = entity_resolution_file_sha256(decisions_path)
    _write_json(evaluation_path, evaluation)
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    rights["evaluation_sha256"] = entity_resolution_file_sha256(evaluation_path)
    rights = seal_entity_resolution_rights_approval(rights)
    _write_json(rights_path, rights)

    # Re-hashing the tampered file is not enough: authority comes from replaying the
    # pinned matcher, so a row that the replay would never produce is rejected even
    # though every recorded digest agrees with the bytes on disk.
    with pytest.raises(
        EntityResolutionArtifactError, match="does not match pinned matcher/model replay"
    ):
        load_entity_resolution_release(model, evaluation_path, policy, rights_path, as_of=_AS_OF)


def test_legacy_v2_pairwise_evaluation_cannot_authorize_listing_matcher(
    release_paths: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    model, evaluation_path, policy, rights_path = _copy_release(
        release_paths, tmp_path / "legacy-v2"
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["schema_version"] = "pc-build-recommender.er-production-evaluation.v2"
    for field in (
        "label_dataset_sha256",
        "label_dataset_file_sha256",
        "decision_rows_sha256",
        "matcher_decision_version",
        "review_protocol",
        "max_candidates",
        "minimum_text_score",
        "minimum_auto_margin",
        "listing_count",
        "independent_reviewer_count",
        "candidate_blocking_hits",
        "candidate_blocking_denominator",
        "candidate_blocking_recall",
        "winner_selection_correct",
        "winner_selection_denominator",
        "winner_selection_accuracy",
        "ambiguity_case_count",
        "ambiguity_deferred_count",
        "ambiguity_false_auto_match_count",
        # v4 additions: a legacy v2 payload carries none of the listing-matcher or
        # canonical-catalogue evidence either.
        "canonical_catalogue_version",
        "canonical_catalogue_sha256",
        "canonical_catalogue_file_sha256",
        "canonical_catalogue_product_count",
        "in_catalogue_listing_count",
        "unmatched_listing_count",
        "anchor_auto_match_count",
        "model_route_listing_count",
        "model_in_catalogue_listing_count",
        "model_unmatched_listing_count",
        "model_hard_negative_pair_count",
        "model_hard_negative_listing_count",
        "model_auto_match_correct",
        "model_auto_match_count",
        "model_auto_match_precision",
        "model_auto_match_precision_ci_lower",
        "model_auto_match_precision_ci_upper",
        "model_auto_match_recall",
        "model_auto_match_f1",
    ):
        evaluation.pop(field)
    _write_json(evaluation_path, evaluation)
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    rights["evaluation_sha256"] = entity_resolution_file_sha256(evaluation_path)
    rights = seal_entity_resolution_rights_approval(rights)
    _write_json(rights_path, rights)

    with pytest.raises(EntityResolutionArtifactError, match="schema is not production v4"):
        load_entity_resolution_release(model, evaluation_path, policy, rights_path, as_of=_AS_OF)


def test_synthetic_and_v1_evaluation_artifacts_are_never_production_authority(
    release_paths: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    legacy = EntityResolutionEvaluation(
        schema_version=ER_EVALUATION_SCHEMA_VERSION,
        evaluation_id="legacy",
        dataset_version="legacy",
        model_version="legacy",
        label_source="human_reviewed",
        synthetic=False,
        precision=1.0,
        labelled_pair_count=10_000,
        evaluated_at=_AS_OF,
    )
    assert legacy.blockers(minimum_precision=0.99, minimum_labelled_pairs=2500) == (
        "entity-resolution evaluation schema is not production v4",
    )

    model, evaluation, policy, rights = _copy_release(release_paths, tmp_path / "synthetic")
    evidence_path = model / "serving_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["synthetic_rows"] = 1
    # Keep the model core reference intact; release/evaluation rights still cannot override
    # the explicit synthetic provenance check in the diagnostic loader.
    _write_json(evidence_path, evidence)

    with pytest.raises(EntityResolutionArtifactError, match="synthetic"):
        load_entity_resolution_release(model, evaluation, policy, rights, as_of=_AS_OF)


def test_authority_contracts_reject_duplicate_keys_and_nonfinite_numbers(
    release_paths: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    model, evaluation, policy, rights = _copy_release(release_paths, tmp_path / "duplicate-key")
    policy.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    with pytest.raises(EntityResolutionContractError, match="duplicate JSON key"):
        load_entity_resolution_release(model, evaluation, policy, rights, as_of=_AS_OF)

    model, evaluation, policy, rights = _copy_release(release_paths, tmp_path / "nonfinite")
    raw = evaluation.read_text(encoding="utf-8").replace('"precision": 1.0', '"precision": NaN')
    evaluation.write_text(raw, encoding="utf-8")
    with pytest.raises(EntityResolutionContractError, match="non-finite JSON"):
        load_entity_resolution_release(model, evaluation, policy, rights, as_of=_AS_OF)

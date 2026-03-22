from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pc_build_recommender.catalog import CatalogEntityMatcher, EntityResolutionEvaluation
from pc_build_recommender.entity_resolution import (
    ER_CATALOG_MATCHER_DECISION_VERSION,
    ER_EVALUATION_SCHEMA_VERSION,
    ER_EVALUATION_SCHEMA_VERSION_V2,
    ER_POLICY_SCHEMA_VERSION,
    ER_PRODUCTION_CLAIM_SCOPE,
    ER_REQUIRED_APPROVED_USES,
    ER_RIGHTS_APPROVAL_SCHEMA_VERSION,
    ER_SERVING_PROJECTION_VERSION,
    EntityResolutionArtifactError,
    EntityResolutionContractError,
    LightGBMEntityResolver,
    build_entity_resolution_serving_evidence,
    entity_resolution_file_sha256,
    entity_resolution_release_sha256,
    load_entity_resolution_release,
    load_entity_resolution_runtime,
    seal_entity_resolution_policy,
    seal_entity_resolution_rights_approval,
    synthetic_pairs,
)

_REVIEW_SHA = "a" * 64
_TEST_GROUP_SHA = "b" * 64
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
        "minimum_labelled_pairs": 1000,
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


def _create_release(root: Path) -> tuple[Path, Path, Path, Path]:
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
    evaluation_path = root / "evaluation.json"
    _write_json(
        evaluation_path,
        {
            "schema_version": ER_EVALUATION_SCHEMA_VERSION_V2,
            "evaluation_id": "er-held-out-human-eval-v1",
            "dataset_version": "human-sg-pc-er-v1",
            "model_version": model_version,
            "label_source": "human_reviewed",
            "synthetic": False,
            "precision": 1.0,
            "labelled_pair_count": 1000,
            "evaluated_at": "2026-07-21T12:00:00+00:00",
            "artifact_sha256": release_sha,
            "review_queue_sha256": _REVIEW_SHA,
            "frozen_test_groups_sha256": _TEST_GROUP_SHA,
            "auto_match_threshold": 0.98,
            "precision_numerator": 100,
            "precision_denominator": 100,
            "precision_ci_lower": 0.99,
            "precision_ci_upper": 1.0,
            "recall": 0.95,
            "f1": 0.97,
            # Legacy flags are intentionally false: policy + rights derive authority.
            "reportable": False,
            "deployment_eligible": False,
        },
    )
    policy_path = root / "policy.json"
    policy = _policy_payload()
    _write_json(policy_path, policy)
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
    copied_policy = target / "policy.json"
    copied_rights = target / "rights.json"
    shutil.copy2(evaluation, copied_evaluation)
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
            minimum_labelled_pairs=1000,
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
        ("minimum_labelled_pairs", 999),
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
    assert legacy.blockers(minimum_precision=0.99, minimum_labelled_pairs=1000) == (
        "entity-resolution evaluation schema is not production v2",
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

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from training._common import sha256_file, write_json
from training.evaluate_entity_resolution_listing import main

from pc_build_recommender.catalog.entity_resolution_evaluation import (
    evaluate_listing_matcher,
    frozen_listing_groups_sha256,
)
from pc_build_recommender.entity_resolution import (
    ER_CANONICAL_CATALOGUE_SCHEMA_VERSION,
    ER_CATALOG_MATCHER_DECISION_VERSION,
    ER_HUMAN_LABEL_SOURCE,
    ER_LISTING_LABEL_DOMAIN,
    ER_LISTING_LABEL_SET_SCHEMA_VERSION,
    ER_LISTING_LABEL_SOURCE,
    ER_LISTING_LABEL_TERRITORY,
    ER_LISTING_REVIEW_PROTOCOL,
    ER_POLICY_SCHEMA_VERSION,
    ER_PRODUCTION_CLAIM_SCOPE,
    ER_SERVING_PROJECTION_VERSION,
    CanonicalProductRecord,
    EntityResolutionPolicy,
    EntityResolutionRuntime,
    EntityResolutionServingEvidence,
    FrozenListingLabelSet,
    LightGBMEntityResolver,
    ListingLabelSetError,
    ListingRecord,
    MatchThresholds,
    SourceUsePolicy,
    build_entity_resolution_serving_evidence,
    canonical_catalogue_sha256,
    load_entity_resolution_evaluation,
    load_frozen_listing_label_set,
    seal_entity_resolution_policy,
    synthetic_pairs,
)
from pc_build_recommender.evaluation.manifest import sha256_json

_QUEUE_SHA = "a" * 64
_CREATED_AT = "2026-08-15T10:00:00+08:00"
_CANONICAL_CATALOGUE_VERSION = "pc-er-canonical-test-v1"


def _product(product_id: str, model: str, mpn: str) -> CanonicalProductRecord:
    return CanonicalProductRecord(
        product_id=product_id,
        category="gpu",
        brand="Aster",
        model=model,
        canonical_name=f"Aster {model} 16GB",
        manufacturer_part_number=mpn,
        attributes={"vram_gb": 16, "length_mm": 300},
        is_synthetic=False,
    )


def _judgment(
    reviewer_id: str,
    label: str,
    *,
    assignment_suffix: str,
) -> dict[str, str]:
    return {
        "reviewer_id": reviewer_id,
        "assignment_id": f"assignment-{assignment_suffix}",
        "label": label,
        "reviewed_at": _CREATED_AT,
        "evidence_reference": f"evidence://pc-er/{assignment_suffix}",
    }


def _pair_label(product_id: str, label: str, suffix: str) -> dict[str, object]:
    return {
        "product_id": product_id,
        "judgments": [
            _judgment("reviewer-a", label, assignment_suffix=f"{suffix}-a"),
            _judgment("reviewer-b", label, assignment_suffix=f"{suffix}-b"),
        ],
        "adjudication": None,
        "resolved_label": label,
    }


def _label_payload(
    products: Sequence[CanonicalProductRecord],
    listing: ListingRecord,
    *,
    matching_product_id: str,
    policy: SourceUsePolicy,
    canonical_catalogue_file_sha256: str = "b" * 64,
) -> dict[str, object]:
    canonical_products = tuple(sorted(products, key=lambda product: product.product_id))
    content: dict[str, object] = {
        "schema_version": ER_LISTING_LABEL_SET_SCHEMA_VERSION,
        "dataset_version": policy.data_version,
        "territory": ER_LISTING_LABEL_TERRITORY,
        "domain": ER_LISTING_LABEL_DOMAIN,
        "label_source": ER_LISTING_LABEL_SOURCE,
        "review_protocol": ER_LISTING_REVIEW_PROTOCOL,
        "created_at": _CREATED_AT,
        "source_review_queue_sha256": _QUEUE_SHA,
        "canonical_catalogue_version": _CANONICAL_CATALOGUE_VERSION,
        "canonical_catalogue_sha256": canonical_catalogue_sha256(
            _CANONICAL_CATALOGUE_VERSION, canonical_products
        ),
        "canonical_catalogue_file_sha256": canonical_catalogue_file_sha256,
        "source_policy": policy.to_dict(),
        "products": [product.to_dict() for product in canonical_products],
        "listing_groups": [
            {
                "listing": listing.to_dict(),
                "match_disposition": "in_catalogue_match",
                "pair_labels": [
                    _pair_label(
                        product.product_id,
                        "MATCH"
                        if product.product_id == matching_product_id
                        else "NON_MATCH",
                        product.product_id,
                    )
                    for product in canonical_products
                ],
            }
        ],
    }
    return {**content, "dataset_sha256": sha256_json(content)}


def _write_labels(path: Path, payload: dict[str, object]) -> FrozenListingLabelSet:
    write_json(path, payload)
    return load_frozen_listing_label_set(path)


def _write_canonical_catalogue(
    path: Path, products: Sequence[CanonicalProductRecord]
) -> str:
    canonical_products = tuple(sorted(products, key=lambda product: product.product_id))
    write_json(
        path,
        {
            "schema_version": ER_CANONICAL_CATALOGUE_SCHEMA_VERSION,
            "catalogue_version": _CANONICAL_CATALOGUE_VERSION,
            "products": [product.to_dict() for product in canonical_products],
            "catalogue_sha256": canonical_catalogue_sha256(
                _CANONICAL_CATALOGUE_VERSION, canonical_products
            ),
        },
    )
    return sha256_file(path)


def _policy_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": ER_POLICY_SCHEMA_VERSION,
        "policy_id": "er-listing-evaluation-test-policy-v1",
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


class _FixedResolver:
    thresholds = MatchThresholds(auto_match=0.98, manual_review=0.80)

    def predict_proba(self, features: object) -> np.ndarray[Any, np.dtype[np.float64]]:
        row_count = len(cast(Sequence[object], features))
        if row_count != 2:
            raise AssertionError(f"expected two candidates, received {row_count}")
        return np.asarray([0.99, 0.98], dtype=np.float64)


def _runtime(
    labels: FrozenListingLabelSet,
    *,
    resolver: object,
) -> EntityResolutionRuntime:
    evidence = EntityResolutionServingEvidence(
        artifact_core_sha256="b" * 64,
        dataset_version=labels.dataset_version,
        label_source=ER_HUMAN_LABEL_SOURCE,
        claim_scope=ER_PRODUCTION_CLAIM_SCOPE,
        synthetic_rows=0,
        transfer_only=False,
        deployment_eligible=False,
        source_training_eligible=True,
        source_published_metrics_eligible=True,
        source_model_serving_eligible=True,
        listing_source=labels.source_policy.listing_source,
        catalogue_source=labels.source_policy.catalogue_source,
        source_scope_note=labels.source_policy.scope_note,
        review_queue_sha256=labels.source_review_queue_sha256,
        frozen_test_groups_sha256=frozen_listing_groups_sha256(labels),
        feature_contract_sha256="c" * 64,
        matcher_decision_version=ER_CATALOG_MATCHER_DECISION_VERSION,
        serving_projection_version=ER_SERVING_PROJECTION_VERSION,
        auto_match_threshold=0.98,
        manual_review_threshold=0.80,
    )
    return EntityResolutionRuntime(
        resolver=cast(LightGBMEntityResolver, resolver),
        evidence=evidence,
        artifact_path=Path("diagnostic-model"),
        release_sha256="d" * 64,
    )


def test_label_set_requires_two_independent_primary_reviews(tmp_path: Path) -> None:
    product = _product("gpu-a", "Nova A", "GPU-A")
    listing = ListingRecord(
        listing_id="listing-a",
        title="Aster Nova A 16GB",
        category="gpu",
        brand="Aster",
        manufacturer_part_number="GPU-A",
    )
    policy = SourceUsePolicy(
        listing_source="approved-retailer",
        catalogue_source="approved-manufacturer",
        data_version="pc-er-human-v1",
        training_eligible=True,
        published_metrics_eligible=True,
        model_serving_eligible=True,
        scope_note="independently reviewed test fixture",
    )
    payload = _label_payload(
        [product], listing, matching_product_id=product.product_id, policy=policy
    )
    group = cast(list[dict[str, Any]], payload["listing_groups"])[0]
    pair = cast(list[dict[str, Any]], group["pair_labels"])[0]
    pair["judgments"] = cast(list[object], pair["judgments"])[:1]
    content = dict(payload)
    content.pop("dataset_sha256")
    payload["dataset_sha256"] = sha256_json(content)
    write_json(tmp_path / "labels.json", payload)

    with pytest.raises(ListingLabelSetError, match="exactly two"):
        load_frozen_listing_label_set(tmp_path / "labels.json")


def test_disagreement_requires_independent_adjudication(tmp_path: Path) -> None:
    product = _product("gpu-a", "Nova A", "GPU-A")
    listing = ListingRecord(
        listing_id="listing-a",
        title="Aster Nova A 16GB",
        category="gpu",
        brand="Aster",
        manufacturer_part_number="GPU-A",
    )
    policy = SourceUsePolicy(
        listing_source="approved-retailer",
        catalogue_source="approved-manufacturer",
        data_version="pc-er-human-v1",
        training_eligible=True,
        published_metrics_eligible=True,
        model_serving_eligible=True,
        scope_note="independently reviewed test fixture",
    )
    payload = _label_payload(
        [product], listing, matching_product_id=product.product_id, policy=policy
    )
    group = cast(list[dict[str, Any]], payload["listing_groups"])[0]
    pair = cast(list[dict[str, Any]], group["pair_labels"])[0]
    judgments = cast(list[dict[str, str]], pair["judgments"])
    judgments[1]["label"] = "NON_MATCH"
    content = dict(payload)
    content.pop("dataset_sha256")
    payload["dataset_sha256"] = sha256_json(content)
    write_json(tmp_path / "labels.json", payload)

    with pytest.raises(ListingLabelSetError, match="require adjudication"):
        load_frozen_listing_label_set(tmp_path / "labels.json")


def test_listing_evaluator_measures_margin_deferral_and_blocks_small_fixture(
    tmp_path: Path,
) -> None:
    products = (
        _product("gpu-a", "Nova 16G Alpha", "GPU-A"),
        _product("gpu-b", "Nova 16G Beta", "GPU-B"),
    )
    listing = ListingRecord(
        listing_id="listing-margin",
        title="Aster Nova 16G graphics card",
        category="gpu",
        brand="Aster",
    )
    source_policy = SourceUsePolicy(
        listing_source="approved-retailer",
        catalogue_source="approved-manufacturer",
        data_version="pc-er-human-v1",
        training_eligible=True,
        published_metrics_eligible=True,
        model_serving_eligible=True,
        scope_note="independently reviewed test fixture",
    )
    labels = _write_labels(
        tmp_path / "labels.json",
        _label_payload(
            products,
            listing,
            matching_product_id=products[0].product_id,
            policy=source_policy,
        ),
    )
    policy = EntityResolutionPolicy.from_dict(_policy_payload())

    result = evaluate_listing_matcher(labels, _runtime(labels, resolver=_FixedResolver()), policy)

    assert result.candidate_blocking_recall == 1.0
    assert result.winner_selection_accuracy == 1.0
    assert result.ambiguity_case_count == 1
    assert result.ambiguity_deferred_count == 1
    assert result.ambiguity_false_auto_match_count == 0
    assert result.auto_match_count == 0
    assert any(
        "pair count=2 below production minimum=2500" in item
        for item in result.promotion_blockers
    )
    assert result.promotion_eligible is False


def test_cli_writes_immutable_diagnostic_without_inventing_rights(
    tmp_path: Path,
) -> None:
    product = _product("gpu-a", "Nova A", "GPU-A")
    listing = ListingRecord(
        listing_id="listing-a",
        title="Aster Nova A 16GB",
        category="gpu",
        brand="Aster",
        manufacturer_part_number="GPU-A",
    )
    source_policy = SourceUsePolicy(
        listing_source="approved-retailer",
        catalogue_source="approved-manufacturer",
        data_version="pc-er-human-v1",
        training_eligible=True,
        published_metrics_eligible=True,
        model_serving_eligible=True,
        scope_note="independently reviewed test fixture",
    )
    canonical_catalogue_path = tmp_path / "canonical-catalogue.json"
    canonical_catalogue_file_sha256 = _write_canonical_catalogue(
        canonical_catalogue_path, [product]
    )
    labels_path = tmp_path / "source-labels.json"
    labels = _write_labels(
        labels_path,
        _label_payload(
            [product],
            listing,
            matching_product_id=product.product_id,
            policy=source_policy,
            canonical_catalogue_file_sha256=canonical_catalogue_file_sha256,
        ),
    )
    model_dir = tmp_path / "source-model"
    resolver = LightGBMEntityResolver(device="cpu", random_state=23)
    resolver.fit(synthetic_pairs(seed=23, product_count=12), calibrate=True)
    resolver.thresholds = MatchThresholds(auto_match=0.98, manual_review=0.80)
    resolver.save_artifact(model_dir)
    write_json(
        model_dir / "serving_evidence.json",
        build_entity_resolution_serving_evidence(
            model_dir,
            dataset_version=labels.dataset_version,
            source_policy=source_policy.to_dict(),
            deployment_eligible=False,
            review_queue_sha256=labels.source_review_queue_sha256,
            frozen_test_groups_sha256=frozen_listing_groups_sha256(labels),
        ),
    )
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, _policy_payload())
    output = tmp_path / "release-candidate"

    assert (
        main(
            [
                "--model-artifact",
                str(model_dir),
                "--labels",
                str(labels_path),
                "--canonical-catalogue",
                str(canonical_catalogue_path),
                "--policy",
                str(policy_path),
                "--output-dir",
                str(output),
                "--evaluated-at",
                "2026-08-15T12:00:00+08:00",
            ]
        )
        == 0
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    evaluation = load_entity_resolution_evaluation(output / "evaluation.json")
    assert evaluation is not None
    assert manifest["release_class"] == "diagnostic"
    assert manifest["evaluation_summary"]["promotion_eligible"] is False
    assert evaluation.labelled_pair_count == 1
    assert evaluation.deployment_eligible is False
    assert evaluation.label_dataset_file_sha256 == sha256_file(output / "labels.json")
    assert evaluation.decision_rows_sha256 == sha256_file(output / "decisions.jsonl")
    assert not (output / "rights.json").exists()
    with pytest.raises(FileExistsError, match="not empty"):
        main(
            [
                "--model-artifact",
                str(model_dir),
                    "--labels",
                    str(labels_path),
                    "--canonical-catalogue",
                    str(canonical_catalogue_path),
                    "--policy",
                str(policy_path),
                "--output-dir",
                str(output),
            ]
        )

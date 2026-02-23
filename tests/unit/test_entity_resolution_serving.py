from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pc_build_recommender.catalog import (
    ER_EVALUATION_SCHEMA_VERSION_V2,
    CatalogEntityMatcher,
    EntityResolutionEvaluation,
    MappingOutcome,
)
from pc_build_recommender.entity_resolution import (
    CanonicalProductRecord,
    EntityResolutionArtifactError,
    EntityResolutionRuntime,
    EntityResolutionServingEvidence,
    LightGBMEntityResolver,
    ListingRow,
    MatchThresholds,
    build_entity_resolution_serving_evidence,
    entity_resolution_release_sha256,
    load_entity_resolution_runtime,
    synthetic_pairs,
)


class _FixedResolver:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.thresholds = MatchThresholds(auto_match=0.98, manual_review=0.80)

    def predict_proba(self, matrix: object) -> np.ndarray:
        rows = np.asarray(matrix).shape[0]
        return np.asarray(self.probabilities[:rows], dtype=np.float64)


@dataclass(frozen=True)
class _Runtime:
    resolver: _FixedResolver
    model_version: str = "er-lightgbm-fixture"
    production_authorized: bool = False


def _product(
    product_id: str,
    *,
    model: str,
    mpn: str,
    gtin: str | None = None,
    vram_gb: int = 16,
) -> CanonicalProductRecord:
    return CanonicalProductRecord(
        product_id=product_id,
        category="gpu",
        brand="ASUS",
        model=model,
        canonical_name=f"ASUS {model} {vram_gb}GB",
        manufacturer_part_number=mpn,
        gtin=gtin,
        attributes={"vram_gb": vram_gb, "colour": "black"},
    )


def _listing(
    title: str,
    *,
    mpn: str | None = None,
    gtin: str | None = None,
) -> ListingRow:
    return ListingRow(
        listing_id="listing-1",
        title=title,
        category="gpu",
        brand="ASUS",
        manufacturer_part_number=mpn,
        gtin=gtin,
        current_price_sgd=899,
    )


def _matcher(
    products: tuple[CanonicalProductRecord, ...],
    probabilities: list[float],
    *,
    production_authorized: bool,
) -> CatalogEntityMatcher:
    runtime = _Runtime(
        resolver=_FixedResolver(probabilities),
        production_authorized=production_authorized,
    )
    return CatalogEntityMatcher(
        products,
        runtime=cast(EntityResolutionRuntime, runtime),
    )


def test_exact_gtin_anchor_precedes_model_and_is_not_a_probability() -> None:
    product = _product("p1", model="Prime RTX 5060 Ti", mpn="GPU-ONE", gtin="12345678")
    matcher = _matcher((product,), [0.01], production_authorized=True)

    result = matcher.match(_listing("ASUS unrelated retailer wording", gtin="12345678"))

    assert result.outcome is MappingOutcome.AUTO_MATCHED
    assert result.matched_product_id == "p1"
    assert result.method == "exact_gtin"
    assert result.probability is None


def test_numeric_variant_conflict_rejects_exact_identifier_before_model() -> None:
    product = _product("p1", model="Prime RTX 5060 Ti", mpn="GPU-ONE", vram_gb=16)
    matcher = _matcher((product,), [1.0], production_authorized=True)

    result = matcher.match(_listing("ASUS Prime RTX 5060 Ti 8GB", mpn="GPU-ONE"))

    assert result.outcome is MappingOutcome.HARD_CONFLICT
    assert result.matched_product_id is None
    assert result.probability is None


def test_unpromoted_model_is_shadow_only_even_above_auto_threshold() -> None:
    product = _product("p1", model="Prime RTX 5060 Ti", mpn="GPU-ONE")
    matcher = _matcher((product,), [0.999], production_authorized=False)

    result = matcher.match(_listing("ASUS Prime RTX 5060 Ti graphics card"))

    assert result.outcome is MappingOutcome.MANUAL_REVIEW
    assert result.matched_product_id is None
    assert result.probability == pytest.approx(0.999)
    assert result.evidence["production_authorized"] is False


def test_authorized_model_uses_three_way_thresholds_and_winner_margin() -> None:
    products = (
        _product("p1", model="Prime RTX 5060 Ti OC", mpn="GPU-ONE"),
        _product("p2", model="Prime RTX 5060 Ti Pro", mpn="GPU-TWO"),
    )
    close = _matcher(products, [0.999, 0.995], production_authorized=True).match(
        _listing("ASUS Prime RTX 5060 Ti graphics card")
    )
    review = _matcher(products[:1], [0.90], production_authorized=True).match(
        _listing("ASUS Prime RTX 5060 Ti graphics card")
    )
    rejected = _matcher(products[:1], [0.40], production_authorized=True).match(
        _listing("ASUS Prime RTX 5060 Ti graphics card")
    )

    assert close.outcome is MappingOutcome.MANUAL_REVIEW
    assert review.outcome is MappingOutcome.MANUAL_REVIEW
    assert rejected.outcome is MappingOutcome.MODEL_REJECTED


def _write_shadow_artifact(path: Path, *, promoted: bool = False) -> Path:
    resolver = LightGBMEntityResolver(device="cpu", random_state=7)
    resolver.fit(synthetic_pairs(seed=7, product_count=12), calibrate=True)
    resolver.save_artifact(path)
    evidence = build_entity_resolution_serving_evidence(
        path,
        dataset_version="human-pc-fixture-v1",
        source_policy={
            "listing_source": "authorized-retailer-fixture",
            "catalogue_source": "authorized-catalogue-fixture",
            "data_version": "human-pc-fixture-v1",
            "training_eligible": True,
            "published_metrics_eligible": True,
            "model_serving_eligible": True,
            "scope_note": "unit-test contract fixture",
        },
        deployment_eligible=promoted,
        review_queue_sha256="a" * 64,
        frozen_test_groups_sha256="b" * 64,
        end_to_end_matcher_evaluated=promoted,
    )
    (path / "serving_evidence.json").write_text(
        json.dumps(evidence, sort_keys=True), encoding="utf-8"
    )
    return path


def test_unpromoted_human_artifact_requires_explicit_shadow_opt_in(tmp_path: Path) -> None:
    artifact = _write_shadow_artifact(tmp_path / "model")

    with pytest.raises(EntityResolutionArtifactError, match="not promoted"):
        load_entity_resolution_runtime(artifact)

    runtime = load_entity_resolution_runtime(
        artifact,
        allow_unpromoted_human_diagnostic=True,
    )
    assert runtime.production_authorized is False


def test_direct_evaluation_cannot_self_authorize_a_runtime(tmp_path: Path) -> None:
    artifact = _write_shadow_artifact(tmp_path / "model", promoted=True)
    runtime = load_entity_resolution_runtime(artifact)
    evaluation = EntityResolutionEvaluation(
        schema_version=ER_EVALUATION_SCHEMA_VERSION_V2,
        evaluation_id="catalog-matcher-eval-v1",
        dataset_version="human-pc-fixture-v1",
        model_version=runtime.model_version,
        label_source="human_reviewed",
        synthetic=False,
        precision=1.0,
        labelled_pair_count=50,
        evaluated_at=datetime.now(UTC),
        artifact_sha256=entity_resolution_release_sha256(artifact),
        review_queue_sha256="a" * 64,
        frozen_test_groups_sha256="b" * 64,
        auto_match_threshold=0.98,
        precision_numerator=50,
        precision_denominator=50,
        precision_ci_lower=0.95,
        precision_ci_upper=1.0,
        recall=0.96,
        f1=0.98,
        reportable=True,
        deployment_eligible=True,
    )

    with pytest.raises(EntityResolutionArtifactError, match="direct.*authorization is disabled"):
        runtime.authorize_for_production(
            evaluation,
            minimum_precision=0.95,
            minimum_labelled_pairs=50,
        )


def test_artifact_tamper_and_existing_non_serving_artifacts_fail_closed(tmp_path: Path) -> None:
    artifact = _write_shadow_artifact(tmp_path / "model")
    model_path = artifact / "model.txt"
    model_path.write_bytes(model_path.read_bytes() + b"\n# tampered")

    with pytest.raises(EntityResolutionArtifactError, match="does not match artifact bytes"):
        load_entity_resolution_runtime(
            artifact,
            allow_unpromoted_human_diagnostic=True,
        )

    repository_root = Path(__file__).resolve().parents[2]
    for current_artifact in (
        repository_root / "artifacts/models/entity-resolution-synthetic/lightgbm",
        repository_root / "artifacts/models/er-transfer-dn7/lightgbm",
    ):
        with pytest.raises((EntityResolutionArtifactError, FileNotFoundError)):
            load_entity_resolution_runtime(
                current_artifact,
                allow_unpromoted_human_diagnostic=True,
            )


def test_serving_evidence_type_is_not_constructed_from_untrusted_defaults() -> None:
    with pytest.raises(EntityResolutionArtifactError, match="eligibility fields"):
        EntityResolutionServingEvidence.from_dict(
            {
                "schema_version": "pc-build-recommender.er-serving-evidence.v1",
                "artifact_core_sha256": "a" * 64,
                "dataset_version": "dataset-v1",
                "label_source": "attributable_human_reviews",
                "claim_scope": "pc_retailer_catalog",
                "synthetic_rows": 0,
                "transfer_only": False,
                "deployment_eligible": False,
                "source_policy": {
                    "listing_source": "retailer",
                    "catalogue_source": "catalogue",
                    "data_version": "dataset-v1",
                    "training_eligible": True,
                    "published_metrics_eligible": True,
                    "model_serving_eligible": "yes",
                    "scope_note": "fixture",
                },
                "thresholds": {"auto_match": 0.98, "manual_review": 0.80},
                "review_queue_sha256": "b" * 64,
                "frozen_test_groups_sha256": "c" * 64,
                "feature_contract_sha256": "d" * 64,
                "matcher_decision_version": "catalog-er-decision-v1",
                "serving_projection_version": "governed-offer-er-projection-v1",
            }
        )

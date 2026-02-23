from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pc_build_recommender.entity_resolution import (
    FEATURE_NAMES,
    ExactMatchBaseline,
    LightGBMEntityResolver,
    LogisticMatchBaseline,
    MatchOutcome,
    PairFeatureExtractor,
    load_entity_resolver,
    synthetic_pairs,
)


def _dataset() -> tuple[object, ...]:
    return synthetic_pairs(
        seed=23,
        product_count=24,
        positive_variants=2,
        negatives_per_listing=2,
    )


def test_exact_baseline_conflict_overrides_exact_identifier() -> None:
    pairs = list(_dataset())
    hard_negative = next(
        pair
        for pair in pairs
        if not pair.label
        and PairFeatureExtractor().extract(pair.listing, pair.product).numeric_conflict
    )
    # Force an exact identifier while retaining the known capacity/VRAM/wattage mismatch.
    listing = hard_negative.listing.__class__(
        **{
            **hard_negative.listing.to_dict(),
            "manufacturer_part_number": hard_negative.product.manufacturer_part_number,
        }
    )
    conflict_pair = hard_negative.__class__(
        pair_id="forced-exact-conflict",
        listing=listing,
        product=hard_negative.product,
        label=0,
        is_synthetic=True,
    )
    model = ExactMatchBaseline().fit(pairs, calibrate=False)

    assert model.predict_proba([conflict_pair]).tolist() == [0.0]
    decision = model.predict_decisions([conflict_pair])[0]
    assert decision.outcome is MatchOutcome.REJECT
    assert decision.hard_conflict


def test_logistic_accepts_examples_and_feature_matrix_and_round_trips(tmp_path: Path) -> None:
    pairs = _dataset()
    extractor = PairFeatureExtractor()
    matrix = extractor.transform(pairs)
    labels = np.asarray([pair.label for pair in pairs])

    examples_model = LogisticMatchBaseline().fit(pairs, calibrate=True)
    matrix_model = LogisticMatchBaseline().fit(matrix, labels, calibrate=False)
    assert examples_model.predict_proba(pairs).shape == (len(pairs),)
    assert matrix_model.predict_proba(matrix).shape == (len(pairs),)

    artifact = examples_model.save_artifact(tmp_path / "logistic")
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert "coefficients" in metadata
    restored = load_entity_resolver(artifact)
    np.testing.assert_allclose(
        restored.predict_proba(pairs),
        examples_model.predict_proba(pairs),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.model
def test_lightgbm_cpu_artifact_round_trip(tmp_path: Path) -> None:
    pairs = _dataset()
    model = LightGBMEntityResolver(
        device="cpu",
        parameters={"n_estimators": 35, "min_child_samples": 2, "num_leaves": 7},
    ).fit(pairs, calibrate=True)
    before = model.predict_proba(pairs)
    artifact = model.save_artifact(tmp_path / "lightgbm")
    restored = LightGBMEntityResolver.load_artifact(artifact)

    assert model.actual_device == "cpu"
    assert (artifact / "model.txt").is_file()
    np.testing.assert_allclose(restored.predict_proba(pairs), before, rtol=0.0, atol=1e-12)


@pytest.mark.model
def test_lightgbm_auto_records_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = _dataset()
    original = LightGBMEntityResolver._fit_device

    def fail_gpu_then_fit_cpu(
        self: LightGBMEntityResolver,
        matrix: np.ndarray,
        labels: np.ndarray,
        device: str,
    ) -> object:
        if device == "gpu":
            raise RuntimeError("synthetic GPU learner unavailable")
        return original(self, matrix, labels, device)

    monkeypatch.setattr(LightGBMEntityResolver, "_fit_device", fail_gpu_then_fit_cpu)
    model = LightGBMEntityResolver(
        device="auto",
        parameters={"n_estimators": 15, "min_child_samples": 2},
    ).fit(pairs, calibrate=False)

    assert model.requested_device == "auto"
    assert model.actual_device == "cpu"
    assert model.fallback_reason is not None
    assert "synthetic GPU learner unavailable" in model.fallback_reason


def test_synthetic_model_evaluation_requires_explicit_non_promotable_opt_in() -> None:
    pairs = _dataset()
    model = ExactMatchBaseline().fit(pairs)

    with pytest.raises(ValueError, match="no evaluable non-synthetic rows"):
        model.evaluate(pairs)
    evaluation = model.evaluate(pairs, include_synthetic=True)
    assert evaluation.sample_count == len(pairs)
    assert not evaluation.is_promotable

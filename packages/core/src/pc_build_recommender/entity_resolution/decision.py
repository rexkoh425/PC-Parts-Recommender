"""Probability calibration and conservative entity-resolution decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]


class MatchOutcome(StrEnum):
    """Operational routing outcome for a listing/product pair."""

    AUTO_MATCH = "AUTO_MATCH"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class MatchDecision:
    probability: float
    outcome: MatchOutcome
    hard_conflict: bool = False
    reason: str | None = None

    @property
    def value(self) -> str:
        """Enum-like convenience for API callers."""

        return self.outcome.value

    def to_dict(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "outcome": self.outcome.value,
            "hard_conflict": self.hard_conflict,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MatchThresholds:
    """Versionable thresholds tuned for precision-first canonical mapping."""

    auto_match: float = 0.98
    manual_review: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.manual_review <= self.auto_match <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= manual_review <= auto_match <= 1"
            )

    def decide(self, probability: float, *, hard_conflict: bool = False) -> MatchDecision:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        if hard_conflict:
            return MatchDecision(
                probability=probability,
                outcome=MatchOutcome.REJECT,
                hard_conflict=True,
                reason="numeric_variant_conflict",
            )
        if probability >= self.auto_match:
            outcome = MatchOutcome.AUTO_MATCH
        elif probability >= self.manual_review:
            outcome = MatchOutcome.MANUAL_REVIEW
        else:
            outcome = MatchOutcome.REJECT
        return MatchDecision(probability=probability, outcome=outcome)

    def to_dict(self) -> dict[str, float]:
        return {"auto_match": self.auto_match, "manual_review": self.manual_review}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MatchThresholds:
        return cls(
            auto_match=float(data["auto_match"]),
            manual_review=float(data["manual_review"]),
        )


def _logit(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0)))


@dataclass(slots=True)
class PlattCalibrator:
    """One-dimensional logistic calibration with JSON-safe state."""

    coefficient: float | None = None
    intercept: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.coefficient is not None and self.intercept is not None

    def fit(
        self,
        probabilities: Sequence[float] | NDArray[np.float64],
        labels: Sequence[int] | NDArray[np.int_],
    ) -> PlattCalibrator:
        values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        targets = np.asarray(labels, dtype=np.int64).reshape(-1)
        if values.shape[0] != targets.shape[0] or values.size < 2:
            raise ValueError("probabilities and labels must have equal length of at least two")
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("calibration probabilities must be finite and in [0, 1]")
        if set(np.unique(targets)) != {0, 1}:
            raise ValueError("calibration requires both positive and negative labels")
        model = LogisticRegression(C=1e6, solver="lbfgs", random_state=0, max_iter=1000)
        model.fit(_logit(values).reshape(-1, 1), targets)
        self.coefficient = float(model.coef_[0, 0])
        self.intercept = float(model.intercept_[0])
        return self

    def predict_proba(
        self,
        probabilities: Sequence[float] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator must be fitted before prediction")
        values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("probabilities must be finite and in [0, 1]")
        assert self.coefficient is not None
        assert self.intercept is not None
        return _sigmoid(self.coefficient * _logit(values) + self.intercept)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "platt",
            "coefficient": self.coefficient,
            "intercept": self.intercept,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlattCalibrator:
        coefficient = data.get("coefficient")
        intercept = data.get("intercept")
        return cls(
            coefficient=float(coefficient) if coefficient is not None else None,
            intercept=float(intercept) if intercept is not None else None,
        )

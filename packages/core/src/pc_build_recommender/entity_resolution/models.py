"""Exact, logistic, and LightGBM entity-resolution models.

Artifacts are deliberately JSON plus LightGBM's text format.  No pickle payload is
required to serve a saved model, which makes the persisted contract inspectable and safer
to move between training and API environments.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self, cast

import lightgbm as lgb
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from .decision import MatchDecision, MatchThresholds, PlattCalibrator
from .features import (
    FEATURE_NAMES,
    NUMERIC_CONFLICT_FEATURE_INDEX,
    PairFeatureExtractor,
    validate_feature_matrix,
)
from .metrics import EntityResolutionEvaluation, evaluate_binary_predictions
from .records import PairExample

type FeatureInput = Sequence[PairExample] | ArrayLike
ARTIFACT_FORMAT_VERSION = 2


def _as_examples(values: FeatureInput) -> tuple[PairExample, ...] | None:
    if isinstance(values, np.ndarray):
        return None
    if hasattr(values, "to_numpy"):
        return None
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        return None
    if not items:
        return ()
    if isinstance(items[0], PairExample):
        if not all(isinstance(item, PairExample) for item in items):
            raise TypeError("feature input cannot mix PairExample and numeric rows")
        return items  # type: ignore[return-value]
    return None


def _prepare_features(
    values: FeatureInput,
    extractor: PairFeatureExtractor,
) -> tuple[NDArray[np.float64], tuple[PairExample, ...] | None, NDArray[np.bool_]]:
    examples = _as_examples(values)
    if examples is not None:
        matrix = extractor.transform(examples)
        conflicts = extractor.hard_conflict_mask(examples)
    else:
        matrix = validate_feature_matrix(values)
        conflicts = matrix[:, NUMERIC_CONFLICT_FEATURE_INDEX] >= 0.5
    return matrix, examples, conflicts


def _training_labels(
    examples: tuple[PairExample, ...] | None,
    labels: Sequence[int] | ArrayLike | None,
    row_count: int,
) -> NDArray[np.int64]:
    if labels is None:
        if examples is None:
            raise ValueError("y is required when fitting from a feature matrix")
        result = np.asarray([example.label for example in examples], dtype=np.int64)
    else:
        result = np.asarray(labels, dtype=np.int64).reshape(-1)
    if result.shape[0] != row_count:
        raise ValueError("feature rows and labels must have equal length")
    if not set(np.unique(result)).issubset({0, 1}):
        raise ValueError("entity-resolution labels must be binary")
    if row_count == 0:
        raise ValueError("at least one training example is required")
    return result


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0)))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing entity-resolution metadata: {metadata_path}")
    payload: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("entity-resolution metadata must be a JSON object")
    if payload.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported entity-resolution artifact format")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("artifact feature contract does not match this runtime")
    return cast(dict[str, Any], payload)


class BaseEntityResolver(ABC):
    """Shared serving, calibration, evaluation, and artifact behaviour."""

    model_type: str

    def __init__(
        self,
        *,
        thresholds: MatchThresholds | None = None,
        feature_extractor: PairFeatureExtractor | None = None,
    ) -> None:
        self.thresholds = thresholds or MatchThresholds()
        self.feature_extractor = feature_extractor or PairFeatureExtractor()
        self.calibrator: PlattCalibrator | None = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: FeatureInput,
        y: Sequence[int] | ArrayLike | None = None,
        *,
        calibrate: bool = True,
    ) -> Self:
        raise NotImplementedError

    @abstractmethod
    def _predict_uncalibrated(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def fit_calibrator(
        self,
        X: FeatureInput,
        y: Sequence[int] | ArrayLike | None = None,
    ) -> Self:
        """Fit calibration on a held-out set after the underlying estimator is trained."""

        matrix, examples, _ = _prepare_features(X, self.feature_extractor)
        labels = _training_labels(examples, y, matrix.shape[0])
        raw = self._predict_uncalibrated(matrix)
        self.calibrator = PlattCalibrator().fit(raw, labels)
        return self

    def predict_proba(self, X: FeatureInput) -> NDArray[np.float64]:
        """Return calibrated positive-class probabilities after hard conflict gates."""

        matrix, _, conflicts = _prepare_features(X, self.feature_extractor)
        raw = np.asarray(self._predict_uncalibrated(matrix), dtype=np.float64).reshape(-1)
        if raw.shape[0] != matrix.shape[0]:
            raise RuntimeError("model returned an invalid number of probabilities")
        probabilities = (
            self.calibrator.predict_proba(raw)
            if self.calibrator is not None and self.calibrator.is_fitted
            else raw
        )
        # This is intentionally after calibration: no calibrator can override a known variant.
        return np.where(conflicts, 0.0, np.clip(probabilities, 0.0, 1.0))

    def predict(self, X: FeatureInput, *, threshold: float = 0.5) -> NDArray[np.int64]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def predict_decisions(
        self,
        X: FeatureInput,
        *,
        thresholds: MatchThresholds | None = None,
    ) -> tuple[MatchDecision, ...]:
        matrix, _, conflicts = _prepare_features(X, self.feature_extractor)
        probabilities = self.predict_proba(matrix)
        policy = thresholds or self.thresholds
        return tuple(
            policy.decide(float(probability), hard_conflict=bool(conflict))
            for probability, conflict in zip(probabilities, conflicts, strict=True)
        )

    def evaluate(
        self,
        X: FeatureInput,
        y: Sequence[int] | ArrayLike | None = None,
        *,
        include_synthetic: bool = False,
        classification_threshold: float = 0.5,
    ) -> EntityResolutionEvaluation:
        matrix, examples, conflicts = _prepare_features(X, self.feature_extractor)
        labels = _training_labels(examples, y, matrix.shape[0])
        synthetic = (
            [example.is_synthetic for example in examples]
            if examples is not None
            else [False] * matrix.shape[0]
        )
        return evaluate_binary_predictions(
            labels,
            self.predict_proba(matrix),
            hard_conflicts=conflicts,
            is_synthetic=synthetic,
            include_synthetic=include_synthetic,
            classification_threshold=classification_threshold,
            thresholds=self.thresholds,
        )

    def _common_metadata(self) -> dict[str, Any]:
        return {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "model_type": self.model_type,
            "feature_names": list(FEATURE_NAMES),
            "thresholds": self.thresholds.to_dict(),
            "calibrator": self.calibrator.to_dict() if self.calibrator is not None else None,
            "is_fitted": self.is_fitted,
        }

    @abstractmethod
    def save_artifact(self, path: str | Path) -> Path:
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        return self.save_artifact(path)

    @classmethod
    @abstractmethod
    def load_artifact(cls, path: str | Path) -> Self:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls.load_artifact(path)


def _restore_common(model: BaseEntityResolver, metadata: Mapping[str, Any]) -> None:
    model.thresholds = MatchThresholds.from_dict(metadata["thresholds"])
    calibration = metadata.get("calibrator")
    model.calibrator = PlattCalibrator.from_dict(calibration) if calibration else None
    model.is_fitted = bool(metadata.get("is_fitted", True))


class ExactMatchBaseline(BaseEntityResolver):
    """High-precision baseline using identifiers and exact normalised titles."""

    model_type = "exact_match_baseline"

    def fit(
        self,
        X: FeatureInput,
        y: Sequence[int] | ArrayLike | None = None,
        *,
        calibrate: bool = False,
    ) -> Self:
        matrix, examples, _ = _prepare_features(X, self.feature_extractor)
        labels = _training_labels(examples, y, matrix.shape[0])
        self.is_fitted = True
        if calibrate:
            self.calibrator = PlattCalibrator().fit(self._predict_uncalibrated(matrix), labels)
        return self

    def _predict_uncalibrated(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        exact_identifier = (matrix[:, 0] >= 0.5) | (matrix[:, 1] >= 0.5)
        exact_text = (
            (matrix[:, 2] >= 0.5)
            & (matrix[:, 3] >= 0.5)
            & (matrix[:, 4] >= 0.999)
        )
        return np.where(exact_identifier, 0.995, np.where(exact_text, 0.985, 0.01))

    def save_artifact(self, path: str | Path) -> Path:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "metadata.json", self._common_metadata())
        return target

    @classmethod
    def load_artifact(cls, path: str | Path) -> ExactMatchBaseline:
        metadata = _read_metadata(Path(path))
        if metadata["model_type"] != cls.model_type:
            raise ValueError(
                f"artifact contains {metadata['model_type']!r}, not {cls.model_type!r}"
            )
        result = cls()
        _restore_common(result, metadata)
        return result


class LogisticMatchBaseline(BaseEntityResolver):
    """Deterministic interpretable baseline for learned pair matching."""

    model_type = "logistic_match_baseline"

    def __init__(
        self,
        *,
        regularization_c: float = 1.0,
        class_weight: str | Mapping[int, float] | None = "balanced",
        max_iter: int = 2000,
        thresholds: MatchThresholds | None = None,
        feature_extractor: PairFeatureExtractor | None = None,
    ) -> None:
        super().__init__(thresholds=thresholds, feature_extractor=feature_extractor)
        if regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        self.regularization_c = regularization_c
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.coefficients: NDArray[np.float64] | None = None
        self.intercept: float | None = None

    def fit(
        self,
        X: FeatureInput,
        y: Sequence[int] | ArrayLike | None = None,
        *,
        calibrate: bool = True,
    ) -> Self:
        matrix, examples, _ = _prepare_features(X, self.feature_extractor)
        labels = _training_labels(examples, y, matrix.shape[0])
        if len(np.unique(labels)) != 2:
            raise ValueError("logistic training requires both positive and negative examples")
        estimator = LogisticRegression(
            C=self.regularization_c,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=0,
            solver="liblinear",
        )
        estimator.fit(matrix, labels)
        self.coefficients = np.asarray(estimator.coef_[0], dtype=np.float64)
        self.intercept = float(estimator.intercept_[0])
        self.is_fitted = True
        if calibrate:
            self.calibrator = PlattCalibrator().fit(self._predict_uncalibrated(matrix), labels)
        return self

    def _predict_uncalibrated(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coefficients is None or self.intercept is None:
            raise RuntimeError("LogisticMatchBaseline must be fitted before prediction")
        return _sigmoid(matrix @ self.coefficients + self.intercept)

    def save_artifact(self, path: str | Path) -> Path:
        if self.coefficients is None or self.intercept is None:
            raise RuntimeError("cannot save an unfitted logistic model")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        metadata = self._common_metadata()
        metadata.update(
            {
                "regularization_c": self.regularization_c,
                "class_weight": self.class_weight,
                "max_iter": self.max_iter,
                "coefficients": self.coefficients.tolist(),
                "intercept": self.intercept,
            }
        )
        _write_json(target / "metadata.json", metadata)
        return target

    @classmethod
    def load_artifact(cls, path: str | Path) -> LogisticMatchBaseline:
        metadata = _read_metadata(Path(path))
        if metadata["model_type"] != cls.model_type:
            raise ValueError(
                f"artifact contains {metadata['model_type']!r}, not {cls.model_type!r}"
            )
        result = cls(
            regularization_c=float(metadata["regularization_c"]),
            class_weight=metadata.get("class_weight"),
            max_iter=int(metadata["max_iter"]),
        )
        coefficients = np.asarray(metadata["coefficients"], dtype=np.float64)
        if coefficients.shape != (len(FEATURE_NAMES),):
            raise ValueError("artifact contains an invalid logistic coefficient vector")
        result.coefficients = coefficients
        result.intercept = float(metadata["intercept"])
        _restore_common(result, metadata)
        return result


class LightGBMEntityResolver(BaseEntityResolver):
    """Non-linear duplicate classifier with deterministic CPU fallback."""

    model_type = "lightgbm_entity_resolver"

    def __init__(
        self,
        *,
        device: str = "auto",
        random_state: int = 42,
        parameters: Mapping[str, Any] | None = None,
        thresholds: MatchThresholds | None = None,
        feature_extractor: PairFeatureExtractor | None = None,
    ) -> None:
        super().__init__(thresholds=thresholds, feature_extractor=feature_extractor)
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be one of: auto, cpu, gpu")
        self.requested_device = device
        self.actual_device: str | None = None
        self.fallback_reason: str | None = None
        self.random_state = random_state
        self.parameters = dict(parameters or {})
        self.booster: lgb.Booster | None = None

    def _parameters_for(self, device: str) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "objective": "binary",
            "n_estimators": 250,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 8,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.05,
            "reg_lambda": 0.2,
            "random_state": self.random_state,
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "device_type": device,
        }
        defaults.update(self.parameters)
        defaults["device_type"] = device
        return defaults

    def _fit_device(
        self,
        matrix: NDArray[np.float64],
        labels: NDArray[np.int64],
        device: str,
    ) -> lgb.Booster:
        estimator = lgb.LGBMClassifier(**self._parameters_for(device))
        estimator.fit(matrix, labels, feature_name=list(FEATURE_NAMES))
        return estimator.booster_

    def fit(
        self,
        X: FeatureInput,
        y: Sequence[int] | ArrayLike | None = None,
        *,
        calibrate: bool = True,
    ) -> Self:
        matrix, examples, _ = _prepare_features(X, self.feature_extractor)
        labels = _training_labels(examples, y, matrix.shape[0])
        if len(np.unique(labels)) != 2:
            raise ValueError("LightGBM training requires both positive and negative examples")

        first_device = "gpu" if self.requested_device in {"auto", "gpu"} else "cpu"
        try:
            self.booster = self._fit_device(matrix, labels, first_device)
            self.actual_device = first_device
            self.fallback_reason = None
        except Exception as error:
            if self.requested_device != "auto":
                raise
            self.fallback_reason = f"{type(error).__name__}: {error}"[:1000]
            self.booster = self._fit_device(matrix, labels, "cpu")
            self.actual_device = "cpu"

        self.is_fitted = True
        if calibrate:
            self.calibrator = PlattCalibrator().fit(self._predict_uncalibrated(matrix), labels)
        return self

    def _predict_uncalibrated(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.booster is None:
            raise RuntimeError("LightGBMEntityResolver must be fitted before prediction")
        result = np.asarray(self.booster.predict(matrix), dtype=np.float64)
        if result.ndim == 2:
            result = result[:, -1]
        return result.reshape(-1)

    def save_artifact(self, path: str | Path) -> Path:
        if self.booster is None:
            raise RuntimeError("cannot save an unfitted LightGBM model")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / "model.txt"
        self.booster.save_model(str(model_path))
        metadata = self._common_metadata()
        metadata.update(
            {
                "requested_device": self.requested_device,
                "actual_device": self.actual_device,
                "fallback_reason": self.fallback_reason,
                "random_state": self.random_state,
                "parameters": self.parameters,
                "model_file": model_path.name,
            }
        )
        _write_json(target / "metadata.json", metadata)
        return target

    @classmethod
    def load_artifact(cls, path: str | Path) -> LightGBMEntityResolver:
        target = Path(path)
        metadata = _read_metadata(target)
        if metadata["model_type"] != cls.model_type:
            raise ValueError(
                f"artifact contains {metadata['model_type']!r}, not {cls.model_type!r}"
            )
        result = cls(
            device=str(metadata["requested_device"]),
            random_state=int(metadata["random_state"]),
            parameters=metadata.get("parameters", {}),
        )
        model_file = target / str(metadata.get("model_file", "model.txt"))
        if not model_file.is_file():
            raise FileNotFoundError(f"missing LightGBM model file: {model_file}")
        result.booster = lgb.Booster(model_file=str(model_file))
        result.actual_device = metadata.get("actual_device")
        result.fallback_reason = metadata.get("fallback_reason")
        _restore_common(result, metadata)
        return result


def load_entity_resolver(path: str | Path) -> BaseEntityResolver:
    """Load any supported entity resolver from its inspectable artifact directory."""

    target = Path(path)
    metadata = _read_metadata(target)
    model_type = metadata.get("model_type")
    model_classes: dict[str, type[BaseEntityResolver]] = {
        ExactMatchBaseline.model_type: ExactMatchBaseline,
        LogisticMatchBaseline.model_type: LogisticMatchBaseline,
        LightGBMEntityResolver.model_type: LightGBMEntityResolver,
    }
    try:
        model_class = model_classes[str(model_type)]
    except KeyError as error:
        raise ValueError(f"unsupported entity-resolution model type: {model_type!r}") from error
    return model_class.load_artifact(target)

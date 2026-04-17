"""LightGBM LambdaMART training and serving with query-group integrity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from inspect import signature
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from pc_build_recommender.evaluation.manifest import sha256_file, json_sha256
from pc_build_recommender.retrieval.benchmark import QueryGroupSplit
from pc_build_recommender.retrieval.evaluation import RelevanceLabelSource

from .features import FeatureBatch, RankingFeatureBuilder
from .models import (
    LabeledRankingQuery,
    RankedCandidate,
    RankerArtifactIdentity,
    RankerMetadata,
    RankingCandidate,
    RankingContext,
    RankingQuery,
)
from .publication import (
    RankerStageActivityLock,
    _fsync_directory,
    _is_linklike,
    _rename_directory_noreplace,
    acquire_ranker_stage_activity_lock,
)

DEFAULT_LAMBDAMART_PARAMETERS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 0.5,
    "random_state": 20260722,
    "n_jobs": -1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}

RANKER_ARTIFACT_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.lambdamart-artifact-manifest.v1"
RANKER_BUNDLE_ARTIFACT_MANIFEST_SCHEMA_VERSION = (
    "pc-build-recommender.lambdamart-artifact-manifest.v2"
)
MAX_RANKER_JSON_BYTES = 1024 * 1024
MAX_RANKER_MODEL_BYTES = 128 * 1024 * 1024


def ranker_artifact_manifest_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_suffix(path.suffix + ".artifact-manifest.json")


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".metadata.json")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not permitted: {value}")


def _decode_json_object(payload_bytes: bytes, *, label: str) -> dict[str, Any]:
    payload = json.loads(
        payload_bytes.decode("utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _read_bounded_bytes(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    with path.open("rb") as handle:
        payload = handle.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"{label} exceeds the {maximum_bytes}-byte safety limit")
    return payload


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json_temp(path: Path, payload: Mapping[str, object]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _manifest_content(
    *,
    ranker_version: str,
    model_path: Path,
    metadata_path: Path,
    publication_intent_sha256: str | None = None,
) -> dict[str, object]:
    content: dict[str, object] = {
        "schema_version": (
            RANKER_BUNDLE_ARTIFACT_MANIFEST_SCHEMA_VERSION
            if publication_intent_sha256 is not None
            else RANKER_ARTIFACT_MANIFEST_SCHEMA_VERSION
        ),
        "ranker_version": ranker_version,
        "files": {
            model_path.name: {
                "size_bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
            },
            metadata_path.name: {
                "size_bytes": metadata_path.stat().st_size,
                "sha256": sha256_file(metadata_path),
            },
        },
    }
    if publication_intent_sha256 is not None:
        if not _is_sha256(publication_intent_sha256):
            raise ValueError("publication_intent_sha256 must be a lowercase SHA-256 digest")
        content["publication_intent_sha256"] = publication_intent_sha256
    return content


@dataclass(frozen=True, slots=True)
class PreparedRankingData:
    """Contiguous feature rows and LightGBM group sizes."""

    features: NDArray[np.float64]
    labels: NDArray[np.int32]
    group_sizes: tuple[int, ...]
    row_keys: tuple[tuple[str, str], ...]


def prepare_lgbm_data(
    queries: Sequence[LabeledRankingQuery],
    feature_builder: RankingFeatureBuilder,
) -> PreparedRankingData:
    """Materialise groups contiguously and reject duplicate query IDs."""

    if not queries:
        raise ValueError("at least one labeled query is required")
    query_ids = [query.context.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique in a ranking dataset")

    matrices: list[NDArray[np.float64]] = []
    labels: list[int] = []
    groups: list[int] = []
    row_keys: list[tuple[str, str]] = []
    for query in queries:
        batch = feature_builder.build(query.context, query.candidates)
        matrices.append(batch.values)
        labels.extend(query.relevance_grades)
        groups.append(len(query.candidates))
        row_keys.extend(
            (query.context.query_id, candidate.product_id) for candidate in query.candidates
        )
    return PreparedRankingData(
        features=np.vstack(matrices),
        labels=np.asarray(labels, dtype=np.int32),
        group_sizes=tuple(groups),
        row_keys=tuple(row_keys),
    )


class LambdaMARTRanker:
    """Train, persist, load, and serve an ``LGBMRanker`` safely.

    Pass ``parameters={"device_type": "gpu"}`` to use a GPU-enabled LightGBM
    build.  CPU remains the portable default because Windows wheels do not all
    include the OpenCL/CUDA learner.
    """

    def __init__(
        self,
        *,
        feature_builder: RankingFeatureBuilder | None = None,
        parameters: Mapping[str, Any] | None = None,
        ranker_version: str = "ltr-v1",
    ) -> None:
        self.feature_builder = feature_builder or RankingFeatureBuilder()
        self.parameters = dict(DEFAULT_LAMBDAMART_PARAMETERS)
        if parameters:
            self.parameters.update(parameters)
        if self.parameters.get("objective") not in {"lambdarank", "rank_xendcg"}:
            raise ValueError("ranking objective must be lambdarank or rank_xendcg")
        self.ranker_version = ranker_version
        self._model: lgb.LGBMRanker | None = None
        self._booster: lgb.Booster | None = None
        self._metadata: RankerMetadata | None = None
        self._artifact_identity: RankerArtifactIdentity | None = None
        self._publication_intent_sha256: str | None = None
        self._verified_artifact_loaded = False

    @property
    def metadata(self) -> RankerMetadata:
        if self._metadata is None:
            raise RuntimeError("ranker has not been trained or loaded")
        return self._metadata

    @property
    def artifact_identity(self) -> RankerArtifactIdentity:
        if self._artifact_identity is None:
            raise RuntimeError("ranker has not been persisted as a verified artifact")
        return self._artifact_identity

    @property
    def verified_artifact_loaded(self) -> bool:
        return self._verified_artifact_loaded

    @property
    def publication_intent_sha256(self) -> str | None:
        """Stable publication identity for directory-bundled artifacts."""

        return self._publication_intent_sha256

    def fit(
        self,
        training_queries: Sequence[LabeledRankingQuery],
        *,
        validation_queries: Sequence[LabeledRankingQuery] | None = None,
        training_data_version: str,
        candidate_set_version: str | None = None,
        early_stopping_rounds: int = 30,
        training_label_source: RelevanceLabelSource | str = RelevanceLabelSource.UNVERIFIED,
        training_adjudication_complete: bool = False,
        contains_synthetic_labels: bool = False,
        training_judgment_manifest_sha256: str | None = None,
        training_dataset_manifest_sha256: str | None = None,
        training_prelabel_snapshot_sha256: str | None = None,
        training_feature_contract_sha256: str | None = None,
        query_group_ids: Mapping[str, str] | None = None,
        query_group_split_checksum: str | None = None,
        frozen_query_split: QueryGroupSplit | None = None,
    ) -> LambdaMARTRanker:
        """Fit with query-contiguous rows and optional query-held-out validation."""

        self._artifact_identity = None
        self._publication_intent_sha256 = None
        self._verified_artifact_loaded = False
        if not training_data_version:
            raise ValueError("training_data_version is required for model provenance")
        label_source = RelevanceLabelSource(training_label_source)
        for name, digest in (
            ("training_judgment_manifest_sha256", training_judgment_manifest_sha256),
            ("training_dataset_manifest_sha256", training_dataset_manifest_sha256),
            ("training_prelabel_snapshot_sha256", training_prelabel_snapshot_sha256),
            ("training_feature_contract_sha256", training_feature_contract_sha256),
            ("query_group_split_checksum", query_group_split_checksum),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        all_query_ids = {query.context.query_id for query in training_queries}
        all_query_ids.update(query.context.query_id for query in validation_queries or ())
        split_membership_verified = False
        if frozen_query_split is not None:
            if query_group_ids is not None or query_group_split_checksum is not None:
                raise ValueError(
                    "frozen_query_split cannot be combined with manual group IDs or checksum"
                )
            required_split_names = {"train", "validation", "test"}
            if not required_split_names.issubset(frozen_query_split.weights):
                raise ValueError("frozen query split must define train, validation, and test")
            split_label_source = RelevanceLabelSource(frozen_query_split.label_source)
            if (
                label_source is not RelevanceLabelSource.UNVERIFIED
                and label_source is not split_label_source
            ):
                raise ValueError("training label source conflicts with the frozen query split")
            if (
                training_judgment_manifest_sha256 is not None
                and training_judgment_manifest_sha256 != frozen_query_split.judgment_manifest_sha256
            ):
                raise ValueError("training judgment manifest conflicts with the frozen query split")
            if training_adjudication_complete and not frozen_query_split.adjudication_complete:
                raise ValueError(
                    "training adjudication claim conflicts with the frozen query split"
                )
            label_source = split_label_source
            training_adjudication_complete = frozen_query_split.adjudication_complete
            contains_synthetic_labels = (
                contains_synthetic_labels or frozen_query_split.contains_synthetic_labels
            )
            training_judgment_manifest_sha256 = frozen_query_split.judgment_manifest_sha256
            training_query_ids = {query.context.query_id for query in training_queries}
            validation_query_ids = {query.context.query_id for query in validation_queries or ()}
            expected_training_ids = {
                query_id
                for query_id, split_name in frozen_query_split.assignments.items()
                if split_name == "train"
            }
            expected_validation_ids = {
                query_id
                for query_id, split_name in frozen_query_split.assignments.items()
                if split_name == "validation"
            }
            if training_query_ids != expected_training_ids:
                raise ValueError("training query IDs must exactly match the frozen train split")
            if validation_query_ids != expected_validation_ids:
                raise ValueError(
                    "validation query IDs must exactly match the frozen validation split"
                )
            query_group_ids = {
                query_id: frozen_query_split.query_group_ids[query_id] for query_id in all_query_ids
            }
            query_group_split_checksum = frozen_query_split.checksum
            split_membership_verified = True
        if query_group_ids is not None:
            if set(query_group_ids) != all_query_ids:
                raise ValueError(
                    "query_group_ids must cover exactly the training and validation queries"
                )
            if any(not group_id for group_id in query_group_ids.values()):
                raise ValueError("query group IDs must not be empty")
        elif query_group_split_checksum is not None:
            raise ValueError("a query-group split checksum requires query_group_ids")
        if validation_queries:
            training_query_ids = {query.context.query_id for query in training_queries}
            validation_query_ids = {query.context.query_id for query in validation_queries}
            overlap = training_query_ids.intersection(validation_query_ids)
            if overlap:
                raise ValueError(
                    f"training and validation query IDs must be disjoint; overlap={sorted(overlap)}"
                )
            if query_group_ids is not None:
                training_groups = {query_group_ids[query_id] for query_id in training_query_ids}
                validation_groups = {query_group_ids[query_id] for query_id in validation_query_ids}
                group_overlap = training_groups.intersection(validation_groups)
                if group_overlap:
                    raise ValueError(
                        "training and validation query groups must be disjoint; "
                        f"overlap={sorted(group_overlap)}"
                    )
            if early_stopping_rounds < 1:
                raise ValueError("early_stopping_rounds must be positive")
        training = prepare_lgbm_data(training_queries, self.feature_builder)
        if len(set(training.labels.tolist())) < 2:
            raise ValueError("training labels must contain at least two relevance grades")
        model = lgb.LGBMRanker(**self.parameters)

        fit_kwargs: dict[str, Any] = {
            "X": training.features,
            "y": training.labels,
            "group": list(training.group_sizes),
            "feature_name": list(self.feature_builder.feature_names),
            "eval_at": [10],
            "callbacks": [lgb.log_evaluation(period=0)],
        }
        if validation_queries:
            validation = prepare_lgbm_data(validation_queries, self.feature_builder)
            fit_kwargs.update(
                {
                    "eval_group": [list(validation.group_sizes)],
                    "callbacks": [
                        lgb.log_evaluation(period=0),
                        lgb.early_stopping(early_stopping_rounds, verbose=False),
                    ],
                }
            )
            # LightGBM 4.7 introduced non-deprecated eval_X/eval_y keyword-only
            # arguments.  Keep 4.5/4.6 compatibility for the declared range.
            if "eval_X" in signature(model.fit).parameters:
                fit_kwargs["eval_X"] = validation.features
                fit_kwargs["eval_y"] = validation.labels
            else:
                fit_kwargs["eval_set"] = [(validation.features, validation.labels)]
        model.fit(**fit_kwargs)
        self._model = model
        self._booster = model.booster_

        metrics: dict[str, float] = {}
        if validation_queries:
            for dataset_metrics in model.evals_result_.values():
                for metric_name, values in dataset_metrics.items():
                    if values:
                        metrics[f"validation_{metric_name}"] = float(values[-1])
            if model.best_iteration_:
                metrics["best_iteration"] = float(model.best_iteration_)

        synthetic_present = (
            contains_synthetic_labels or label_source is RelevanceLabelSource.SYNTHETIC
        )
        promotion_block_reasons: list[str] = []
        if label_source is not RelevanceLabelSource.HUMAN:
            promotion_block_reasons.append(
                f"training label source is {label_source.value}, not human"
            )
        if not training_adjudication_complete:
            promotion_block_reasons.append("training labels are not fully adjudicated")
        if synthetic_present:
            promotion_block_reasons.append("synthetic labels are present")
        if training_judgment_manifest_sha256 is None:
            promotion_block_reasons.append("training judgment manifest hash is missing")
        if training_dataset_manifest_sha256 is None:
            promotion_block_reasons.append("training dataset manifest hash is missing")
        if training_prelabel_snapshot_sha256 is None:
            promotion_block_reasons.append("pre-label feature snapshot hash is missing")
        if training_feature_contract_sha256 is None:
            promotion_block_reasons.append("ranking feature contract hash is missing")
        if query_group_split_checksum is None:
            promotion_block_reasons.append("frozen query-group split hash is missing")
        if not split_membership_verified:
            promotion_block_reasons.append("training/validation split membership was not verified")

        self._metadata = RankerMetadata(
            ranker_version=self.ranker_version,
            ranking_basis="lightgbm_lambdamart",
            feature_version=self.feature_builder.feature_version,
            model_type="LGBMRanker",
            feature_names=self.feature_builder.feature_names,
            created_at_utc=datetime.now(UTC).isoformat(),
            training_data_version=training_data_version,
            candidate_set_version=candidate_set_version,
            training_query_count=len(training.group_sizes),
            training_row_count=len(training.labels),
            parameters=self.parameters,
            metrics=metrics,
            training_label_source=label_source.value,
            training_adjudication_complete=training_adjudication_complete,
            contains_synthetic_labels=synthetic_present,
            training_judgment_manifest_sha256=training_judgment_manifest_sha256,
            training_dataset_manifest_sha256=training_dataset_manifest_sha256,
            training_prelabel_snapshot_sha256=training_prelabel_snapshot_sha256,
            training_feature_contract_sha256=training_feature_contract_sha256,
            query_group_split_checksum=query_group_split_checksum,
            query_split_membership_verified=split_membership_verified,
            promotion_eligible=not promotion_block_reasons,
            promotion_block_reasons=tuple(promotion_block_reasons),
        )
        return self

    def predict(
        self, context: RankingContext, candidates: Sequence[RankingCandidate]
    ) -> NDArray[np.float64]:
        """Predict scores for one already-filtered query group."""

        if self._booster is None:
            raise RuntimeError("ranker has not been trained or loaded")
        if not candidates:
            return np.empty(0, dtype=np.float64)
        batch = self.feature_builder.build(context, candidates)
        return self.predict_feature_batch(batch)

    def predict_feature_batch(self, batch: FeatureBatch) -> NDArray[np.float64]:
        """Score one already-built feature snapshot without rebuilding its inputs."""

        if self._booster is None:
            raise RuntimeError("ranker has not been trained or loaded")
        if self.feature_builder.feature_version != self.metadata.feature_version:
            raise ValueError("feature batch version does not match ranker metadata")
        if batch.feature_names != self.metadata.feature_names:
            raise ValueError("feature batch order does not match ranker metadata")
        scores = np.asarray(self._booster.predict(batch.values), dtype=np.float64)
        if scores.shape != (batch.values.shape[0],) or not np.isfinite(scores).all():
            raise RuntimeError("LightGBM returned invalid ranking scores")
        return scores

    def rank_query(
        self, context: RankingContext, candidates: Sequence[RankingCandidate]
    ) -> list[RankedCandidate]:
        scores = self.predict(context, candidates)
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates[index].product_id),
        )
        return [
            RankedCandidate(
                candidate=candidates[index],
                score=float(scores[index]),
                rank=rank,
                ranker_version=self.metadata.ranker_version,
                ranking_basis=self.metadata.ranking_basis,
            )
            for rank, index in enumerate(order, start=1)
        ]

    def rank_queries(self, queries: Sequence[RankingQuery]) -> dict[str, list[RankedCandidate]]:
        """Score multiple groups without losing query boundaries."""

        query_ids = [query.context.query_id for query in queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query IDs must be unique")
        return {
            query.context.query_id: self.rank_query(query.context, query.candidates)
            for query in queries
        }

    def save(
        self,
        path: str | Path,
        *,
        _bundle_publication_intent_sha256: str | None = None,
    ) -> tuple[Path, Path]:
        """Publish a legacy content-bound native booster and metadata pair.

        The self-hashed manifest is replaced last and is the commit marker.  An
        interrupted publication can leave files behind, but ``load`` will reject
        them unless both exact files match that final manifest. Production writers
        should use :meth:`publish_bundle`, which adds directory-level crash recovery.
        """

        if self._booster is None or self._metadata is None:
            raise RuntimeError("ranker has not been trained or loaded")
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = _metadata_path(model_path)
        manifest_path = ranker_artifact_manifest_path(model_path)
        lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
        temporary_paths: list[Path] = []
        published_paths: list[Path] = []
        publication_complete = False
        try:
            lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise FileExistsError(f"ranker artifact publication is locked: {lock_path}") from error
        try:
            try:
                os.write(lock_descriptor, f"pid={os.getpid()}\n".encode())
                os.fsync(lock_descriptor)
            finally:
                os.close(lock_descriptor)
            if any(target.exists() for target in (model_path, metadata_path, manifest_path)):
                raise FileExistsError("ranker artifacts are immutable and must not be overwritten")
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=model_path.parent,
                prefix=f".{model_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_model_path = Path(handle.name)
            temporary_paths.append(temporary_model_path)
            self._booster.save_model(str(temporary_model_path))
            if temporary_model_path.stat().st_size > MAX_RANKER_MODEL_BYTES:
                raise ValueError("ranker model exceeds the artifact safety limit")
            with temporary_model_path.open("r+b") as handle:
                os.fsync(handle.fileno())
            bound_metadata = replace(
                self._metadata,
                model_sha256=sha256_file(temporary_model_path),
            )
            temporary_metadata_path = _write_json_temp(
                metadata_path,
                bound_metadata.to_dict(),
            )
            temporary_paths.append(temporary_metadata_path)
            if temporary_metadata_path.stat().st_size > MAX_RANKER_JSON_BYTES:
                raise ValueError("ranker metadata exceeds the artifact safety limit")
            manifest_content = _manifest_content(
                ranker_version=bound_metadata.ranker_version,
                model_path=temporary_model_path,
                metadata_path=temporary_metadata_path,
                publication_intent_sha256=_bundle_publication_intent_sha256,
            )
            manifest_files = manifest_content["files"]
            assert isinstance(manifest_files, dict)
            manifest_content["files"] = {
                model_path.name: manifest_files[temporary_model_path.name],
                metadata_path.name: manifest_files[temporary_metadata_path.name],
            }
            manifest_payload = {
                **manifest_content,
                "manifest_sha256": json_sha256(manifest_content),
            }
            temporary_manifest_path = _write_json_temp(manifest_path, manifest_payload)
            temporary_paths.append(temporary_manifest_path)
            if temporary_manifest_path.stat().st_size > MAX_RANKER_JSON_BYTES:
                raise ValueError("ranker artifact manifest exceeds the safety limit")
            artifact_identity = RankerArtifactIdentity(
                model_sha256=bound_metadata.model_sha256 or "",
                metadata_sha256=sha256_file(temporary_metadata_path),
                manifest_sha256=sha256_file(temporary_manifest_path),
            )

            os.replace(temporary_model_path, model_path)
            temporary_paths.remove(temporary_model_path)
            published_paths.append(model_path)
            os.replace(temporary_metadata_path, metadata_path)
            temporary_paths.remove(temporary_metadata_path)
            published_paths.append(metadata_path)
            os.replace(temporary_manifest_path, manifest_path)
            temporary_paths.remove(temporary_manifest_path)
            published_paths.append(manifest_path)
            self._metadata = bound_metadata
            self._artifact_identity = artifact_identity
            self._publication_intent_sha256 = _bundle_publication_intent_sha256
            publication_complete = True
        finally:
            for temporary_path in temporary_paths:
                if temporary_path.exists():
                    temporary_path.unlink()
            if not publication_complete:
                for published_path in reversed(published_paths):
                    if published_path.exists():
                        published_path.unlink()
            if lock_path.exists():
                lock_path.unlink()
        return model_path, metadata_path

    def _adopt_loaded_artifact(self, loaded: LambdaMARTRanker) -> None:
        """Make this instance serve the exact bytes selected by publication."""

        self.parameters = dict(loaded.parameters)
        self.ranker_version = loaded.ranker_version
        self._model = None
        self._booster = loaded._booster
        self._metadata = loaded._metadata
        self._artifact_identity = loaded._artifact_identity
        self._publication_intent_sha256 = loaded._publication_intent_sha256
        self._verified_artifact_loaded = loaded._verified_artifact_loaded

    def publish_bundle(
        self,
        bundle_directory: str | Path,
        *,
        publication_intent_sha256: str,
        model_name: str = "ranker.txt",
    ) -> tuple[Path, Path]:
        """Crash-retryably publish a complete ranker through one directory rename.

        The final directory is immutable.  Files are first sealed and verified in
        a sibling hidden directory, so a process crash before the atomic rename
        cannot expose a partial artifact or block a later retry.  A retry after a
        completed rename adopts the committed artifact when its stable training
        intent matches this ranker.
        """

        if self._booster is None or self._metadata is None:
            raise RuntimeError("ranker has not been trained or loaded")
        if not _is_sha256(publication_intent_sha256):
            raise ValueError("publication_intent_sha256 must be a lowercase SHA-256 digest")
        if not model_name or Path(model_name).name != model_name:
            raise ValueError("model_name must be one direct filename")
        bundle_path = Path(bundle_directory)
        parent = bundle_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if _is_linklike(parent) or not parent.is_dir():
            raise ValueError("ranker bundle parent must be a direct regular directory")
        resolved_parent = parent.resolve(strict=True)
        bundle_path = resolved_parent / bundle_path.name
        if bundle_path == resolved_parent or not bundle_path.name:
            raise ValueError("ranker bundle directory must be a direct child")
        model_path = bundle_path / model_name
        metadata_path = _metadata_path(model_path)
        intent_sha256 = publication_intent_sha256

        def adopt_existing() -> tuple[Path, Path]:
            if _is_linklike(bundle_path) or not bundle_path.is_dir():
                raise FileExistsError("ranker bundle destination is not a regular directory")
            loaded = type(self).load(model_path, feature_builder=self.feature_builder)
            if loaded.publication_intent_sha256 != intent_sha256:
                raise FileExistsError(
                    "ranker bundle is immutable and belongs to a different publication intent"
                )
            self._adopt_loaded_artifact(loaded)
            return model_path, metadata_path

        if bundle_path.exists() or _is_linklike(bundle_path):
            return adopt_existing()

        staging_path = Path(
            tempfile.mkdtemp(
                dir=resolved_parent,
                prefix=f".{bundle_path.name}.publish-",
            )
        )
        committed = False
        activity_lock: RankerStageActivityLock | None = None
        try:
            activity_lock = acquire_ranker_stage_activity_lock(staging_path)
            staged_model_path = staging_path / model_name
            self.save(
                staged_model_path,
                _bundle_publication_intent_sha256=intent_sha256,
            )
            staged = type(self).load(
                staged_model_path,
                feature_builder=self.feature_builder,
            )
            if staged.publication_intent_sha256 != intent_sha256:
                raise RuntimeError("staged ranker publication intent changed during validation")
            activity_lock.release(remove=True)
            _fsync_directory(staging_path)
            try:
                _rename_directory_noreplace(staging_path, bundle_path)
            except FileExistsError:
                return adopt_existing()
            committed = True
            _fsync_directory(resolved_parent)
            committed_artifact = type(self).load(
                model_path,
                feature_builder=self.feature_builder,
            )
            if committed_artifact.publication_intent_sha256 != intent_sha256:
                raise RuntimeError("committed ranker publication intent does not match")
            self._adopt_loaded_artifact(committed_artifact)
            return model_path, metadata_path
        finally:
            # Normal failures remove only this process's private stage.  An abrupt
            # process death may leave an orphan stage, but it remains invisible and
            # never participates in subsequent publication attempts.
            if activity_lock is not None:
                activity_lock.release(remove=True)
            if not committed and staging_path.exists():
                shutil.rmtree(staging_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        feature_builder: RankingFeatureBuilder | None = None,
    ) -> LambdaMARTRanker:
        """Load a native booster and reject feature-contract drift."""

        model_path = Path(path)
        metadata_path = _metadata_path(model_path)
        manifest_path = ranker_artifact_manifest_path(model_path)
        manifest_bytes = _read_bounded_bytes(
            manifest_path,
            maximum_bytes=MAX_RANKER_JSON_BYTES,
            label="ranker artifact manifest",
        )
        manifest = _decode_json_object(manifest_bytes, label="ranker artifact manifest")
        schema_version = manifest.get("schema_version")
        if schema_version not in {
            RANKER_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            RANKER_BUNDLE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported ranker artifact manifest schema")
        expected_manifest_fields = {"schema_version", "ranker_version", "files", "manifest_sha256"}
        if schema_version == RANKER_BUNDLE_ARTIFACT_MANIFEST_SCHEMA_VERSION:
            expected_manifest_fields.add("publication_intent_sha256")
        if set(manifest) != expected_manifest_fields:
            raise ValueError("ranker artifact manifest fields are incomplete or unexpected")
        publication_intent = manifest.get("publication_intent_sha256")
        if schema_version == RANKER_BUNDLE_ARTIFACT_MANIFEST_SCHEMA_VERSION and not _is_sha256(
            publication_intent
        ):
            raise ValueError("ranker artifact publication intent must be SHA-256")
        stored_manifest_sha256 = manifest.get("manifest_sha256")
        manifest_content = dict(manifest)
        manifest_content.pop("manifest_sha256", None)
        if not _is_sha256(stored_manifest_sha256) or stored_manifest_sha256 != json_sha256(
            manifest_content
        ):
            raise ValueError("ranker artifact manifest hash does not match its contents")
        files = manifest.get("files")
        expected_names = {model_path.name, metadata_path.name}
        if not isinstance(files, Mapping) or set(files) != expected_names:
            raise ValueError("ranker artifact manifest file set is invalid")
        file_payloads: dict[str, bytes] = {}
        for file_path in (model_path, metadata_path):
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            entry = files[file_path.name]
            if not isinstance(entry, Mapping):
                raise TypeError("ranker artifact manifest file entry must be an object")
            size_bytes = entry.get("size_bytes")
            digest = entry.get("sha256")
            if type(size_bytes) is not int or size_bytes < 0:
                raise ValueError(
                    "ranker artifact manifest file size must be a non-negative integer"
                )
            if not _is_sha256(digest):
                raise ValueError("ranker artifact manifest file hash must be SHA-256")
            maximum_bytes = (
                MAX_RANKER_MODEL_BYTES if file_path == model_path else MAX_RANKER_JSON_BYTES
            )
            payload_bytes = _read_bounded_bytes(
                file_path,
                maximum_bytes=maximum_bytes,
                label=f"ranker artifact {file_path.name}",
            )
            if len(payload_bytes) != size_bytes:
                raise ValueError(f"ranker artifact {file_path.name} size does not match manifest")
            if _bytes_sha256(payload_bytes) != digest:
                raise ValueError(f"ranker artifact {file_path.name} digest does not match manifest")
            file_payloads[file_path.name] = payload_bytes

        metadata = RankerMetadata.from_dict(
            _decode_json_object(file_payloads[metadata_path.name], label="ranker metadata")
        )
        if manifest.get("ranker_version") != metadata.ranker_version:
            raise ValueError("ranker artifact manifest version does not match metadata")
        if metadata.model_sha256 != _bytes_sha256(file_payloads[model_path.name]):
            raise ValueError("ranker model digest does not match metadata")
        builder = feature_builder or RankingFeatureBuilder()
        if metadata.feature_version != builder.feature_version:
            raise ValueError(
                "feature version mismatch: "
                f"model={metadata.feature_version}, runtime={builder.feature_version}"
            )
        if metadata.feature_names != builder.feature_names:
            raise ValueError("feature order does not match the persisted model")
        instance = cls(
            feature_builder=builder,
            parameters=metadata.parameters,
            ranker_version=metadata.ranker_version,
        )
        try:
            model_text = file_payloads[model_path.name].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("ranker model must be valid UTF-8 LightGBM text") from error
        booster = lgb.Booster(model_str=model_text)
        if booster.num_feature() != len(builder.feature_names):
            raise ValueError("persisted booster has an incompatible feature count")
        if booster.feature_name() != list(builder.feature_names):
            raise ValueError("persisted booster feature names/order do not match metadata")
        expected_objective = str(metadata.parameters.get("objective", ""))
        actual_objective = str(booster.params.get("objective", ""))
        if actual_objective != expected_objective:
            raise ValueError("persisted booster objective does not match metadata")
        instance._booster = booster
        instance._metadata = metadata
        instance._artifact_identity = RankerArtifactIdentity(
            model_sha256=_bytes_sha256(file_payloads[model_path.name]),
            metadata_sha256=_bytes_sha256(file_payloads[metadata_path.name]),
            manifest_sha256=_bytes_sha256(manifest_bytes),
        )
        instance._publication_intent_sha256 = (
            str(publication_intent)
            if schema_version == RANKER_BUNDLE_ARTIFACT_MANIFEST_SCHEMA_VERSION
            else None
        )
        instance._verified_artifact_loaded = True
        return instance


def relative_ndcg_improvement(new_ndcg: float, baseline_ndcg: float) -> float:
    """Return the portfolio claim formula as a percentage."""

    if not (math.isfinite(new_ndcg) and math.isfinite(baseline_ndcg)):
        raise ValueError("NDCG values must be finite")
    if baseline_ndcg <= 0:
        raise ValueError("baseline NDCG must be positive")
    return (new_ndcg - baseline_ndcg) / baseline_ndcg * 100.0

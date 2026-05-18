"""Embedding and vector-search interfaces with a dependency-light fallback."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .models import ProductDocument, SearchHit
from .text import token_features

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

FloatMatrix = NDArray[np.float32]


def resolve_embedding_device(requested_device: str | None = "auto") -> str:
    """Resolve ``auto``/``cuda``/``cpu`` without silently downgrading CUDA."""

    requested = (requested_device or "auto").casefold()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        cuda_available = False
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    return "cuda" if cuda_available else "cpu"


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """Interface implemented by SentenceTransformers and deterministic fallback encoders."""

    model_name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        """Return one L2-normalised vector per input text."""


@runtime_checkable
class VectorSearchBackend(Protocol):
    """Storage-agnostic vector retrieval boundary (for pgvector or in-memory use)."""

    source_name: str

    def search(
        self,
        query: str,
        *,
        category: str,
        top_k: int = 50,
        candidate_ids: set[str] | frozenset[str] | None = None,
    ) -> list[SearchHit]:
        """Return vector-similarity hits scoped to one component category."""


class StableHashEmbeddingEncoder:
    """Deterministic lexical feature hashing for offline and degraded operation.

    This is intentionally labelled as a fallback, not a semantic model.  It
    permits reproducible tests, development without model downloads, and safe
    service degradation when a configured SentenceTransformer is unavailable.
    Python's process-randomised ``hash`` is never used.
    """

    def __init__(self, dimension: int = 512) -> None:
        if dimension < 32:
            raise ValueError("dimension must be at least 32")
        self.dimension = dimension
        self.model_name = f"stable-lexical-hash-v1-{dimension}"

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for feature, weight in token_features(text):
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                column = int.from_bytes(digest[:8], "big") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                matrix[row_index, column] += np.float32(sign * weight)
            norm = float(np.linalg.norm(matrix[row_index]))
            if norm:
                matrix[row_index] /= np.float32(norm)
        return matrix


class SentenceTransformerEmbeddingEncoder:
    """Lazy SentenceTransformers adapter implementing :class:`EmbeddingEncoder`."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str | None = "auto",
        batch_size: int = 64,
        model_path: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if local_files_only and model_path is None:
            raise ValueError("local_files_only requires an explicit model_path")
        self.model_name = model_name
        self.revision = revision
        self.requested_device = device or "auto"
        self.resolved_device = resolve_embedding_device(device)
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.model_path: Path | None = None
        if model_path is not None:
            candidate = Path(model_path)
            if not candidate.exists():
                raise FileNotFoundError(candidate)
            if not candidate.is_dir():
                raise NotADirectoryError(candidate)
            self.model_path = candidate.resolve(strict=True)
        self._model: SentenceTransformer | None = None
        self.dimension = 0

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            load_options: dict[str, object] = {
                "revision": self.revision,
                "device": self.resolved_device,
            }
            if self.local_files_only:
                load_options["local_files_only"] = True
            model = SentenceTransformer(
                str(self.model_path) if self.model_path is not None else self.model_name,
                **load_options,
            )
            self._model = model
            current_dimension_getter = getattr(model, "get_embedding_dimension", None)
            dimension = (
                current_dimension_getter()
                if current_dimension_getter is not None
                else model.get_sentence_embedding_dimension()
            )
            if dimension is None:
                raise RuntimeError("embedding model did not report a dimension")
            self.dimension = int(dimension)
        return self._model

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        model = self._load()
        values = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def warmup(self, *, expected_dimension: int) -> int:
        """Load and probe the model, failing startup on an incompatible bundle."""

        if expected_dimension < 1:
            raise ValueError("expected_dimension must be positive")
        values = self.encode(("BuildSignal semantic encoder readiness probe",))
        if values.shape != (1, expected_dimension):
            raise RuntimeError(
                "semantic encoder warmup returned shape "
                f"{values.shape}; expected (1, {expected_dimension})"
            )
        if not np.isfinite(values).all():
            raise RuntimeError("semantic encoder warmup returned non-finite values")
        norm = float(np.linalg.norm(values[0]))
        if norm <= 0 or not np.isclose(norm, 1.0, rtol=1e-4, atol=1e-4):
            raise RuntimeError("semantic encoder warmup did not return an L2-normalised vector")
        if self.dimension != expected_dimension:
            raise RuntimeError(
                "semantic encoder reported dimension "
                f"{self.dimension}; expected {expected_dimension}"
            )
        return self.dimension


class InMemoryVectorIndex:
    """Cosine search suitable for tests and catalogues before pgvector deployment."""

    source_name = "vector"

    def __init__(
        self,
        documents: Iterable[ProductDocument],
        *,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.encoder = encoder or StableHashEmbeddingEncoder()
        self._documents = tuple(sorted(documents, key=lambda item: item.product_id))
        ids = [document.product_id for document in self._documents]
        if len(ids) != len(set(ids)):
            raise ValueError("product_id values must be unique")
        self._vectors = self.encoder.encode([document.text for document in self._documents])
        if self._vectors.shape[0] != len(self._documents):
            raise ValueError("encoder returned the wrong number of vectors")

    def search(
        self,
        query: str,
        *,
        category: str,
        top_k: int = 50,
        candidate_ids: set[str] | frozenset[str] | None = None,
    ) -> list[SearchHit]:
        if top_k < 1 or not self._documents:
            return []
        query_vector = self.encoder.encode([query])
        if query_vector.shape[0] != 1 or query_vector.shape[1] != self._vectors.shape[1]:
            raise ValueError("encoder returned an incompatible query vector")
        scores = np.matmul(self._vectors, query_vector[0])
        category_key = category.casefold()
        eligible = [
            (document.product_id, float(scores[position]))
            for position, document in enumerate(self._documents)
            if document.category == category_key
            and (candidate_ids is None or document.product_id in candidate_ids)
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchHit(product_id=product_id, score=score, rank=rank, source=self.source_name)
            for rank, (product_id, score) in enumerate(eligible[:top_k], start=1)
        ]

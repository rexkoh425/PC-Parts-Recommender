"""Refresh-safe in-memory storage for generated recommendation results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Protocol

from pc_build_recommender.domain import (
    BuildGenerationRequest,
    BuildRecommendation,
)

from .models import (
    ApplicationBuildGenerationResponse,
    RequestConflictError,
    ResultNotFoundError,
)


@dataclass(frozen=True, slots=True)
class StoredGeneration:
    request: BuildGenerationRequest
    response: ApplicationBuildGenerationResponse
    no_cost_product_ids: frozenset[str]
    owned_product_ids: frozenset[str]
    stored_at: datetime


class ResultStore(Protocol):
    """Storage boundary shared by in-memory and durable serving adapters."""

    def save(
        self,
        request: BuildGenerationRequest,
        response: ApplicationBuildGenerationResponse,
        *,
        no_cost_product_ids: frozenset[str] | None = None,
        owned_product_ids: frozenset[str] | None = None,
    ) -> ApplicationBuildGenerationResponse: ...

    def get_generation(self, request_id: str) -> StoredGeneration | None: ...

    def require_generation(self, request_id: str) -> StoredGeneration: ...

    def get_response(self, request_id: str) -> ApplicationBuildGenerationResponse | None: ...

    def get_request(self, request_id: str) -> BuildGenerationRequest | None: ...

    def request_id_for_build(self, build_id: str) -> str | None: ...

    def get_build(self, build_id: str) -> BuildRecommendation | None: ...

    def require_build(self, build_id: str) -> BuildRecommendation: ...

    def generation_for_build(self, build_id: str) -> StoredGeneration: ...


class InMemoryResultStore:
    """Thread-safe request/build lookup for one application process.

    Values are deep-copied on ingress and egress.  A browser refresh can look
    up an earlier request or build without the caller being able to mutate the
    canonical stored result.  Durable deployments can replace this object with
    a database-backed adapter exposing the same methods.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._requests: dict[str, StoredGeneration] = {}
        self._build_to_request: dict[str, str] = {}

    def save(
        self,
        request: BuildGenerationRequest,
        response: ApplicationBuildGenerationResponse,
        *,
        no_cost_product_ids: frozenset[str] | None = None,
        owned_product_ids: frozenset[str] | None = None,
    ) -> ApplicationBuildGenerationResponse:
        request_copy = request.model_copy(deep=True)
        response_copy = response.model_copy(deep=True)
        free_ids = no_cost_product_ids
        if free_ids is None:
            free_ids = frozenset(item.product_id for item in request.existing_products)
        owner_ids = owned_product_ids
        if owner_ids is None:
            owner_ids = frozenset(item.product_id for item in request.existing_products)
        stored = StoredGeneration(
            request=request_copy,
            response=response_copy,
            no_cost_product_ids=frozenset(free_ids),
            owned_product_ids=frozenset(owner_ids),
            stored_at=datetime.now(UTC),
        )
        with self._lock:
            prior = self._requests.get(response.request_id)
            if prior is not None:
                if (
                    prior.request != request_copy
                    or prior.response != response_copy
                    or prior.no_cost_product_ids != stored.no_cost_product_ids
                    or prior.owned_product_ids != stored.owned_product_ids
                ):
                    raise RequestConflictError(
                        f"request_id is already bound to another result: {response.request_id}"
                    )
                return prior.response.model_copy(deep=True)

            duplicate_build_ids = set(self._build_to_request).intersection(
                build.build_id for build in response.builds
            )
            if duplicate_build_ids:
                raise RequestConflictError(
                    "build IDs are already stored: " + ", ".join(sorted(duplicate_build_ids))
                )
            self._requests[response.request_id] = stored
            for build in response.builds:
                self._build_to_request[build.build_id] = response.request_id
        return response_copy.model_copy(deep=True)

    def get_generation(self, request_id: str) -> StoredGeneration | None:
        with self._lock:
            value = self._requests.get(request_id)
            if value is None:
                return None
            return StoredGeneration(
                request=value.request.model_copy(deep=True),
                response=value.response.model_copy(deep=True),
                no_cost_product_ids=value.no_cost_product_ids,
                owned_product_ids=value.owned_product_ids,
                stored_at=value.stored_at,
            )

    def require_generation(self, request_id: str) -> StoredGeneration:
        value = self.get_generation(request_id)
        if value is None:
            raise ResultNotFoundError(f"request result not found: {request_id}")
        return value

    def get_response(self, request_id: str) -> ApplicationBuildGenerationResponse | None:
        value = self.get_generation(request_id)
        return None if value is None else value.response

    def get_request(self, request_id: str) -> BuildGenerationRequest | None:
        value = self.get_generation(request_id)
        return None if value is None else value.request

    def request_id_for_build(self, build_id: str) -> str | None:
        with self._lock:
            return self._build_to_request.get(build_id)

    def get_build(self, build_id: str) -> BuildRecommendation | None:
        with self._lock:
            request_id = self._build_to_request.get(build_id)
            if request_id is None:
                return None
            generation = self._requests[request_id]
            for build in generation.response.builds:
                if build.build_id == build_id:
                    return build.model_copy(deep=True)
        return None

    def require_build(self, build_id: str) -> BuildRecommendation:
        build = self.get_build(build_id)
        if build is None:
            raise ResultNotFoundError(f"build not found: {build_id}")
        return build

    def generation_for_build(self, build_id: str) -> StoredGeneration:
        request_id = self.request_id_for_build(build_id)
        if request_id is None:
            raise ResultNotFoundError(f"build not found: {build_id}")
        return self.require_generation(request_id)

    def clear(self) -> None:
        with self._lock:
            self._requests.clear()
            self._build_to_request.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._requests)

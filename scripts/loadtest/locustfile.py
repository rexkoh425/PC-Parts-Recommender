"""Bounded Locust workload for PC Build Recommender search and build APIs.

Run this only against a controlled target.  The profile parser blocks insecure remote URLs and
requires an explicit confirmation string for a remote HTTPS origin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

# Locust is deliberately invoked through a pinned ``uvx`` command rather than installed as a
# serving dependency. Its external runtime has no published typing contract in this environment.
from locust import HttpUser, constant, events, task  # type: ignore[import-not-found]

from scripts.loadtest.profile import (
    EndpointProfile,
    LoadProfileError,
    load_profile,
    normalise_target_url,
)

DEFAULT_PROFILE_PATH: Final = Path(__file__).with_name("development-profile.json")
MAX_WAIT_SECONDS: Final = 60.0
MAX_REQUEST_TIMEOUT_SECONDS: Final = 60.0


def _bounded_positive_float(*, name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not 0.0 <= value <= MAX_WAIT_SECONDS:
        raise RuntimeError(f"{name} must be between 0 and {MAX_WAIT_SECONDS}")
    return value


try:
    PROFILE = load_profile(Path(os.environ.get("PCBR_LOAD_PROFILE_FILE", DEFAULT_PROFILE_PATH)))
    TARGET_ORIGIN = normalise_target_url(
        os.environ.get("PCBR_LOAD_BASE_URL", "http://127.0.0.1:8000"),
        confirmation=os.environ.get("PCBR_LOAD_CONFIRM"),
    )
except LoadProfileError as exc:
    raise RuntimeError(f"invalid PCBR load-test configuration: {exc}") from exc

WAIT_SECONDS = _bounded_positive_float(name="PCBR_LOAD_WAIT_SECONDS", default=0.1)
REQUEST_TIMEOUT_SECONDS = _bounded_positive_float(
    name="PCBR_LOAD_REQUEST_TIMEOUT_SECONDS", default=10.0
)
if REQUEST_TIMEOUT_SECONDS == 0.0:
    raise RuntimeError("PCBR_LOAD_REQUEST_TIMEOUT_SECONDS must be greater than zero")


class PcbrApiUser(HttpUser):  # type: ignore[misc]
    """Exercise the declared search/build mix without introducing user-controlled metric labels."""

    host = TARGET_ORIGIN
    wait_time = constant(WAIT_SECONDS)

    def _request(self, endpoint: EndpointProfile) -> None:
        with self.client.request(
            endpoint.method,
            endpoint.path,
            json=endpoint.body,
            name=endpoint.request_name,
            timeout=REQUEST_TIMEOUT_SECONDS,
            catch_response=True,
        ) as response:
            if response.status_code not in endpoint.expected_statuses:
                response.failure(
                    "unexpected status "
                    f"{response.status_code}; expected {endpoint.expected_statuses}"
                )
                return
            if not response.headers.get("x-request-id"):
                response.failure("response is missing x-request-id")
                return
            try:
                payload = response.json()
            except ValueError:
                response.failure("response is not JSON")
                return
            if not isinstance(payload, dict):
                response.failure("response JSON must be an object")

    @task(PROFILE.search.task_weight)  # type: ignore[untyped-decorator]
    def search_products(self) -> None:
        self._request(PROFILE.search)

    @task(PROFILE.build.task_weight)  # type: ignore[untyped-decorator]
    def generate_builds(self) -> None:
        self._request(PROFILE.build)


@events.test_start.add_listener  # type: ignore[untyped-decorator]
def emit_profile_metadata(**_: object) -> None:
    """Expose safe run identity without logging bodies, query text, or credentials."""

    print(
        json.dumps(
            {
                "event": "pcbr_load_test_started",
                "profile_name": PROFILE.profile_name,
                "profile_sha256": PROFILE.sha256,
                "reportability": PROFILE.reportability,
                "target_origin": TARGET_ORIGIN,
                "task_weights": {
                    PROFILE.search.request_name: PROFILE.search.task_weight,
                    PROFILE.build.request_name: PROFILE.build.task_weight,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

"""Pinned BuildCores OpenDB catalogue adapter."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from pipelines.parsing.normalizers import BUILDCORES_CATEGORY_MAP, normalise_buildcores_product
from pipelines.sources.base import (
    ParseResult,
    FetchedSnapshot,
    fetch_http_snapshot,
    rejected_record,
    snapshot_local_file,
)
from pc_build_recommender.catalog.canonical_identity import audit_canonical_envelopes

BUILDCORES_COMMIT = "6a64ab14fb1ab1bc1f3030d36b70bddcc2afeb0f"
BUILDCORES_ARCHIVE_URL = (
    "https://github.com/buildcores/buildcores-open-db/archive/"
    f"{BUILDCORES_COMMIT}.zip"
)
BUILDCORES_LICENSE_NOTE = (
    "BuildCores OpenDB database licensed ODC-By 1.0; attribution required. "
    "Community-maintained specifications require field-level verification for hard compatibility."
)
BUILDCORES_PARSER_VERSION = "buildcores-open-db-v1"
DEFAULT_CATEGORIES = tuple(BUILDCORES_CATEGORY_MAP)

# TODO: rest of this module still to come.

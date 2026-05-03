"""Prepare a conservative real-data Blender performance-training cohort.

The input snapshots are licensed, normalized records produced by the ingestion layer:

* Blender Open Data benchmark observations (CC0 1.0), and
* BuildCores OpenDB component specifications (ODC-By 1.0).

Only one benchmark-version, scene, backend, operating-system, Blender build, benchmark script,
scene checksum, and score-semantics cohort is retained. Hardware names are joined conservatively
to CPU/GPU model families; fuzzy matching is deliberately excluded. Repeated source scores are
aggregated by median. Native higher-is-better throughput scores remain in their source unit, while
supported lower-is-better render durations are converted into a positive higher-is-better
throughput index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from training._common import (
    portable_path_reference,
    require_host_memory_headroom,
    sha256_file,
    write_json,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BLENDER_RECORDS = Path(
    "data/processed/blender_open_data/"
    "c0f9d35c20807776138b0590097177b8ef2172119cc19aae8d1bad1b55af4833/"
    "records.jsonl"
)
DEFAULT_BUILDCORES_RECORDS = Path(
    "data/processed/buildcores_open_db/"
    "f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/"
    "portfolio-3000/records.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/model_training/blender_performance")
DEFAULT_MAX_RECORD_BYTES = 1_000_000
DEFAULT_MAX_HOST_USED_GB = 55.0
DEFAULT_MINIMUM_FREE_MEMORY_MB = 1_024.0
DEFAULT_CATALOG_MEMORY_EXPANSION_FACTOR = 12.0
DEFAULT_PREPARATION_RUNTIME_MEMORY_MB = 512.0
MATCHING_METHOD_VERSION = "conservative_normalized_hardware_family_v4"

CPU_FEATURE_COLUMNS: tuple[str, ...] = (
    "core_count",
    "thread_count",
    "base_clock_ghz",
    "boost_clock_ghz",
    "tdp_watts",
)
GPU_FEATURE_COLUMNS: tuple[str, ...] = (
    "base_clock_mhz",
    "boost_clock_mhz",
    "vram_gb",
    "board_power_watts",
)
OUTPUT_METADATA_COLUMNS: tuple[str, ...] = (
    "product_id",
    "product_family",
    "hardware_generation",
    "category",
    "workload",
)
OUTPUT_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "benchmark_version",
    "scene",
    "backend",
    "operating_system",
    "blender_build_hash",
    "benchmark_script",
    "scene_checksum",
    "observed_source_value_median",
    "observed_source_value_mad",
    "observed_source_unit",
    "observed_source_higher_is_better",
    "observed_source_score_field",
    "benchmark_observation_count",
    "target_score",
    "target_transform",
    "is_synthetic",
    "eligible_for_external_claims",
    "dataset_role",
    "matching_method",
    "benchmark_hardware_names_json",
    "catalog_product_ids_json",
    "source_urls_json",
    "data_provenance",
)

Category = Literal["cpu", "gpu"]
Cohort = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class ScoreSemantics:
    """Comparable meaning of a normalized Blender score."""

    unit: str
    higher_is_better: bool
    source_field: str


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionIdentity:
    """Source fields that make otherwise similar benchmark results comparable."""

    blender_build_hash: str
    benchmark_script: str
    scene_checksum: str

    def __post_init__(self) -> None:
        invalid = [
            name
            for name, value in self.as_dict().items()
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid:
            raise ValueError(
                "execution identity requires explicit non-empty fields: " + ", ".join(invalid)
            )

    def as_dict(self) -> dict[str, str]:
        return {
            "blender_build_hash": self.blender_build_hash,
            "benchmark_script": self.benchmark_script,
            "scene_checksum": self.scene_checksum,
        }


_CPU_FEATURE_SOURCE: dict[str, str] = {
    "core_count": "core_count",
    "thread_count": "thread_count",
    "base_clock_ghz": "base_clock_ghz",
    "boost_clock_ghz": "boost_clock_ghz",
    "tdp_watts": "tdp_watts",
}
_GPU_FEATURE_SOURCE: dict[str, str] = {
    "base_clock_mhz": "base_clock_mhz",
    "boost_clock_mhz": "boost_clock_mhz",
    "vram_gb": "vram_gb",
    "board_power_watts": "board_power_watts",
}
_BACKEND_CATEGORY: dict[str, Category] = {"CPU": "cpu", "CUDA": "gpu", "OPTIX": "gpu"}


def normalize_hardware_name(value: object) -> str:
    """Normalize hardware text without deleting model-variant boundaries."""

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def _single_match(pattern: str, text: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text))
    normalized_matches = {tuple(match.groups()) for match in matches}
    if len(normalized_matches) != 1:
        return None
    return matches[0]


def extract_hardware_family(value: object, category: Category) -> str | None:
    """Extract an exact CPU/GPU silicon-family key; never use approximate similarity."""

    text = normalize_hardware_name(value)
    if category == "cpu":
        match = _single_match(
            r"amd(?: r)? ryzen (threadripper(?: pro)?|[3579])\s*(\d{4,5})([a-z]{0,3})\b",
            text,
        )
        if match:
            series, number, suffix = match.groups()
            return f"cpu:amd_ryzen_{series.replace(' ', '_')}_{number}{suffix}"
        match = _single_match(
            r"intel(?: r)? core(?: tm)? i([3579])\s*(\d{4,5})([a-z]{0,3})\b",
            text,
        )
        if match:
            tier, number, suffix = match.groups()
            return f"cpu:intel_core_i{tier}_{number}{suffix}"
        match = _single_match(r"amd(?: r)? epyc\s*(\d{4})([a-z]{0,2})\b", text)
        if match:
            number, suffix = match.groups()
            return f"cpu:amd_epyc_{number}{suffix}"
        return None

    match = _single_match(
        r"(?:nvidia )?geforce\s*(rtx|gtx|gt)\s*(\d{3,4})"
        r"(?:\s*(ti|super))?(?:\s*(laptop gpu))?",
        text,
    )
    if match:
        series, number, variant, laptop = match.groups()
        parts = ["gpu:nvidia_geforce", series, number]
        if variant:
            parts.append(variant)
        if laptop:
            parts.append("laptop")
        return "_".join(parts)
    match = _single_match(
        r"(?:amd )?radeon\s*(rx|r9|r7|r5|hd)\s*(\d{3,4})"
        r"(?:\s*(xtx|xt|gre|x2))?",
        text,
    )
    if match:
        series, number, variant = match.groups()
        parts = ["gpu:amd_radeon", series, number]
        if variant:
            parts.append(variant)
        return "_".join(parts)
    match = _single_match(r"(?:nvidia )?quadro\s*rtx\s*(\d{4})", text)
    if match:
        return f"gpu:nvidia_quadro_rtx_{match.group(1)}"
    match = _single_match(r"(?:nvidia )?rtx\s*a\s*(\d{4})", text)
    if match:
        return f"gpu:nvidia_rtx_a{match.group(1)}"
    if re.search(r"(?:amd )?radeon\s*vii(?:\s|$)", text):
        return "gpu:amd_radeon_vii"
    return None


def extract_explicit_vram_gb(value: object) -> int | None:
    text = normalize_hardware_name(value)
    values = {int(match.group(1)) for match in re.finditer(r"(\d{1,3})\s*gb(?:\s|$)", text)}
    return next(iter(values)) if len(values) == 1 else None


@dataclass(slots=True)
class _JsonlRecordStream:
    """One-pass JSONL reader that records the observed source-row count.

    The Blender snapshot can contain hundreds of thousands of records.  The
    preparer needs a catalogue index and only the conservatively joined
    observations, not a second in-memory copy of every source envelope.
    """

    path: Path
    maximum_record_bytes: int = DEFAULT_MAX_RECORD_BYTES
    records_read: int = 0
    _consumed: bool = False

    def __post_init__(self) -> None:
        if self.maximum_record_bytes <= 0:
            raise ValueError("maximum_record_bytes must be positive")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._consumed:
            raise RuntimeError(f"JSONL record stream was already consumed: {self.path}")
        self._consumed = True
        # ``readline(limit + 1)`` refuses a pathological source row before
        # decoding JSON.  Iterating a text file directly would first allocate
        # the entire line, defeating the preparer's otherwise bounded design.
        with self.path.open("rb") as source:
            line_number = 0
            while line := source.readline(self.maximum_record_bytes + 1):
                line_number += 1
                if len(line) > self.maximum_record_bytes:
                    raise MemoryError(
                        f"{self.path}:{line_number}: JSONL record exceeds the "
                        f"{self.maximum_record_bytes}-byte limit"
                    )
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{self.path}:{line_number}: expected a JSON object")
                self.records_read += 1
                yield value
        if not self.records_read:
            raise ValueError(f"{self.path} contains no records")


def estimate_blender_preparation_memory_mib(
    catalog_records_path: Path,
    *,
    maximum_record_bytes: int,
    catalog_memory_expansion_factor: float,
    runtime_memory_mb: float,
) -> float:
    """Return a conservative host-RAM reservation for streamed preparation.

    Blender records are line-streamed and matched observations live in the
    temporary SQLite store, so the raw Blender file is deliberately absent
    from this estimate. The catalogue is the only full source retained in
    Python, with a conservative expansion multiplier for decoded mappings,
    family indexes, and feature summaries.
    """

    if maximum_record_bytes <= 0:
        raise ValueError("maximum_record_bytes must be positive")
    if catalog_memory_expansion_factor < 1:
        raise ValueError("catalog_memory_expansion_factor must be at least one")
    if runtime_memory_mb < 0:
        raise ValueError("runtime_memory_mb must be non-negative")
    if not catalog_records_path.is_file():
        raise FileNotFoundError(catalog_records_path)
    mebibyte = 1024 * 1024
    catalog_mib = catalog_records_path.stat().st_size / mebibyte
    record_buffer_mib = (2 * maximum_record_bytes) / mebibyte
    sqlite_cache_mib = _MATCH_STORE_CACHE_KIB / 1024
    return (
        (catalog_mib * catalog_memory_expansion_factor)
        + record_buffer_mib
        + sqlite_cache_mib
        + runtime_memory_mb
    )


def _median_numeric(products: Sequence[Mapping[str, Any]], source_field: str) -> float | None:
    values: list[float] = []
    for product in products:
        attributes = product.get("category_attributes")
        if not isinstance(attributes, Mapping):
            continue
        value = attributes.get(source_field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return float(statistics.median(values)) if values else None


def _integer_like(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _generation_for(category: Category, family: str, products: Sequence[Mapping[str, Any]]) -> str:
    if category == "cpu":
        architectures: list[str] = []
        canonical_architectures = {
            "coffee_lake_refresh": "coffee_lake",
            "haswell_refresh": "haswell",
            "raptor_lake_refresh": "raptor_lake",
            "broadwell_e": "broadwell_e",
            "ivy_bridge_e": "ivy_bridge_e",
            "sandy_bridge_ep": "sandy_bridge_ep",
            "ivy_bridge_ep": "ivy_bridge_ep",
        }
        for product in products:
            attributes = product.get("category_attributes")
            architecture = (
                attributes.get("architecture") if isinstance(attributes, Mapping) else None
            )
            if architecture:
                normalized = normalize_hardware_name(architecture).replace(" ", "_")
                architectures.append(canonical_architectures.get(normalized, normalized))
        if architectures:
            return Counter(architectures).most_common(1)[0][0]

    if category == "gpu":
        match = re.search(r"nvidia_geforce_(rtx|gtx|gt)_(\d{3,4})", family)
        if match:
            series, number = match.groups()
            prefix_length = 2 if len(number) == 4 else 1
            return f"nvidia_{series}_{number[:prefix_length]}_series"
        match = re.search(r"amd_radeon_(rx|r9|r7|r5|hd)_(\d{3,4})", family)
        if match:
            series, number = match.groups()
            return f"amd_{series}_{number[0]}_series"
    else:
        match = re.search(r"cpu:amd_ryzen_[^_]+_(\d{4,5})", family)
        if match:
            return f"amd_ryzen_{match.group(1)[0]}000_series"
        match = re.search(r"cpu:intel_core_i\d_(\d{4,5})", family)
        if match:
            number = match.group(1)
            generation = number[:2] if len(number) == 5 else number[0]
            return f"intel_core_gen_{generation}"

    return "unknown"


@dataclass(frozen=True, slots=True)
class CatalogFamily:
    family: str
    leakage_family: str
    category: Category
    generation: str
    features: dict[str, float]
    product_ids: tuple[str, ...]
    vram_values: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CohortSummary:
    """Aggregate metadata for one exact comparable benchmark cohort."""

    cohort: Cohort
    execution_identity: BenchmarkExecutionIdentity
    score_semantics: ScoreSemantics
    joined_observations: int
    hardware_identities: int
    leakage_groups: int


_MATCH_STORE_INSERT_BATCH_SIZE = 1_000
_MATCH_STORE_CACHE_KIB = 8 * 1024
_MATCH_STORE_COHORT_COLUMNS = (
    "benchmark_version, scene, backend, operating_system, blender_build_hash, "
    "benchmark_script, scene_checksum, source_unit, source_higher_is_better, "
    "source_score_field"
)
_MATCH_STORE_COHORT_WHERE = " AND ".join(
    f"{column} = ?"
    for column in (
        "benchmark_version",
        "scene",
        "backend",
        "operating_system",
        "blender_build_hash",
        "benchmark_script",
        "scene_checksum",
        "source_unit",
        "source_higher_is_better",
        "source_score_field",
    )
)


def _configure_match_store(connection: sqlite3.Connection) -> None:
    """Create a small-cache, file-backed temporary store for matched observations.

    A Blender source snapshot can have hundreds of thousands of rows.  Holding
    every conservatively joined envelope in Python makes memory use proportional
    to raw benchmark volume.  The preparer only needs all rows long enough to
    pick an exact cohort; SQLite lets it do that aggregation on disk and later
    read one selected hardware family at a time.
    """

    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA mmap_size = 0")
    connection.execute(f"PRAGMA cache_size = {-_MATCH_STORE_CACHE_KIB}")
    connection.execute(
        """
        CREATE TABLE matched_observations (
            benchmark_version TEXT NOT NULL,
            scene TEXT NOT NULL,
            backend TEXT NOT NULL,
            operating_system TEXT NOT NULL,
            blender_build_hash TEXT NOT NULL,
            benchmark_script TEXT NOT NULL,
            scene_checksum TEXT NOT NULL,
            source_unit TEXT NOT NULL,
            source_higher_is_better INTEGER NOT NULL,
            source_score_field TEXT NOT NULL,
            family TEXT NOT NULL,
            leakage_family TEXT NOT NULL,
            score REAL NOT NULL,
            source_url TEXT NOT NULL,
            hardware_name TEXT NOT NULL
        )
        """
    )


def _cohort_parameters(
    cohort: Cohort,
    execution_identity: BenchmarkExecutionIdentity,
    score_semantics: ScoreSemantics,
) -> tuple[str | int, ...]:
    return (
        *cohort,
        execution_identity.blender_build_hash,
        execution_identity.benchmark_script,
        execution_identity.scene_checksum,
        score_semantics.unit,
        int(score_semantics.higher_is_better),
        score_semantics.source_field,
    )


def _cohort_summary_from_row(row: tuple[Any, ...]) -> _CohortSummary:
    return _CohortSummary(
        cohort=(str(row[0]), str(row[1]), str(row[2]), str(row[3])),
        execution_identity=BenchmarkExecutionIdentity(
            blender_build_hash=str(row[4]),
            benchmark_script=str(row[5]),
            scene_checksum=str(row[6]),
        ),
        score_semantics=ScoreSemantics(
            unit=str(row[7]),
            higher_is_better=bool(row[8]),
            source_field=str(row[9]),
        ),
        joined_observations=int(row[10]),
        hardware_identities=int(row[11]),
        leakage_groups=int(row[12]),
    )


def _candidate_cohorts(connection: sqlite3.Connection) -> list[_CohortSummary]:
    rows = connection.execute(
        f"""
        SELECT
            {_MATCH_STORE_COHORT_COLUMNS},
            COUNT(*) AS joined_observations,
            COUNT(DISTINCT family) AS hardware_identities,
            COUNT(DISTINCT leakage_family) AS leakage_groups
        FROM matched_observations
        GROUP BY {_MATCH_STORE_COHORT_COLUMNS}
        ORDER BY
            leakage_groups DESC,
            hardware_identities DESC,
            joined_observations DESC,
            benchmark_version,
            scene,
            backend,
            operating_system,
            blender_build_hash,
            benchmark_script,
            scene_checksum,
            source_unit,
            source_higher_is_better,
            source_score_field
        """
    ).fetchall()
    return [_cohort_summary_from_row(cast(tuple[Any, ...], row)) for row in rows]


def _median_score(
    connection: sqlite3.Connection,
    *,
    where_clause: str,
    parameters: tuple[str | int, ...],
    count: int,
) -> float:
    if count <= 0:
        raise RuntimeError("cannot calculate a median for an empty matched-observation group")
    limit = 2 if count % 2 == 0 else 1
    offset = (count - 1) // 2
    rows = connection.execute(
        f"""
        SELECT score
        FROM matched_observations
        WHERE {where_clause}
        ORDER BY score
        LIMIT ? OFFSET ?
        """,
        (*parameters, limit, offset),
    ).fetchall()
    values = [float(row[0]) for row in rows]
    if len(values) != limit:
        raise RuntimeError("matched-observation median query returned an incomplete result")
    return float(statistics.mean(values))


def _mad_score(
    connection: sqlite3.Connection,
    *,
    where_clause: str,
    parameters: tuple[str | int, ...],
    count: int,
    median: float,
) -> float:
    if count <= 0:
        raise RuntimeError("cannot calculate a MAD for an empty matched-observation group")
    limit = 2 if count % 2 == 0 else 1
    offset = (count - 1) // 2
    rows = connection.execute(
        f"""
        SELECT ABS(score - ?)
        FROM matched_observations
        WHERE {where_clause}
        ORDER BY ABS(score - ?)
        LIMIT ? OFFSET ?
        """,
        (median, *parameters, median, limit, offset),
    ).fetchall()
    values = [float(row[0]) for row in rows]
    if len(values) != limit:
        raise RuntimeError("matched-observation MAD query returned an incomplete result")
    return float(statistics.mean(values))


def _catalog_index(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[tuple[Category, str], tuple[Mapping[str, Any], ...]], Counter[str]]:
    indexed: dict[tuple[Category, str], list[Mapping[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for envelope in records:
        data = envelope.get("data")
        if not isinstance(data, Mapping) or data.get("category") not in {"cpu", "gpu"}:
            continue
        counts[f"catalog_{data['category']}_records"] += 1
        if not bool(envelope.get("training_eligible", False)):
            counts["catalog_training_ineligible"] += 1
            continue
        category: Category = data["category"]
        category_attributes = data.get("category_attributes")
        gpu_architecture = (
            category_attributes.get("architecture")
            if category == "gpu" and isinstance(category_attributes, Mapping)
            else None
        )
        identity_text = gpu_architecture or data.get("canonical_name", "")
        family = extract_hardware_family(identity_text, category)
        if family is None:
            counts[f"catalog_{category}_unparsed_name"] += 1
            continue
        indexed[(category, family)].append(data)
        counts[f"catalog_{category}_parsed_name"] += 1
    return {key: tuple(value) for key, value in indexed.items()}, counts


def _leakage_family(category: Category, exact_family: str) -> str:
    """Collapse close package/performance variants only for split isolation."""

    if category == "cpu":
        # Intel X-series and AMD Threadripper PRO WX processors are distinct
        # workstation/HEDT product lines, rather than ordinary factory-bin
        # variants.  Collapsing (for example) Core i9-9900 and i9-9900X puts
        # Skylake-X and Coffee Lake rows in one leakage group with mutually
        # incompatible generation labels.  Keep those boundaries exact while
        # still grouping ordinary suffix-only desktop/mobile variants.
        if exact_family.startswith("cpu:intel_core_"):
            return re.sub(r"(?<=\d)(?:kf|ks|k|f|g|t|h|u|p)$", "", exact_family)
        if exact_family.startswith("cpu:amd_ryzen_"):
            return re.sub(r"(?<=\d)(?:xt|x|g|t|h|u|p)$", "", exact_family)
        return exact_family
    family = re.sub(r":\d+gb$", "", exact_family)
    return re.sub(r"_(?:ti|super)$", "", family)


def _catalog_family(
    *,
    category: Category,
    base_family: str,
    products: Sequence[Mapping[str, Any]],
    observed_hardware_name: object,
) -> tuple[CatalogFamily | None, str | None]:
    selected = list(products)
    vram_values = sorted(
        {
            int(value)
            for product in products
            if isinstance((attributes := product.get("category_attributes")), Mapping)
            and isinstance((value := attributes.get("vram_gb")), int | float)
        }
    )
    family = base_family
    if category == "gpu":
        explicit_vram = extract_explicit_vram_gb(observed_hardware_name)
        if explicit_vram is not None:
            selected = [
                product
                for product in products
                if isinstance((attributes := product.get("category_attributes")), Mapping)
                and attributes.get("vram_gb") == explicit_vram
            ]
            if not selected:
                return None, "explicit_vram_conflict"
            family = f"{base_family}:{explicit_vram}gb"
            vram_values = [explicit_vram]
        elif len(vram_values) > 1:
            return None, "ambiguous_vram_variant"
        elif len(vram_values) == 1:
            family = f"{base_family}:{vram_values[0]}gb"

    source_fields = _CPU_FEATURE_SOURCE if category == "cpu" else _GPU_FEATURE_SOURCE
    features = {
        output_name: _median_numeric(selected, source_name)
        for output_name, source_name in source_fields.items()
    }
    missing = [name for name, value in features.items() if value is None]
    if missing:
        return None, f"missing_numeric_features:{','.join(missing)}"
    product_ids = tuple(sorted(str(product["product_id"]) for product in selected))
    return (
        CatalogFamily(
            family=family,
            leakage_family=_leakage_family(category, family),
            category=category,
            generation=_generation_for(category, family, selected),
            features={name: float(value) for name, value in features.items() if value is not None},
            product_ids=product_ids,
            vram_values=tuple(vram_values),
        ),
        None,
    )


def _cohort_for(envelope: Mapping[str, Any]) -> Cohort:
    data = envelope["data"]
    metadata = envelope["normalisation_metadata"]
    return (
        str(data["benchmark_version"]),
        str(data["preset"]),
        str(metadata["device_type"]),
        str(data["operating_system"]),
    )


def _execution_identity(envelope: Mapping[str, Any]) -> BenchmarkExecutionIdentity:
    metadata = envelope.get("normalisation_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("execution identity requires normalisation metadata")
    fields = {
        "blender_build_hash": metadata.get("blender_build_hash"),
        "benchmark_script": metadata.get("benchmark_script"),
        "scene_checksum": metadata.get("scene_checksum"),
    }
    invalid = [
        name for name, value in fields.items() if not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        raise ValueError(
            "execution identity requires explicit non-empty fields: " + ", ".join(invalid)
        )
    return BenchmarkExecutionIdentity(
        blender_build_hash=cast(str, fields["blender_build_hash"]).strip(),
        benchmark_script=cast(str, fields["benchmark_script"]).strip(),
        scene_checksum=cast(str, fields["scene_checksum"]).strip(),
    )


def _score_semantics(envelope: Mapping[str, Any]) -> ScoreSemantics:
    """Return a supported explicit score contract, rejecting ambiguous direction or units."""

    data = envelope.get("data")
    metadata = envelope.get("normalisation_metadata")
    if not isinstance(data, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("score semantics require data and normalisation metadata")
    unit = data.get("unit")
    higher_is_better = data.get("higher_is_better")
    source_field = metadata.get("score_source_field")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("score unit must be an explicit non-empty string")
    if not isinstance(higher_is_better, bool):
        raise ValueError("score direction must be an explicit boolean")
    if not isinstance(source_field, str) or not source_field.strip():
        raise ValueError("score source field must be an explicit non-empty string")

    semantics = ScoreSemantics(
        unit=unit.strip(),
        higher_is_better=higher_is_better,
        source_field=source_field.strip(),
    )
    supported = {
        ScoreSemantics("samples/minute", True, "samples_per_minute"),
        ScoreSemantics("seconds", False, "total_render_time"),
        ScoreSemantics("seconds", False, "render_time_no_sync"),
    }
    if semantics not in supported:
        raise ValueError(
            "unsupported or inconsistent Blender score semantics: "
            f"unit={semantics.unit!r}, higher_is_better={semantics.higher_is_better!r}, "
            f"score_source_field={semantics.source_field!r}"
        )
    return semantics


def _workload_slug(cohort: Cohort, execution_identity: BenchmarkExecutionIdentity) -> str:
    version, scene, backend, operating_system = cohort
    identity_digest = hashlib.sha256(
        json.dumps(execution_identity.as_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    raw = f"blender_{version}_{scene}_{backend}_{operating_system}_execution_{identity_digest}"
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", raw.casefold())).strip("_")


def _stable_product_id(
    family: str,
    cohort: Cohort,
    execution_identity: BenchmarkExecutionIdentity,
    score_semantics: ScoreSemantics,
) -> str:
    payload = json.dumps(
        [
            family,
            cohort,
            execution_identity.as_dict(),
            score_semantics.unit,
            score_semantics.higher_is_better,
            score_semantics.source_field,
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    return f"blender_perf_{hashlib.sha256(payload).hexdigest()[:24]}"


def prepare_blender_performance(
    blender_records: Iterable[Mapping[str, Any]],
    catalog_records: Iterable[Mapping[str, Any]],
    *,
    category_filter: Literal["auto", "cpu", "gpu"] = "auto",
    pinned_cohort: Cohort | None = None,
    pinned_execution_identity: BenchmarkExecutionIdentity | None = None,
    target_scale: float = 1000.0,
    minimum_pilot_products: int = 15,
    minimum_credible_products: int = 100,
    source_blockers: Sequence[str] | None = None,
    workload: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join, select, and aggregate one comparable Blender benchmark cohort."""

    if category_filter not in {"auto", "cpu", "gpu"}:
        raise ValueError("category_filter must be auto, cpu, or gpu")
    if not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("target_scale must be finite and positive")
    if minimum_pilot_products < 3:
        raise ValueError("minimum_pilot_products must be at least three")
    if minimum_credible_products < minimum_pilot_products:
        raise ValueError("minimum_credible_products cannot be below the pilot minimum")
    if workload is not None and not workload.strip():
        raise ValueError("workload override must not be blank")
    if pinned_execution_identity is not None and pinned_cohort is None:
        raise ValueError("pinned_execution_identity requires pinned_cohort")

    catalog, match_counts = _catalog_index(catalog_records)
    family_cache: dict[tuple[Category, str, int | None], CatalogFamily | None] = {}
    catalog_families: dict[str, CatalogFamily] = {}
    with tempfile.TemporaryDirectory(prefix="pc-build-recommender-blender-") as temporary_directory:
        database_path = Path(temporary_directory) / "matched-observations.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            _configure_match_store(connection)
            pending_insertions: list[tuple[str | int | float, ...]] = []
            for envelope in blender_records:
                match_counts["blender_observations_seen"] += 1
                if not bool(envelope.get("training_eligible", False)):
                    match_counts["blender_training_ineligible"] += 1
                    continue
                data = envelope.get("data")
                metadata = envelope.get("normalisation_metadata")
                if not isinstance(data, Mapping) or not isinstance(metadata, Mapping):
                    match_counts["blender_invalid_envelope"] += 1
                    continue
                backend = str(metadata.get("device_type", ""))
                category = _BACKEND_CATEGORY.get(backend)
                if category is None:
                    match_counts["blender_unsupported_backend"] += 1
                    continue
                if category_filter != "auto" and category != category_filter:
                    match_counts["blender_category_filtered"] += 1
                    continue
                hardware_name = metadata.get("hardware_name", "")
                base_family = extract_hardware_family(hardware_name, category)
                if base_family is None:
                    match_counts[f"blender_{category}_unparsed_name"] += 1
                    continue
                products = catalog.get((category, base_family))
                if not products:
                    match_counts[f"blender_{category}_no_catalog_family"] += 1
                    continue
                observed_vram = (
                    extract_explicit_vram_gb(hardware_name) if category == "gpu" else None
                )
                cache_key = (category, base_family, observed_vram)
                if cache_key not in family_cache:
                    summary, rejection = _catalog_family(
                        category=category,
                        base_family=base_family,
                        products=products,
                        observed_hardware_name=hardware_name,
                    )
                    family_cache[cache_key] = summary
                    if rejection:
                        match_counts[f"join_rejected_{rejection}"] += 1
                summary = family_cache[cache_key]
                if summary is None:
                    match_counts["join_rejected_catalog_variant_or_features"] += 1
                    continue
                if category == "cpu":
                    socket_count = _integer_like(metadata.get("system_cpu_sockets"))
                    observed_cores = _integer_like(metadata.get("system_cpu_cores"))
                    observed_threads = _integer_like(metadata.get("cpu_threads"))
                    expected_cores = int(summary.features["core_count"])
                    expected_threads = int(summary.features["thread_count"])
                    if socket_count != 1:
                        match_counts["join_rejected_cpu_not_single_socket"] += 1
                        continue
                    if observed_cores != expected_cores or observed_threads != expected_threads:
                        match_counts["join_rejected_cpu_topology_mismatch"] += 1
                        continue
                try:
                    execution_identity = _execution_identity(envelope)
                except ValueError:
                    match_counts["join_rejected_execution_identity"] += 1
                    continue
                try:
                    score_semantics = _score_semantics(envelope)
                except ValueError:
                    match_counts["join_rejected_score_semantics"] += 1
                    continue
                score = data.get("score")
                if isinstance(score, bool) or not isinstance(score, int | float):
                    match_counts["join_rejected_invalid_score"] += 1
                    continue
                numeric_score = float(score)
                if not math.isfinite(numeric_score) or numeric_score <= 0:
                    match_counts["join_rejected_invalid_score"] += 1
                    continue
                cohort = _cohort_for(envelope)
                catalog_families[summary.family] = summary
                pending_insertions.append(
                    (
                        *_cohort_parameters(cohort, execution_identity, score_semantics),
                        summary.family,
                        summary.leakage_family,
                        numeric_score,
                        str(data["source_url"]),
                        str(metadata["hardware_name"]),
                    )
                )
                match_counts["joined_observations"] += 1
                if len(pending_insertions) >= _MATCH_STORE_INSERT_BATCH_SIZE:
                    connection.executemany(
                        """
                        INSERT INTO matched_observations VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        pending_insertions,
                    )
                    pending_insertions.clear()
            if pending_insertions:
                connection.executemany(
                    """
                    INSERT INTO matched_observations VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    pending_insertions,
                )
            connection.execute(
                f"""
                CREATE INDEX matched_observations_cohort_family_score
                ON matched_observations (
                    {_MATCH_STORE_COHORT_COLUMNS}, family, leakage_family, score
                )
                """
            )
            connection.commit()

            candidate_summaries = _candidate_cohorts(connection)
            if pinned_cohort is not None:
                pinned_matches = [
                    summary
                    for summary in candidate_summaries
                    if summary.cohort == pinned_cohort
                    and (
                        pinned_execution_identity is None
                        or summary.execution_identity == pinned_execution_identity
                    )
                ]
                if not pinned_matches:
                    identity_detail = (
                        f", execution_identity={pinned_execution_identity.as_dict()}"
                        if pinned_execution_identity is not None
                        else ""
                    )
                    raise ValueError(
                        "pinned cohort has no conservatively joined rows: "
                        f"{pinned_cohort}{identity_detail}"
                    )
                if len(pinned_matches) != 1:
                    contracts = sorted(
                        (
                            summary.execution_identity.blender_build_hash,
                            summary.execution_identity.benchmark_script,
                            summary.execution_identity.scene_checksum,
                            summary.score_semantics.unit,
                            summary.score_semantics.higher_is_better,
                            summary.score_semantics.source_field,
                        )
                        for summary in pinned_matches
                    )
                    raise ValueError(
                        "pinned cohort contains multiple execution or score contracts and cannot "
                        "be aggregated; pin blender_build_hash, benchmark_script, and "
                        f"scene_checksum: {contracts}"
                    )
                selected_summary = pinned_matches[0]
            else:
                if not candidate_summaries:
                    raise ValueError("no Blender observations joined conservatively to BuildCores")
                selected_summary = candidate_summaries[0]

            selected = selected_summary.cohort
            selected_execution_identity = selected_summary.execution_identity
            selected_semantics = selected_summary.score_semantics
            if selected_summary.hardware_identities < minimum_pilot_products:
                raise ValueError(
                    f"largest valid cohort has {selected_summary.hardware_identities} products; "
                    f"pilot minimum is {minimum_pilot_products}"
                )

            leakage_group_count = selected_summary.leakage_groups
            size_gate_passed = leakage_group_count >= minimum_credible_products
            effective_source_blockers = (
                list(source_blockers)
                if source_blockers is not None
                else ["source snapshot completeness and sampling provenance were not supplied"]
            )
            dataset_promotable = size_gate_passed and not effective_source_blockers
            if dataset_promotable:
                role = "measured_evaluation_candidate"
            elif size_gate_passed:
                role = "measured_research_candidate_non_promotable"
            else:
                role = "measured_pilot_non_promotable"
            selected_workload = (
                workload.strip()
                if workload is not None
                else _workload_slug(selected, selected_execution_identity)
            )
            selected_parameters = _cohort_parameters(
                selected, selected_execution_identity, selected_semantics
            )
            family_rows = connection.execute(
                f"""
                SELECT DISTINCT family
                FROM matched_observations
                WHERE {_MATCH_STORE_COHORT_WHERE}
                ORDER BY family
                """,
                selected_parameters,
            ).fetchall()
            output_rows: list[dict[str, Any]] = []
            for family_row in family_rows:
                family = str(family_row[0])
                catalog_family = catalog_families.get(family)
                if catalog_family is None:
                    raise RuntimeError(f"matched catalogue family is unavailable: {family}")
                family_where = f"{_MATCH_STORE_COHORT_WHERE} AND family = ?"
                family_parameters = (*selected_parameters, family)
                count_row = connection.execute(
                    f"SELECT COUNT(*) FROM matched_observations WHERE {family_where}",
                    family_parameters,
                ).fetchone()
                if count_row is None:
                    raise RuntimeError("matched-observation count query returned no result")
                observation_count = int(count_row[0])
                median_source_value = _median_score(
                    connection,
                    where_clause=family_where,
                    parameters=family_parameters,
                    count=observation_count,
                )
                mad_source_value = _mad_score(
                    connection,
                    where_clause=family_where,
                    parameters=family_parameters,
                    count=observation_count,
                    median=median_source_value,
                )
                if selected_semantics.higher_is_better:
                    target_score = median_source_value
                    target_transform = f"median({selected_semantics.source_field})"
                else:
                    target_score = target_scale / median_source_value
                    target_transform = (
                        f"{target_scale:g} / median({selected_semantics.source_field})"
                    )
                source_urls = [
                    str(row[0])
                    for row in connection.execute(
                        f"""
                        SELECT DISTINCT source_url
                        FROM matched_observations
                        WHERE {family_where}
                        ORDER BY source_url
                        """,
                        family_parameters,
                    )
                ]
                hardware_names = [
                    str(row[0])
                    for row in connection.execute(
                        f"""
                        SELECT DISTINCT hardware_name
                        FROM matched_observations
                        WHERE {family_where}
                        ORDER BY hardware_name
                        """,
                        family_parameters,
                    )
                ]
                row: dict[str, Any] = {
                    "product_id": _stable_product_id(
                        family, selected, selected_execution_identity, selected_semantics
                    ),
                    "product_family": catalog_family.leakage_family,
                    "hardware_generation": catalog_family.generation,
                    "category": catalog_family.category,
                    "workload": selected_workload,
                    **catalog_family.features,
                    "benchmark_version": selected[0],
                    "scene": selected[1],
                    "backend": selected[2],
                    "operating_system": selected[3],
                    **selected_execution_identity.as_dict(),
                    "observed_source_value_median": median_source_value,
                    "observed_source_value_mad": mad_source_value,
                    "observed_source_unit": selected_semantics.unit,
                    "observed_source_higher_is_better": selected_semantics.higher_is_better,
                    "observed_source_score_field": selected_semantics.source_field,
                    "benchmark_observation_count": observation_count,
                    "target_score": target_score,
                    "target_transform": target_transform,
                    "is_synthetic": False,
                    "eligible_for_external_claims": dataset_promotable,
                    "dataset_role": role,
                    "matching_method": MATCHING_METHOD_VERSION,
                    "benchmark_hardware_names_json": json.dumps(
                        hardware_names, separators=(",", ":")
                    ),
                    "catalog_product_ids_json": json.dumps(
                        catalog_family.product_ids, separators=(",", ":")
                    ),
                    "source_urls_json": json.dumps(source_urls, separators=(",", ":")),
                    "data_provenance": (
                        "Blender Open Data CC0-1.0 observations joined to BuildCores OpenDB "
                        "ODC-By-1.0 specifications; measured aggregate, not synthetic"
                    ),
                }
                output_rows.append(row)

            candidate_cohorts = [
                {
                    "benchmark_version": summary.cohort[0],
                    "scene": summary.cohort[1],
                    "backend": summary.cohort[2],
                    "operating_system": summary.cohort[3],
                    **summary.execution_identity.as_dict(),
                    "joined_observations": summary.joined_observations,
                    "hardware_identities": summary.hardware_identities,
                    "leakage_groups": summary.leakage_groups,
                    "source_unit": summary.score_semantics.unit,
                    "source_higher_is_better": summary.score_semantics.higher_is_better,
                    "source_score_field": summary.score_semantics.source_field,
                }
                for summary in candidate_summaries
            ]
            blockers: list[str] = []
            if not size_gate_passed:
                blockers.append(
                    f"selected cohort has {leakage_group_count} leakage groups; "
                    f"credible grouped-evaluation minimum is {minimum_credible_products}"
                )
            blockers.extend(effective_source_blockers)
            manifest: dict[str, Any] = {
                "schema_version": "pc-build-recommender.blender-performance-dataset.v3",
                "is_synthetic": False,
                "dataset_role": role,
                "promotion": {
                    "eligible": dataset_promotable,
                    "block_reasons": blockers,
                    "minimum_pilot_products": minimum_pilot_products,
                    "minimum_credible_products": minimum_credible_products,
                },
                "selected_cohort": {
                    "benchmark_version": selected[0],
                    "scene": selected[1],
                    "backend": selected[2],
                    "operating_system": selected[3],
                    **selected_execution_identity.as_dict(),
                    "category": output_rows[0]["category"],
                    "workload": selected_workload,
                    "source_unit": selected_semantics.unit,
                    "source_higher_is_better": selected_semantics.higher_is_better,
                    "source_score_field": selected_semantics.source_field,
                    "joined_observations": selected_summary.joined_observations,
                    "hardware_identities": selected_summary.hardware_identities,
                    "leakage_groups": leakage_group_count,
                },
                "candidate_cohorts": candidate_cohorts,
                "matching": {
                    "method": MATCHING_METHOD_VERSION,
                    "counts": dict(sorted(match_counts.items())),
                    "fuzzy_matching_used": False,
                },
                "features": list(
                    CPU_FEATURE_COLUMNS
                    if output_rows[0]["category"] == "cpu"
                    else GPU_FEATURE_COLUMNS
                ),
                "target": {
                    "column": "target_score",
                    "formula": (
                        f"median({selected_semantics.source_field})"
                        if selected_semantics.higher_is_better
                        else f"{target_scale:g} / median({selected_semantics.source_field})"
                    ),
                    "target_scale": None if selected_semantics.higher_is_better else target_scale,
                    "higher_is_better": True,
                    "unit": (
                        selected_semantics.unit
                        if selected_semantics.higher_is_better
                        else "render_throughput_index"
                    ),
                    "source_value_kind": "observed",
                    "source_unit": selected_semantics.unit,
                    "source_higher_is_better": selected_semantics.higher_is_better,
                    "source_score_field": selected_semantics.source_field,
                    "aggregation": (
                        "median within exact benchmark, execution, score-semantics cohort and "
                        "hardware family"
                    ),
                },
                "row_count": len(output_rows),
                "source_licences": {
                    "blender_open_data": "CC0 1.0",
                    "buildcores_open_db": "ODC-By 1.0; attribution required",
                },
            }
            return output_rows, manifest
        finally:
            connection.close()


def _read_batch_manifest(records_path: Path) -> dict[str, Any]:
    manifest_path = records_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return {}
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def source_promotion_blockers(
    blender_records_path: Path,
    buildcores_records_path: Path,
) -> list[str]:
    """Derive truthful promotion blockers from ingestion manifests."""

    blockers: list[str] = []
    blender_manifest = _read_batch_manifest(blender_records_path)
    blender_stats = blender_manifest.get("statistics")
    if not isinstance(blender_stats, Mapping):
        blockers.append("Blender ingestion manifest is missing source-selection statistics")
    else:
        if blender_stats.get("selection") != "hash_sample":
            blockers.append(
                "Blender observations were not selected with deterministic hash sampling"
            )
        if blender_stats.get("scan_complete") is not True:
            blockers.append("Blender sampling did not scan the complete pinned snapshot")

    buildcores_manifest = _read_batch_manifest(buildcores_records_path)
    buildcores_stats = buildcores_manifest.get("statistics")
    if not isinstance(buildcores_stats, Mapping):
        blockers.append("BuildCores ingestion manifest is missing catalogue coverage statistics")
    else:
        selected = buildcores_stats.get("selected_records")
        available = buildcores_stats.get("available_records")
        if not isinstance(selected, int) or not isinstance(available, int):
            blockers.append("BuildCores catalogue coverage counts are incomplete")
        elif selected != available:
            blockers.append(
                f"BuildCores input is a bounded slice ({selected} of {available} records)"
            )
    return blockers


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (*OUTPUT_METADATA_COLUMNS, *features, *OUTPUT_EVIDENCE_COLUMNS)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender-records", type=Path, default=DEFAULT_BLENDER_RECORDS)
    parser.add_argument("--buildcores-records", type=Path, default=DEFAULT_BUILDCORES_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--category", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--benchmark-version")
    parser.add_argument("--scene")
    parser.add_argument("--backend", choices=("CPU", "CUDA", "OPTIX"))
    parser.add_argument("--operating-system")
    parser.add_argument("--blender-build-hash")
    parser.add_argument("--benchmark-script")
    parser.add_argument("--scene-checksum")
    parser.add_argument("--target-scale", type=float, default=1000.0)
    parser.add_argument("--minimum-pilot-products", type=int, default=15)
    parser.add_argument("--minimum-credible-products", type=int, default=100)
    parser.add_argument(
        "--maximum-record-bytes",
        type=int,
        default=DEFAULT_MAX_RECORD_BYTES,
        help="fail before decoding a JSONL source row above this byte limit",
    )
    parser.add_argument(
        "--max-host-used-gb",
        type=float,
        default=DEFAULT_MAX_HOST_USED_GB,
        help="refuse preparation when the conservative projected host use reaches this cap",
    )
    parser.add_argument(
        "--minimum-free-memory-mb",
        type=float,
        default=DEFAULT_MINIMUM_FREE_MEMORY_MB,
        help="minimum host RAM that must remain after the preparation reservation",
    )
    parser.add_argument(
        "--catalog-memory-expansion-factor",
        type=float,
        default=DEFAULT_CATALOG_MEMORY_EXPANSION_FACTOR,
        help="conservative decoded-catalogue memory multiplier used for admission",
    )
    parser.add_argument(
        "--preparation-runtime-memory-mb",
        type=float,
        default=DEFAULT_PREPARATION_RUNTIME_MEMORY_MB,
        help="fixed interpreter, JSON, and SQLite-runtime allowance used for admission",
    )
    parser.add_argument(
        "--workload",
        help="optional application workload route; defaults to an exact Blender cohort slug",
    )
    parser.add_argument(
        "--promotion-blocker",
        action="append",
        default=[],
        help="repeatable reason that makes this prepared dataset development-only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for source in (args.blender_records, args.buildcores_records):
        if not source.is_file():
            raise FileNotFoundError(source)
    estimated_preparation_memory_mib = estimate_blender_preparation_memory_mib(
        args.buildcores_records,
        maximum_record_bytes=args.maximum_record_bytes,
        catalog_memory_expansion_factor=args.catalog_memory_expansion_factor,
        runtime_memory_mb=args.preparation_runtime_memory_mb,
    )
    host_memory_preflight = require_host_memory_headroom(
        max_used_gib=args.max_host_used_gb,
        estimated_additional_mib=estimated_preparation_memory_mib,
        minimum_free_mib=args.minimum_free_memory_mb,
    )
    pinned_values = (
        args.benchmark_version,
        args.scene,
        args.backend,
        args.operating_system,
    )
    if any(value is not None for value in pinned_values) and not all(
        value is not None for value in pinned_values
    ):
        raise ValueError(
            "benchmark version, scene, backend, and operating system must be pinned together"
        )
    pinned_cohort: Cohort | None = (
        tuple(str(value) for value in pinned_values)  # type: ignore[assignment]
        if all(value is not None for value in pinned_values)
        else None
    )
    pinned_execution_values = (
        args.blender_build_hash,
        args.benchmark_script,
        args.scene_checksum,
    )
    if any(value is not None for value in pinned_execution_values) and not all(
        value is not None for value in pinned_execution_values
    ):
        raise ValueError(
            "Blender build hash, benchmark script, and scene checksum must be pinned together"
        )
    if all(value is not None for value in pinned_execution_values) and pinned_cohort is None:
        raise ValueError(
            "execution identity can only be pinned with benchmark version, scene, backend, and "
            "operating system"
        )
    pinned_execution_identity: BenchmarkExecutionIdentity | None = (
        BenchmarkExecutionIdentity(
            blender_build_hash=str(args.blender_build_hash).strip(),
            benchmark_script=str(args.benchmark_script).strip(),
            scene_checksum=str(args.scene_checksum).strip(),
        )
        if all(value is not None for value in pinned_execution_values)
        else None
    )
    blender_records = _JsonlRecordStream(
        args.blender_records,
        maximum_record_bytes=args.maximum_record_bytes,
    )
    catalog_records = _JsonlRecordStream(
        args.buildcores_records,
        maximum_record_bytes=args.maximum_record_bytes,
    )
    promotion_blockers = [
        *source_promotion_blockers(args.blender_records, args.buildcores_records),
        *(str(reason) for reason in args.promotion_blocker),
    ]
    rows, manifest = prepare_blender_performance(
        blender_records,
        catalog_records,
        category_filter=args.category,
        pinned_cohort=pinned_cohort,
        pinned_execution_identity=pinned_execution_identity,
        target_scale=args.target_scale,
        minimum_pilot_products=args.minimum_pilot_products,
        minimum_credible_products=args.minimum_credible_products,
        source_blockers=promotion_blockers,
        workload=args.workload,
    )
    features = tuple(str(value) for value in manifest["features"])
    output_dir: Path = args.output_dir
    csv_path = output_dir / "blender_performance.csv"
    _write_csv(csv_path, rows, features)
    manifest.update(
        {
            "sources": {
                "blender_records": {
                    "path": portable_path_reference(
                        args.blender_records,
                        workspace_root=REPOSITORY_ROOT,
                    ),
                    "sha256": sha256_file(args.blender_records),
                    "rows": blender_records.records_read,
                },
                "buildcores_records": {
                    "path": portable_path_reference(
                        args.buildcores_records,
                        workspace_root=REPOSITORY_ROOT,
                    ),
                    "sha256": sha256_file(args.buildcores_records),
                    "rows": catalog_records.records_read,
                },
            },
            "output": {
                "csv": portable_path_reference(csv_path, workspace_root=REPOSITORY_ROOT),
                "sha256": sha256_file(csv_path),
                "rows": len(rows),
            },
            "bounded_memory": {
                "design": (
                    "line-bounded JSONL streams with disk-backed matched-observation SQLite "
                    "store"
                ),
                "maximum_record_bytes": args.maximum_record_bytes,
                "sqlite_cache_kib": _MATCH_STORE_CACHE_KIB,
                "catalog_memory_expansion_factor": args.catalog_memory_expansion_factor,
                "runtime_memory_mb": args.preparation_runtime_memory_mb,
                "estimated_additional_mib": round(estimated_preparation_memory_mib, 3),
                "host_memory_preflight": host_memory_preflight.to_dict(),
            },
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from services.api.core_service import CoreRecommendationService
from services.api.durability import DurableStorageError, SqlAlchemyDurableStore
from services.api.models import (
    ComponentKind as ApiComponentCategory,
)
from services.api.models import ReplacementRequest
from services.api.settings import ApiRuntimeSettings
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pc_build_recommender.application import (
    ActiveServingModels,
    ApplicationCatalog,
    EmptyCatalogError,
    ReplacementMode,
    create_application_services,
)
from pc_build_recommender.catalog import (
    BuildComponentRecord,
    CatalogRepository,
    GeneratedBuildRecord,
    create_db_engine,
    init_database,
)
from pc_build_recommender.domain import (
    BenchmarkResult,
    BuildRequestSpec,
    BuildPreset,
    BuildRequirements,
    MasterProduct,
    CaseAttributes,
    CaseSize,
    CommonProductAttributes,
    ComponentKind,
    CoolerAttributes,
    CoolerType,
    CPUAttributes,
    EfficiencyRating,
    ExistingComponent,
    GPUAttributes,
    ListingCondition,
    MemoryAttributes,
    MemoryType,
    ModularType,
    MotherboardAttributes,
    MotherboardFormFactor,
    PowerSupplyAttributes,
    PowerSupplyFormFactor,
    ProductStatus,
    RetailerListing,
    SourceProvenance,
    SourceType,
    StockState,
    StorageAttributes,
    StorageFormFactor,
    StorageInterface,
    WorkloadLabel,
    WorkloadPreference,
)
from pc_build_recommender.optimizer import OptimizationResult, OptimizationStatus
from pc_build_recommender.performance_models import PerformanceModelArtifact
from pc_build_recommender.ranking import (
    HeuristicRanker,
    RankedCandidate,
    RankerArtifactIdentity,
    RankerMetadata,
    RankingCandidate,
    RankingContext,
)
from pc_build_recommender.retrieval import HybridProductRetriever

NOW = datetime(2026, 7, 22, tzinfo=UTC)


class _NonFeasibleOptimizer:
    def __init__(self, status: OptimizationStatus) -> None:
        self.status = status

    def optimize(self, _problem, *, max_solutions=None) -> OptimizationResult:
        del max_solutions
        return OptimizationResult(
            status=self.status,
            solutions=(),
            infeasibility_reasons=(f"forced {self.status.value} solver outcome",),
        )


class _FeatureBooster:
    def __init__(self, feature: str, divisor: float) -> None:
        self.feature = feature
        self.divisor = divisor

    def predict(self, frame, *, num_iteration):  # type: ignore[no-untyped-def]
        del num_iteration
        return [float(frame.iloc[0][self.feature]) / self.divisor]


def _performance_artifact(
    *,
    workload: WorkloadLabel,
    feature: str = "memory_bandwidth_gbps",
    divisor: float = 6.0,
    promotable: bool = True,
) -> PerformanceModelArtifact:
    version = ("a" if promotable else "b") * 64
    return cast(
        PerformanceModelArtifact,
        SimpleNamespace(
            config=SimpleNamespace(
                category="gpu",
                workload=workload.value,
                feature_columns=(feature,),
                target_column=f"{workload.value}_relative_score",
                strict_inference_features=True,
            ),
            booster=_FeatureBooster(feature, divisor),
            model_version=version,
            best_iteration=1,
            allowed_missing_fraction=0.0,
            feature_profiles={},
            precise_predictions_enabled=promotable,
            schema_version="pc-build-recommender.performance-model.v2",
            calibration=SimpleNamespace(
                interval=lambda value: (max(0.0, value - 1.0), value + 1.0)
            ),
            confidence_level="high" if promotable else "low",
            promotable=promotable,
            promotion_block_reasons=(
                () if promotable else ("development fixture is not promotable",)
            ),
        ),
    )


class _PromotedRanker:
    def __init__(self) -> None:
        self.delegate = HeuristicRanker(ranker_version="delegate-v1")
        self._metadata = RankerMetadata(
            ranker_version="ltr-promoted-v1",
            ranking_basis="lightgbm_lambdamart",
            feature_version=self.delegate.metadata.feature_version,
            model_type="lightgbm_lambdamart",
            feature_names=self.delegate.metadata.feature_names,
            created_at_utc=NOW.isoformat(),
            training_label_source="human",
            training_adjudication_complete=True,
            contains_synthetic_labels=False,
            training_judgment_manifest_sha256="c" * 64,
            training_dataset_manifest_sha256="e" * 64,
            training_prelabel_snapshot_sha256="f" * 64,
            training_feature_contract_sha256="1" * 64,
            query_group_split_checksum="d" * 64,
            query_split_membership_verified=True,
            model_sha256="a" * 64,
            promotion_eligible=True,
        )
        self._artifact_identity = RankerArtifactIdentity(
            model_sha256="a" * 64,
            metadata_sha256="b" * 64,
            manifest_sha256="c" * 64,
        )

    @property
    def metadata(self) -> RankerMetadata:
        return self._metadata

    @property
    def artifact_identity(self) -> RankerArtifactIdentity:
        return self._artifact_identity

    def rank_query(
        self,
        context: RankingContext,
        candidates: Sequence[RankingCandidate],
    ) -> list[RankedCandidate]:
        return [
            replace(
                candidate,
                ranker_version=self.metadata.ranker_version,
                ranking_basis=self.metadata.ranking_basis,
            )
            for candidate in self.delegate.rank_query(context, candidates)
        ]


def _product(
    category: ComponentKind,
    index: int,
    attributes: object,
    *,
    product_id: str | None = None,
    canonical_name: str | None = None,
    status: ProductStatus = ProductStatus.ACTIVE,
) -> MasterProduct:
    resolved_id = product_id or f"{category.value}_{index}"
    name = canonical_name or f"Test {category.value} {index}"
    return MasterProduct(
        product_id=resolved_id,
        category=category,
        brand="TestBrand",
        model=f"Model-{category.value}-{index}",
        manufacturer_part_number=f"MPN-{category.value}-{index}",
        canonical_name=name,
        status=status,
        release_date=date(2025, 1, index + 1),
        common_attributes=CommonProductAttributes(
            warranty_years=5,
            tags=["quiet", "desktop"],
        ),
        category_attributes=attributes,
        provenance=[
            SourceProvenance(
                provenance_id=f"src_{resolved_id}",
                product_id=resolved_id,
                source_name="Test manufacturer",
                source_url=f"https://manufacturer.invalid/{resolved_id}",
                source_type=SourceType.MANUFACTURER,
                retrieved_at=NOW,
                raw_content_hash=(f"{index + 1:064x}"[-64:]),
                parser_version="test-v1",
                licence_or_access_note="Test fixture",
                last_verified_at=NOW,
            )
        ],
        created_at=NOW,
        updated_at=NOW,
    )


def _attributes(category: ComponentKind, index: int) -> object:
    if category == ComponentKind.CPU:
        return CPUAttributes(
            socket="AM5",
            architecture="Zen 4",
            generation="Zen 4",
            core_count=8 + index,
            thread_count=16 + 2 * index,
            base_clock_ghz=4.0,
            boost_clock_ghz=5.0,
            tdp_watts=105,
            peak_power_watts=120 + 5 * index,
        )
    if category == ComponentKind.GPU:
        return GPUAttributes(
            architecture="Test GPU",
            vram_gb=16,
            memory_bandwidth_gbps=500 + 20 * index,
            length_mm=300 + index,
            height_mm=120,
            slot_width=2.5,
            board_power_watts=240 + 5 * index,
            recommended_psu_watts=650,
            power_connectors={"pcie_8_pin": 1},
        )
    if category == ComponentKind.MOTHERBOARD:
        return MotherboardAttributes(
            socket="AM5",
            chipset="B650",
            supported_cpu_generations=["Zen 4"],
            form_factor=MotherboardFormFactor.ATX,
            memory_type=MemoryType.DDR5,
            maximum_memory_gb=192,
            memory_slots=4,
            pcie_slots=3,
            m2_slots=3,
            sata_ports=4,
            wifi_support=True,
            bios_version="2.0",
        )
    if category == ComponentKind.MEMORY:
        return MemoryAttributes(
            memory_type=MemoryType.DDR5,
            capacity_gb=32 + 16 * index,
            module_count=2,
            speed_mt_s=6000,
            cas_latency=30,
            voltage=1.35,
        )
    if category == ComponentKind.STORAGE:
        return StorageAttributes(
            capacity_gb=2000,
            interface=StorageInterface.NVME_PCIE,
            form_factor=StorageFormFactor.M2_2280,
            sequential_read_mbps=7000 + 100 * index,
            sequential_write_mbps=6000,
        )
    if category == ComponentKind.POWER_SUPPLY:
        return PowerSupplyAttributes(
            wattage=850,
            efficiency_rating=EfficiencyRating.GOLD,
            form_factor=PowerSupplyFormFactor.ATX,
            modular_type=ModularType.FULLY_MODULAR,
            pcie_connectors={"pcie_8_pin": 4},
            eps_connectors=2,
            atx_version="3.0",
            warranty_years=10,
        )
    if category == ComponentKind.COOLER:
        return CoolerAttributes(
            cooler_type=CoolerType.AIR,
            supported_sockets=["AM5"],
            height_mm=155,
            fan_count=2,
            estimated_cooling_capacity_watts=220,
        )
    if category == ComponentKind.CASE:
        return CaseAttributes(
            case_size=CaseSize.MID_TOWER,
            supported_motherboard_sizes=[MotherboardFormFactor.ATX],
            maximum_gpu_length_mm=380,
            maximum_gpu_slot_width=4,
            maximum_cooler_height_mm=170,
            supported_psu_sizes=[PowerSupplyFormFactor.ATX],
            radiator_support_mm=[240, 360],
            drive_bays=4,
            included_fans=3,
        )
    raise AssertionError(f"unsupported category: {category}")


def _price(category: ComponentKind, index: int) -> Decimal:
    bases = {
        ComponentKind.CPU: 300,
        ComponentKind.GPU: 650,
        ComponentKind.MOTHERBOARD: 220,
        ComponentKind.MEMORY: 130,
        ComponentKind.STORAGE: 140,
        ComponentKind.POWER_SUPPLY: 150,
        ComponentKind.COOLER: 70,
        ComponentKind.CASE: 130,
    }
    return Decimal(bases[category] + 15 * index)


def _seed_repository(session: Session) -> CatalogRepository:
    repository = CatalogRepository(session)
    for category in ComponentKind:
        for index in range(3):
            product = _product(category, index, _attributes(category, index))
            repository.upsert_product(product)
            repository.upsert_listing(
                RetailerListing(
                    listing_id=f"listing_{product.product_id}",
                    product_id=product.product_id,
                    retailer="Test retailer",
                    source_listing_id=f"offer-{product.product_id}",
                    title=product.canonical_name,
                    currency="SGD",
                    base_price=_price(category, index),
                    shipping_price=Decimal("0"),
                    stock_status=StockState.IN_STOCK,
                    listing_url=f"https://retailer.invalid/{product.product_id}",
                    first_seen_at=NOW,
                    last_seen_at=NOW,
                )
            )

            if category in (ComponentKind.CPU, ComponentKind.GPU):
                workload = (
                    WorkloadLabel.SOFTWARE_DEVELOPMENT
                    if category == ComponentKind.CPU
                    else WorkloadLabel.GAMING_1440P
                )
                repository.upsert_benchmark(
                    BenchmarkResult(
                        benchmark_id=f"bench_{product.product_id}",
                        product_id=product.product_id,
                        workload=workload,
                        benchmark_name=f"Test {category.value} suite",
                        benchmark_version="1",
                        score=100 + 10 * index,
                        unit="points",
                        source_url="https://benchmarks.invalid/test",
                        observed_at=NOW,
                    )
                )

    bad_case = _product(
        ComponentKind.CASE,
        9,
        CaseAttributes(
            case_size=CaseSize.MID_TOWER,
            supported_motherboard_sizes=[MotherboardFormFactor.ATX],
            maximum_gpu_length_mm=200,
            maximum_gpu_slot_width=2,
            maximum_cooler_height_mm=170,
            supported_psu_sizes=[PowerSupplyFormFactor.ATX],
            included_fans=2,
        ),
        product_id="case_incompatible",
        canonical_name="Incompatible tiny case",
    )
    repository.upsert_product(bad_case)
    repository.upsert_listing(
        RetailerListing(
            listing_id="listing_case_incompatible",
            product_id=bad_case.product_id,
            retailer="Test retailer",
            source_listing_id="offer-case-incompatible",
            title=bad_case.canonical_name,
            base_price=Decimal("60"),
            stock_status=StockState.IN_STOCK,
            listing_url="https://retailer.invalid/case-incompatible",
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
    )

    used_gpu = _product(
        ComponentKind.GPU,
        8,
        _attributes(ComponentKind.GPU, 0),
        product_id="gpu_used_only",
        canonical_name="Used-only GPU",
    )
    repository.upsert_product(used_gpu)
    repository.upsert_listing(
        RetailerListing(
            listing_id="listing_gpu_used_only",
            product_id=used_gpu.product_id,
            retailer="Test retailer",
            source_listing_id="offer-gpu-used-only",
            title=used_gpu.canonical_name,
            condition=ListingCondition.USED,
            base_price=Decimal("10"),
            stock_status=StockState.IN_STOCK,
            listing_url="https://retailer.invalid/gpu-used-only",
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
    )

    retained_cpu = _product(
        ComponentKind.CPU,
        7,
        _attributes(ComponentKind.CPU, 0),
        product_id="cpu_discontinued_retained",
        canonical_name="Discontinued retained CPU",
        status=ProductStatus.DISCONTINUED,
    )
    repository.upsert_product(retained_cpu)
    return repository


@pytest.fixture
def application():
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    with Session(engine) as session:
        repository = _seed_repository(session)
        yield create_application_services(
            repository,
            data_version="test-catalog-v1",
            random_seed=7,
        )


def _request(
    *,
    existing_products: list[ExistingComponent] | None = None,
    minimum_gpu_vram_gb: int = 16,
    profiles: list[BuildPreset] | None = None,
) -> BuildRequestSpec:
    return BuildRequestSpec(
        budget_sgd=Decimal("2500"),
        workloads=[
            WorkloadPreference(name=WorkloadLabel.GAMING_1440P, weight=0.6),
            WorkloadPreference(name=WorkloadLabel.SOFTWARE_DEVELOPMENT, weight=0.4),
        ],
        existing_products=existing_products or [],
        requirements=BuildRequirements(
            minimum_gpu_vram_gb=minimum_gpu_vram_gb,
            minimum_memory_gb=32,
            storage_gb=2000,
            required_memory_type=MemoryType.DDR5,
            wifi_required=True,
            case_size=CaseSize.MID_TOWER,
        ),
        raw_query="quiet 1440p development PC",
        requested_profiles=profiles
        or [
            BuildPreset.BEST_OVERALL,
            BuildPreset.BEST_VALUE,
            BuildPreset.HIGHEST_PERFORMANCE,
        ],
    )


def test_development_composition_labels_baselines_truthfully(application) -> None:
    assert application.versions.retrieval_model == "bm25+stable-hash-vector+rrf-development-v1"
    assert application.versions.ranking_model == "heuristic-v1"
    assert application.versions.performance_model == "observed-only-v1"


def test_production_composition_rejects_missing_promoted_evidence() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    try:
        with Session(engine) as session:
            repository = _seed_repository(session)
            with pytest.raises(RuntimeError, match="promoted-serving evidence"):
                create_application_services(
                    repository,
                    data_version="test-catalog-v1",
                    require_promoted_models=True,
                )
    finally:
        engine.dispose()


def test_runtime_identity_cannot_be_overridden_by_a_caller_string() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    try:
        with Session(engine) as session:
            repository = _seed_repository(session)
            with pytest.raises(ValueError, match="runtime retriever"):
                create_application_services(
                    repository,
                    data_version="test-catalog-v1",
                    retrieval_model_version="postgres-hybrid-v1",
                )
            with pytest.raises(RuntimeError, match="runtime inference provider"):
                create_application_services(
                    repository,
                    data_version="test-catalog-v1",
                    performance_model_version="pretend-promoted-v1",
                )
    finally:
        engine.dispose()


def test_artifact_prediction_reaches_optimizer_presenter_and_api() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    try:
        with Session(engine) as session:
            repository = _seed_repository(session)
            artifact = _performance_artifact(workload=WorkloadLabel.LOCAL_AI)
            services = create_application_services(
                repository,
                data_version="test-catalog-v1",
                performance_artifacts=(artifact,),
                random_seed=7,
            )
            request = _request(profiles=[BuildPreset.HIGHEST_PERFORMANCE]).model_copy(
                update={"workloads": [WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=1.0)]}
            )
            response = services.generate_builds.generate(
                request,
                request_id="req_predicted_performance",
            )
            build = response.builds[0]
            gpu = next(
                component
                for component in build.components
                if component.category is ComponentKind.GPU
            )
            signal = gpu.performance_signals[0]

            assert gpu.product_id == "gpu_2"
            assert signal.workload is WorkloadLabel.LOCAL_AI
            assert signal.basis == "predicted"
            assert signal.model_version == artifact.model_version
            assert signal.score == pytest.approx(signal.relative_score)
            assert build.workload_scores[WorkloadLabel.LOCAL_AI] == pytest.approx(signal.score)
            assert response.performance_model.startswith("promotion-eligible[gpu/local_ai=")

            adapter = object.__new__(CoreRecommendationService)
            adapter.services = services
            adapter._generated_at_by_request = {}
            api_response = adapter._generation_response(response, request, generated_at=NOW)
            api_gpu = next(
                component
                for component in api_response.builds[0].components
                if component.category is ApiComponentCategory.GPU
            )
            api_signal = api_gpu.performance_signals[0]
            assert api_signal.basis == "predicted"
            assert api_signal.decision == "precise_model_prediction"
            assert api_signal.model_version == artifact.model_version
            assert api_signal.value == pytest.approx(signal.score)
    finally:
        engine.dispose()


def test_observed_benchmark_precedes_artifact_for_the_same_route() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    try:
        with Session(engine) as session:
            repository = _seed_repository(session)
            artifact = _performance_artifact(workload=WorkloadLabel.GAMING_1440P)
            services = create_application_services(
                repository,
                data_version="test-catalog-v1",
                performance_artifacts=(artifact,),
                random_seed=7,
            )
            request = _request(profiles=[BuildPreset.HIGHEST_PERFORMANCE]).model_copy(
                update={
                    "workloads": [WorkloadPreference(name=WorkloadLabel.GAMING_1440P, weight=1.0)]
                }
            )
            response = services.generate_builds.generate(
                request,
                request_id="req_observed_first",
            )
            gpu = next(
                component
                for component in response.builds[0].components
                if component.category is ComponentKind.GPU
            )
            signal = gpu.performance_signals[0]

            assert signal.basis == "observed"
            assert signal.decision == "observed_benchmark"
            assert signal.model_version is None
            assert signal.supporting_benchmark_ids
            assert signal.supporting_sources == ["https://benchmarks.invalid/test"]
    finally:
        engine.dispose()


def test_unpromoted_artifact_requires_explicit_development_opt_in() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    try:
        with Session(engine) as session:
            repository = _seed_repository(session)
            artifact = _performance_artifact(
                workload=WorkloadLabel.LOCAL_AI,
                promotable=False,
            )
            with pytest.raises(RuntimeError, match="explicit development opt-in"):
                create_application_services(
                    repository,
                    data_version="test-catalog-v1",
                    performance_artifacts=(artifact,),
                )

            services = create_application_services(
                repository,
                data_version="test-catalog-v1",
                performance_artifacts=(artifact,),
                allow_unpromoted_performance_models=True,
                random_seed=7,
            )
            request = _request(profiles=[BuildPreset.BEST_OVERALL]).model_copy(
                update={"workloads": [WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=1.0)]}
            )
            response = services.generate_builds.generate(
                request,
                request_id="req_relative_performance",
            )
            gpu = next(
                component
                for component in response.builds[0].components
                if component.category is ComponentKind.GPU
            )
            signal = gpu.performance_signals[0]
            assert signal.basis == "relative"
            assert signal.decision == "model_not_promotion_eligible"
            assert signal.score is None
            assert signal.model_version == artifact.model_version
            assert WorkloadLabel.LOCAL_AI not in response.builds[0].workload_scores
            assert response.performance_model.startswith("development-relative[")
    finally:
        engine.dispose()


def test_production_composition_requires_exact_promoted_performance_routes() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    try:
        with Session(engine) as session:
            repository = _seed_repository(session)
            catalog = ApplicationCatalog.from_repository(
                repository,
                data_version="test-catalog-v1",
            )
            retriever = HybridProductRetriever(catalog.documents)
            ranker = _PromotedRanker()
            artifact = _performance_artifact(workload=WorkloadLabel.LOCAL_AI)
            active = ActiveServingModels(
                catalog_data_version="test-catalog-v1",
                retrieval_model=retriever.retrieval_model_version,
                ranking_model=ranker.metadata.ranker_version,
                performance_models={"gpu/local_ai": artifact.model_version},
                embedding_index_version="test-index-v1",
                retrieval_report_sha256="e" * 64,
                ranker_promotion_decision_sha256="f" * 64,
                ranker_model_sha256="a" * 64,
                ranker_metadata_sha256="b" * 64,
                ranker_manifest_sha256="c" * 64,
            )
            services = create_application_services(
                repository,
                data_version="test-catalog-v1",
                retriever=retriever,
                ranker=ranker,
                performance_artifacts=(artifact,),
                promoted_serving_models=active,
                require_promoted_models=True,
            )
            assert services.versions.performance_model == active.performance_model_label

            mismatched = ActiveServingModels(
                catalog_data_version="test-catalog-v1",
                retrieval_model=retriever.retrieval_model_version,
                ranking_model=ranker.metadata.ranker_version,
                performance_models={"gpu/local_ai": "0" * 64},
                embedding_index_version="test-index-v1",
                retrieval_report_sha256="e" * 64,
                ranker_promotion_decision_sha256="f" * 64,
                ranker_model_sha256="a" * 64,
                ranker_metadata_sha256="b" * 64,
                ranker_manifest_sha256="c" * 64,
            )
            with pytest.raises(RuntimeError, match="routes/versions"):
                create_application_services(
                    repository,
                    data_version="test-catalog-v1",
                    retriever=retriever,
                    ranker=ranker,
                    performance_artifacts=(artifact,),
                    promoted_serving_models=mismatched,
                    require_promoted_models=True,
                )

            mismatched_ranker_artifact = replace(
                active,
                ranker_model_sha256="9" * 64,
            )
            with pytest.raises(RuntimeError, match="artifact identity"):
                create_application_services(
                    repository,
                    data_version="test-catalog-v1",
                    retriever=retriever,
                    ranker=ranker,
                    performance_artifacts=(artifact,),
                    promoted_serving_models=mismatched_ranker_artifact,
                    require_promoted_models=True,
                )
    finally:
        engine.dispose()


def test_discontinued_product_is_retained_but_not_purchasable(application) -> None:
    product_id = "cpu_discontinued_retained"
    assert application.catalog.get(product_id) is not None
    assert product_id not in {document.product_id for document in application.catalog.documents}

    response = application.generate_builds.generate(
        _request(
            existing_products=[
                ExistingComponent(
                    category=ComponentKind.CPU,
                    product_id=product_id,
                )
            ],
            profiles=[BuildPreset.BEST_OVERALL],
        ),
        request_id="req_retained_discontinued",
    )

    assert not any(
        "not in the canonical catalogue" in reason for reason in response.infeasibility_reasons
    )
    if response.builds:
        selected_cpu = next(
            component
            for component in response.builds[0].components
            if component.category is ComponentKind.CPU
        )
        assert selected_cpu.product_id == product_id


def test_end_to_end_generation_is_versioned_valid_diverse_and_refresh_safe(
    application,
) -> None:
    request = _request()
    response = application.generate_builds.generate(request, request_id="req_integration")

    assert response.request_id == "req_integration"
    assert response.data_version == "test-catalog-v1"
    assert response.ranking_model == "heuristic-v1"
    assert response.retrieval_model == application.versions.retrieval_model
    assert response.performance_model == "observed-only-v1"
    assert response.rule_version == "compat_v2"
    assert response.optimizer_status in {
        OptimizationStatus.OPTIMAL,
        OptimizationStatus.FEASIBLE,
    }
    assert response.optimizer_version == "cp-sat-v1"
    assert response.optimizer_ran is True
    assert len(response.optimizer_profile_statuses) == 3
    assert len(response.builds) == 3
    assert application.generate_builds.get_response("req_integration") == response
    assert application.generate_builds.generate(request, request_id="req_integration") == response

    for build in response.builds:
        assert len(build.components) == 8
        assert {component.category for component in build.components} == set(ComponentKind)
        assert build.total_price_sgd <= request.budget_sgd
        assert build.compatibility_status.value in {"pass", "warning"}
        assert all(
            check.status.value not in {"fail", "unknown"} for check in build.compatibility_checks
        )
        assert "case_incompatible" not in {item.product_id for item in build.components}
        assert application.generate_builds.get_build(build.build_id) == build

    for index, build in enumerate(response.builds):
        current = {item.category: item.product_id for item in build.components}
        for prior in response.builds[:index]:
            previous = {item.category: item.product_id for item in prior.components}
            assert sum(current[key] != previous[key] for key in ComponentKind) >= 2


def test_api_presenter_attributes_observed_scores_to_benchmark_evidence(application) -> None:
    request = _request(profiles=[BuildPreset.BEST_OVERALL])
    response = application.generate_builds.generate(
        request,
        request_id="req_benchmark_evidence_contract",
    )
    adapter = object.__new__(CoreRecommendationService)
    adapter.services = application
    adapter._generated_at_by_request = {}

    presented = adapter._generation_response(response, request)
    observed_signals = [
        signal
        for component in presented.builds[0].components
        for signal in component.performance_signals
        if signal.basis == "observed"
    ]
    product_pages = {
        provenance.source_url
        for component in response.builds[0].components
        for provenance in application.catalog.require(component.product_id).product.provenance
    }

    assert observed_signals
    assert all(signal.supporting_benchmark_ids for signal in observed_signals)
    assert all(signal.benchmark_evidence for signal in observed_signals)
    assert {source.url for signal in observed_signals for source in signal.sources} == {
        "https://benchmarks.invalid/test"
    }
    assert not (
        {source.url for signal in observed_signals for source in signal.sources} & product_pages
    )


def test_search_uses_wire_psu_alias_and_complete_build_compatibility(application) -> None:
    psu_results = application.search_products.search(
        "850 watt gold",
        category="psu",
        top_k=5,
    )
    assert psu_results
    assert all(item.product.category == ComponentKind.POWER_SUPPLY for item in psu_results)

    response = application.generate_builds.generate(
        _request(profiles=[BuildPreset.BEST_OVERALL]),
        request_id="req_search_compat",
    )
    build = response.builds[0]
    case_outcome = application.search_products.search_with_outcome(
        "case",
        category=ComponentKind.CASE,
        top_k=10,
        compatible_with_build_id=build.build_id,
    )
    cases = case_outcome.results
    assert cases
    assert case_outcome.retrieved_candidates == (
        len(cases) + case_outcome.filtered_incompatible + case_outcome.filtered_unknown
    )
    assert case_outcome.filtered_incompatible >= 1
    assert "case_incompatible" not in {item.product_id for item in cases}
    assert all(
        item.compatibility_status is not None
        and item.compatibility_status.value in {"pass", "warning"}
        for item in cases
    )


def test_locked_existing_component_is_retained_and_excluded_from_budget(application) -> None:
    request = _request(
        existing_products=[ExistingComponent(category=ComponentKind.GPU, product_id="gpu_2")],
        profiles=[BuildPreset.BEST_OVERALL],
    )
    response = application.generate_builds.generate(request, request_id="req_locked")
    assert len(response.builds) == 1
    gpu = next(
        item for item in response.builds[0].components if item.category == ComponentKind.GPU
    )
    assert gpu.product_id == "gpu_2"
    assert gpu.price_sgd == 0
    assert "Retained existing" in gpu.selection_reason


def test_used_only_product_is_not_acquired_but_can_be_retained(application) -> None:
    search_results = application.search_products.search(
        "Used-only GPU",
        category=ComponentKind.GPU,
        top_k=10,
    )
    assert "gpu_used_only" not in {item.product_id for item in search_results}

    ordinary = application.generate_builds.generate(
        _request(profiles=[BuildPreset.BEST_OVERALL]),
        request_id="req_no_used_acquisition",
    ).builds[0]
    assert "gpu_used_only" not in {item.product_id for item in ordinary.components}

    retained = application.generate_builds.generate(
        _request(
            existing_products=[
                ExistingComponent(
                    category=ComponentKind.GPU,
                    product_id="gpu_used_only",
                )
            ],
            profiles=[BuildPreset.BEST_OVERALL],
        ),
        request_id="req_used_retained",
    ).builds[0]
    retained_gpu = next(
        item for item in retained.components if item.category == ComponentKind.GPU
    )
    assert retained_gpu.product_id == "gpu_used_only"
    assert retained_gpu.listing_id is None
    assert retained_gpu.price_sgd == 0


def test_owned_component_can_be_explicitly_included_in_budget(application) -> None:
    request = _request(
        existing_products=[ExistingComponent(category=ComponentKind.GPU, product_id="gpu_2")],
        profiles=[BuildPreset.BEST_OVERALL],
    )
    response = application.generate_builds.generate(
        request,
        request_id="req_locked_counted",
        included_existing_product_ids=frozenset({"gpu_2"}),
    )
    build = response.builds[0]
    gpu = next(item for item in build.components if item.category == ComponentKind.GPU)
    assert gpu.product_id == "gpu_2"
    assert gpu.price_sgd == _price(ComponentKind.GPU, 2)
    assert build.total_price_sgd == sum(item.price_sgd for item in build.components)


def test_replacing_owned_component_turns_replacement_into_acquisition_cost(application) -> None:
    initial = application.generate_builds.generate(
        _request(
            existing_products=[
                ExistingComponent(category=ComponentKind.GPU, product_id="gpu_2")
            ],
            profiles=[BuildPreset.BEST_OVERALL],
        ),
        request_id="req_owned_replace_initial",
    ).builds[0]
    assert (
        next(
            item.price_sgd for item in initial.components if item.category == ComponentKind.GPU
        )
        == 0
    )

    replaced = application.replace_component.replace(
        initial.build_id,
        category=ComponentKind.GPU,
        replacement_product_id="gpu_0",
        request_id="req_owned_replaced",
    ).builds[0]
    replacement = next(
        item for item in replaced.components if item.category == ComponentKind.GPU
    )
    assert replacement.product_id == "gpu_0"
    assert replacement.price_sgd == _price(ComponentKind.GPU, 0)
    assert replaced.total_price_sgd == sum(item.price_sgd for item in replaced.components)


def test_infeasible_request_returns_reasons_and_suggested_relaxations(application) -> None:
    response = application.generate_builds.generate(
        _request(
            minimum_gpu_vram_gb=64,
            profiles=[BuildPreset.BEST_OVERALL],
        ),
        request_id="req_infeasible",
    )
    assert response.builds == []
    assert response.optimizer_status == OptimizationStatus.INFEASIBLE
    assert response.optimizer_ran is False
    assert response.optimizer_profile_statuses == []
    assert any("GPU" in reason or "gpu" in reason for reason in response.infeasibility_reasons)
    assert any(
        reason.startswith("Suggested relaxation:") for reason in response.infeasibility_reasons
    )


def test_replacement_modes_preserve_cost_and_lock_semantics(application) -> None:
    initial = application.generate_builds.generate(
        _request(profiles=[BuildPreset.BEST_OVERALL]),
        request_id="req_replace_initial",
    ).builds[0]
    current_gpu = next(
        item.product_id for item in initial.components if item.category == ComponentKind.GPU
    )
    replacement_gpu = next(item for item in ("gpu_0", "gpu_1", "gpu_2") if item != current_gpu)

    locked_response = application.replace_component.replace(
        initial.build_id,
        category=ComponentKind.GPU,
        replacement_product_id=replacement_gpu,
        mode=ReplacementMode.LOCK_OTHER_COMPONENTS,
        request_id="req_replace_locked",
    )
    assert len(locked_response.builds) == 1
    locked_build = locked_response.builds[0]
    old_by_category = {item.category: item.product_id for item in initial.components}
    new_by_category = {item.category: item.product_id for item in locked_build.components}
    assert new_by_category[ComponentKind.GPU] == replacement_gpu
    assert all(
        new_by_category[category] == old_by_category[category]
        for category in ComponentKind
        if category != ComponentKind.GPU
    )

    reoptimized = application.replace_component.replace(
        initial.build_id,
        category=ComponentKind.GPU,
        replacement_product_id=replacement_gpu,
        mode=ReplacementMode.REOPTIMIZE_UNLOCKED,
        request_id="req_replace_reoptimized",
    )
    assert len(reoptimized.builds) == 1
    assert (
        next(
            item.product_id
            for item in reoptimized.builds[0].components
            if item.category == ComponentKind.GPU
        )
        == replacement_gpu
    )
    assert all(
        check.status.value not in {"fail", "unknown"}
        for check in reoptimized.builds[0].compatibility_checks
    )


def test_populated_replacement_survives_restart_with_original_ownership_and_timestamp(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'replacement-restart.db').as_posix()}"
    first_engine = create_db_engine(database_url)
    init_database(first_engine)
    with Session(first_engine) as seed_session:
        _seed_repository(seed_session)
        seed_session.commit()

    first_store = SqlAlchemyDurableStore(first_engine)
    with Session(first_engine) as catalog_session:
        first_services = create_application_services(
            CatalogRepository(catalog_session),
            data_version="test-catalog-v1",
            result_store=first_store,
            random_seed=7,
        )
        initial = first_services.generate_builds.generate(
            _request(
                existing_products=[
                    ExistingComponent(category=ComponentKind.GPU, product_id="gpu_2")
                ],
                profiles=[BuildPreset.BEST_OVERALL],
            ),
            request_id="req_restart_owned_initial",
        ).builds[0]
        current_cpu = next(
            item.product_id for item in initial.components if item.category == ComponentKind.CPU
        )
        replacement_cpu = next(item for item in ("cpu_0", "cpu_1", "cpu_2") if item != current_cpu)
        first_replacement = first_services.replace_component.replace(
            initial.build_id,
            category=ComponentKind.CPU,
            replacement_product_id=replacement_cpu,
            request_id="req_restart_owned_replaced",
        )
        first_replaced_build = first_replacement.builds[0]
        first_stored = first_store.require_generation(first_replacement.request_id)

    assert first_stored.owned_product_ids == frozenset({"gpu_2"})
    assert first_stored.no_cost_product_ids == frozenset({"gpu_2"})
    assert len(first_stored.request.existing_products) == len(ComponentKind)
    first_engine.dispose()

    restarted_engine = create_db_engine(database_url)
    restarted_store = SqlAlchemyDurableStore(restarted_engine)
    with Session(restarted_engine) as catalog_session:
        restarted_services = create_application_services(
            CatalogRepository(catalog_session),
            data_version="test-catalog-v1",
            result_store=restarted_store,
            random_seed=7,
        )
        adapter = object.__new__(CoreRecommendationService)
        adapter.services = restarted_services
        adapter.settings = ApiRuntimeSettings(environment="test")
        adapter._durable_store = restarted_store
        adapter._mutation_lock = asyncio.Lock()
        adapter._interactions = []
        adapter._generated_at_by_request = {}

        restarted_summary = asyncio.run(adapter.get_build(first_replaced_build.build_id))
        repeated_summary = asyncio.run(adapter.get_build(first_replaced_build.build_id))
        owned = {item.product_id for item in restarted_summary.components if item.already_owned}
        assert owned == {"gpu_2"}
        assert restarted_summary.generated_at == first_stored.stored_at
        assert repeated_summary.generated_at == first_stored.stored_at

        current_case = next(
            item.product_id
            for item in first_replaced_build.components
            if item.category == ComponentKind.CASE
        )
        replacement_case = next(
            item for item in ("case_0", "case_1", "case_2") if item != current_case
        )
        immediate = asyncio.run(
            adapter.replace_component(
                first_replaced_build.build_id,
                ReplacementRequest(
                    category=ApiComponentCategory.CASE,
                    replacement_product_id=replacement_case,
                ),
            )
        )
        replacement_stored = restarted_store.generation_for_build(immediate.build.build_id)
        persisted_summary = asyncio.run(adapter.get_build(immediate.build.build_id))

        assert immediate.build.generated_at == replacement_stored.stored_at
        assert persisted_summary.generated_at == replacement_stored.stored_at
        assert replacement_stored.owned_product_ids == frozenset({"gpu_2"})
        assert replacement_stored.no_cost_product_ids == frozenset({"gpu_2"})
        assert {item.product_id for item in immediate.build.components if item.already_owned} == {
            "gpu_2"
        }

        generated_count = catalog_session.scalar(select(func.count(GeneratedBuildRecord.build_id)))
        component_count = catalog_session.scalar(select(func.count(BuildComponentRecord.build_id)))
        assert generated_count == 3
        assert component_count == 3 * len(ComponentKind)
    restarted_engine.dispose()


def test_non_unique_durable_integrity_failure_is_storage_unavailable(
    application,
    tmp_path: Path,
) -> None:
    request = _request(profiles=[BuildPreset.BEST_OVERALL])
    response = application.generate_builds.generate(
        request,
        request_id="req_missing_catalog_fk",
    )
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'missing-catalog.db').as_posix()}")
    init_database(engine)
    store = SqlAlchemyDurableStore(engine)

    with pytest.raises(DurableStorageError, match="database integrity"):
        store.save(request, response)

    assert store.get_generation(response.request_id) is None
    engine.dispose()


@pytest.mark.parametrize(
    "status",
    [OptimizationStatus.UNKNOWN, OptimizationStatus.MODEL_INVALID],
)
def test_non_feasible_optimizer_status_is_preserved_in_response_and_store(
    application,
    status: OptimizationStatus,
) -> None:
    application.generate_builds.optimizer = _NonFeasibleOptimizer(status)
    request_id = f"req_status_{status.value}"
    response = application.generate_builds.generate(
        _request(profiles=[BuildPreset.BEST_OVERALL]),
        request_id=request_id,
    )

    assert response.builds == []
    assert response.optimizer_status == status
    assert response.optimizer_ran is True
    assert response.model_dump(mode="json")["optimizer_status"] == status.value
    stored = application.results.require_generation(request_id)
    assert stored.response.optimizer_status == status


def test_replacement_preserves_non_feasible_optimizer_status(application) -> None:
    initial = application.generate_builds.generate(
        _request(profiles=[BuildPreset.BEST_OVERALL]),
        request_id="req_status_replace_initial",
    ).builds[0]
    current_gpu = next(
        item.product_id for item in initial.components if item.category == ComponentKind.GPU
    )
    replacement_gpu = next(item for item in ("gpu_0", "gpu_1", "gpu_2") if item != current_gpu)
    application.generate_builds.optimizer = _NonFeasibleOptimizer(OptimizationStatus.MODEL_INVALID)

    response = application.replace_component.replace(
        initial.build_id,
        category=ComponentKind.GPU,
        replacement_product_id=replacement_gpu,
        request_id="req_status_replace_invalid",
    )
    assert response.builds == []
    assert response.optimizer_status == OptimizationStatus.MODEL_INVALID
    assert response.optimizer_ran is True


def test_empty_catalog_factory_fails_instead_of_inventing_demo_products() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_database(engine)
    with (
        Session(engine) as session,
        pytest.raises(EmptyCatalogError, match="catalogue is empty"),
    ):
        create_application_services(CatalogRepository(session))

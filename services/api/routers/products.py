"""Product retrieval and evidence endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from services.api.dependencies import (
    ApplicationDependency,
    ImpressionSignerDependency,
    SettingsDependency,
)
from services.api.impressions import prepare_impression_response
from services.api.metrics import DOMAIN_METRICS
from services.api.models import (
    ProductBenchmarksResponse,
    ProductDetail,
    ProductPricesResponse,
    ProductReviewsResponse,
    ProductSearchRequest,
    ProductSearchResponse,
)
from services.api.routers.openapi import (
    NOT_FOUND_ERROR,
    PAYLOAD_TOO_LARGE_ERROR,
    VALIDATION_ERROR,
)
from services.api.routers.response_contracts import validate_service_response

router = APIRouter(prefix="/v1/products", tags=["products"])


@router.post(
    "/search",
    response_model=ProductSearchResponse,
    responses={**NOT_FOUND_ERROR, **VALIDATION_ERROR, **PAYLOAD_TOO_LARGE_ERROR},
)
async def search_products(
    request: ProductSearchRequest,
    http_request: Request,
    http_response: Response,
    application: ApplicationDependency,
    signer: ImpressionSignerDependency,
    settings: SettingsDependency,
) -> ProductSearchResponse:
    response = validate_service_response(
        await application.search_products(request), ProductSearchResponse
    )
    page = response.pagination.page if response.pagination is not None else 1
    page_size = (
        response.pagination.page_size
        if response.pagination is not None
        else request.effective_page_size
    )
    first_rank = (page - 1) * page_size + 1
    actor_id = prepare_impression_response(
        http_request,
        http_response,
        signer=signer,
        secure_cookie=not settings.is_development_environment,
    )
    response = response.model_copy(
        update={
            "products": [
                product.model_copy(
                    update={
                        "impression_token": signer.issue(
                            actor_id=actor_id,
                            query_id=response.query_id,
                            kind="product_search_result",
                            rank_position=first_rank + offset,
                            product_id=product.product_id,
                            model_version=response.retrieval_model,
                            data_version=response.data_version,
                            rule_version=settings.compatibility_rule_version,
                        )
                    },
                    deep=True,
                )
                for offset, product in enumerate(response.products)
            ]
        },
        deep=True,
    )
    DOMAIN_METRICS.record_product_search(
        result_count=len(response.products),
        ranked_candidates=response.total,
        retrieved_candidates=response.retrieved_candidates,
        filtered_category=response.filtered_category,
        filtered_brand=response.filtered_brand,
        filtered_incompatible=response.filtered_incompatible,
        filtered_unknown=response.filtered_unknown,
    )
    return response


@router.get("/{product_id}", response_model=ProductDetail, responses=NOT_FOUND_ERROR)
async def get_product(product_id: str, application: ApplicationDependency) -> ProductDetail:
    return await application.get_product(product_id)


@router.get(
    "/{product_id}/prices",
    response_model=ProductPricesResponse,
    responses=NOT_FOUND_ERROR,
)
async def get_product_prices(
    product_id: str, application: ApplicationDependency
) -> ProductPricesResponse:
    return await application.get_prices(product_id)


@router.get(
    "/{product_id}/benchmarks",
    response_model=ProductBenchmarksResponse,
    responses=NOT_FOUND_ERROR,
)
async def get_product_benchmarks(
    product_id: str, application: ApplicationDependency
) -> ProductBenchmarksResponse:
    return await application.get_benchmarks(product_id)


@router.get(
    "/{product_id}/reviews",
    response_model=ProductReviewsResponse,
    responses=NOT_FOUND_ERROR,
)
async def get_product_reviews(
    product_id: str, application: ApplicationDependency
) -> ProductReviewsResponse:
    return await application.get_reviews(product_id)

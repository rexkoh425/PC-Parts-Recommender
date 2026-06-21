"""Every endpoint in one module.

Split into services/api/routers/ once there were more than a couple of resources.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/products")
def list_products() -> dict[str, list]:
    return {"items": []}


@router.post("/builds")
def create_build() -> dict[str, str]:
    return {"status": "not_implemented"}

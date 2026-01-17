"""First sketch of the component schema.

Superseded by domain/models.py once the typed pydantic models landed.
"""

from dataclasses import dataclass


@dataclass
class ComponentSketch:
    component_id: str
    category: str
    brand: str
    model: str
    specs: dict


@dataclass
class BuildSketch:
    components: list[ComponentSketch]
    total_price: float | None = None

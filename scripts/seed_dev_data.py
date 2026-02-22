"""Write a handful of fake components so the API has something to serve.

Superseded by the controlled demo contract fixture.
"""

import json
from pathlib import Path

ROWS = [
    {"component_id": "cpu-1", "category": "cpu", "brand": "AMD", "model": "7600"},
    {"component_id": "gpu-1", "category": "gpu", "brand": "NVIDIA", "model": "4060"},
]


def main() -> None:
    out = Path("data/processed/dev-seed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ROWS, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

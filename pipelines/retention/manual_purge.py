"""Manual deletion helper.

Deletes by path with no registry, no receipt, and no way to prove what was
removed. Replaced by the retention registry and publication plan.
"""

import shutil
import sys
from pathlib import Path


def purge(target: str) -> None:
    path = Path(target)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    purge(sys.argv[1])

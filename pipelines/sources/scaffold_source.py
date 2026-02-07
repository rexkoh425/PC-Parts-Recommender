"""Placeholder adapter used to shape the source interface.

Deleted once sources/base.py defined the real contract.
"""

from collections.abc import Iterator


class ScaffoldSource:
    name = "scaffold"

    def fetch(self) -> Iterator[dict]:
        raise NotImplementedError("no real source is wired up yet")

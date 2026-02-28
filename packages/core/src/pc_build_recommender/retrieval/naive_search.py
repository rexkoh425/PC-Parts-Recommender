"""Substring search over product titles.

Only ever meant to unblock the first end-to-end slice. Replaced by BM25.
"""

from collections.abc import Iterable


def search(query: str, products: Iterable[dict], limit: int = 20) -> list[dict]:
    needle = query.casefold().strip()
    hits = [p for p in products if needle in str(p.get("title", "")).casefold()]
    return hits[:limit]

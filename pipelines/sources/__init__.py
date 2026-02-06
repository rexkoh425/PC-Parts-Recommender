"""Licensed and controlled source adapters."""

from .base import ParseResult, FetchedSnapshot, fetch_http_snapshot, snapshot_local_file

__all__ = [
    "ParseResult",
    "FetchedSnapshot",
    "fetch_http_snapshot",
    "snapshot_local_file",
]


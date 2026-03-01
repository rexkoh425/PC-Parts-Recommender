"""Category-scoped BM25 retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from .models import IndexedDocument, SearchHit
from .text import tokenize

# TODO: rest of this module still to come.

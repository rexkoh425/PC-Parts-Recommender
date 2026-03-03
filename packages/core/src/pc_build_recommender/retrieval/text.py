"""Deterministic text normalisation shared by lexical and vector fallback search."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*")


def normalise_search_text(text: str) -> str:
    """Return stable, Unicode-normalised search text."""

    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_TOKEN_PATTERN.findall(value))


def tokenize(text: str) -> list[str]:
    """Tokenise model numbers and technical terms without external resources."""

    return normalise_search_text(text).split()


def token_features(text: str) -> list[tuple[str, float]]:
    """Generate weighted lexical features for the deterministic hash encoder."""

    tokens = tokenize(text)
    features: list[tuple[str, float]] = [(f"w:{token}", 1.0) for token in tokens]
    features.extend(
        (f"b:{left}_{right}", 1.35) for left, right in zip(tokens, tokens[1:], strict=False)
    )

    # Character n-grams make near-identical model spellings (RTX4070S / RTX
    # 4070S) less brittle while remaining explicitly lexical, not semantic.
    compact = "".join(tokens)
    if len(compact) >= 4:
        features.extend((f"c:{compact[index:index + 4]}", 0.2) for index in range(len(compact) - 3))
    return features


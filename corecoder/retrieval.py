"""Small, dependency-free text helpers shared by local retrievers."""

from __future__ import annotations

import re

_LATIN_RE = re.compile(r"[a-z0-9_.+#-]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")


def tokenize(text: str) -> set[str]:
    """Return normalized Latin tokens plus CJK chunks and bigrams."""
    normalized = text.lower()
    tokens = set(_LATIN_RE.findall(normalized))
    for chunk in _CJK_RE.findall(normalized):
        tokens.add(chunk)
        if len(chunk) > 1:
            tokens.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return {token for token in tokens if token and len(token) > 1}

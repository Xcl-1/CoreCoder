"""Dependency-free keyword retrieval for durable memories."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .models import Memory

_LATIN_RE = re.compile(r"[a-z0-9_.+#-]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")
_TYPE_WEIGHT = {"feedback": 1.2, "user": 1.15, "project": 1.0, "reference": 0.9}


def tokenize(text: str) -> set[str]:
    normalized = text.lower()
    tokens = set(_LATIN_RE.findall(normalized))
    for chunk in _CJK_RE.findall(normalized):
        tokens.add(chunk)
        if len(chunk) > 1:
            tokens.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return {token for token in tokens if token and len(token) > 1}


@dataclass(frozen=True)
class ScoredMemory:
    memory: Memory
    score: float


class MemoryRetriever:
    def retrieve(
        self,
        query: str,
        memories: list[Memory],
        project_path: Path | str | None = None,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> list[ScoredMemory]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        current_project = str(Path(project_path).resolve()) if project_path else None
        scored: list[ScoredMemory] = []
        normalized_query = query.lower()

        for memory in memories:
            if memory.scope == "project":
                if not current_project or not memory.project_path:
                    continue
                if str(Path(memory.project_path).resolve()) != current_project:
                    continue

            searchable = " ".join([memory.title, memory.description, memory.content, *memory.keywords])
            memory_tokens = tokenize(searchable)
            overlap = len(query_tokens & memory_tokens) / math.sqrt(max(1, len(query_tokens) * len(memory_tokens)))
            keyword_bonus = sum(
                0.12 for keyword in set(memory.keywords) if len(keyword.strip()) > 1 and keyword.lower() in normalized_query
            )
            title_bonus = 0.08 * len(query_tokens & tokenize(memory.title))
            score = (overlap + keyword_bonus + title_bonus) * _TYPE_WEIGHT[memory.type]
            if score >= min_score:
                scored.append(ScoredMemory(memory=memory, score=round(score, 4)))

        scored.sort(key=lambda item: (item.score, item.memory.updated_at), reverse=True)
        return scored[: max(0, top_k)]

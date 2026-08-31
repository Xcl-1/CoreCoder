"""Dependency-free keyword retrieval for durable memories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from corecoder.retrieval import tokenize

from .extractor import MemoryExtractor
from .models import Memory

_TYPE_WEIGHT = {
    "feedback": 1.2,
    "profile": 1.15,
    "user": 1.15,
    "procedure": 1.1,
    "project": 1.0,
    "episode": 0.95,
    "reference": 0.9,
}


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
            if memory.status != "active":
                continue
            # Execution-derived assets are intentionally project-bound. Keep
            # legacy global files readable, but never inject them at runtime.
            if memory.type in ("procedure", "episode") and memory.scope != "project":
                continue
            # Keep old task-request misclassifications readable for audit, but
            # do not inject them after stricter extraction rules are deployed.
            if (
                memory.type in ("user", "feedback", "profile", "project", "reference")
                and memory.evidence
                and not any(
                    MemoryExtractor.is_durable_evidence(memory.type, evidence)
                    for evidence in memory.evidence
                )
            ):
                continue
            if memory.scope == "project":
                if not current_project or not memory.project_path:
                    continue
                if str(Path(memory.project_path).resolve()) != current_project:
                    continue

            # Evidence preserves the user's original language even when the LLM
            # writes an English title/description, improving cross-language recall.
            searchable = " ".join([
                memory.title,
                memory.description,
                memory.content,
                *memory.keywords,
                *memory.evidence,
            ])
            memory_tokens = tokenize(searchable)
            overlap = len(query_tokens & memory_tokens) / math.sqrt(max(1, len(query_tokens) * len(memory_tokens)))
            keyword_bonus = sum(
                0.12 for keyword in set(memory.keywords) if len(keyword.strip()) > 1 and keyword.lower() in normalized_query
            )
            title_bonus = 0.08 * len(query_tokens & tokenize(memory.title))
            reliability = (memory.success_count + 1) / (memory.success_count + memory.failure_count + 2)
            confidence_factor = 0.6 + 0.4 * memory.confidence
            feedback_factor = 0.9 + 0.2 * reliability
            scope_factor = 1.05 if memory.scope == "project" else 1.0
            score = (
                (overlap + keyword_bonus + title_bonus)
                * _TYPE_WEIGHT[memory.type]
                * confidence_factor
                * feedback_factor
                * scope_factor
            )
            if score >= min_score:
                scored.append(ScoredMemory(memory=memory, score=round(score, 4)))

        scored.sort(key=lambda item: (item.score, item.memory.updated_at), reverse=True)
        return scored[: max(0, top_k)]

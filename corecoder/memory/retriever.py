"""Dependency-free keyword retrieval for durable memories."""

from __future__ import annotations

import math
from collections import defaultdict
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
    def __init__(self):
        self._index_signature: tuple[tuple[str, int, str], ...] | None = None
        self._postings: dict[str, set[str]] = {}
        self._tokens: dict[str, set[str]] = {}

    def _ensure_index(self, memories: list[Memory]) -> None:
        signature = tuple(sorted(
            (memory.id, memory.version, memory.updated_at)
            for memory in memories
        ))
        if signature == self._index_signature:
            return
        postings: dict[str, set[str]] = defaultdict(set)
        tokens_by_id: dict[str, set[str]] = {}
        for memory in memories:
            searchable = " ".join([
                memory.title,
                memory.description,
                memory.content,
                *memory.keywords,
                *memory.evidence,
            ])
            tokens = tokenize(searchable)
            tokens_by_id[memory.id] = tokens
            for token in tokens:
                postings[token].add(memory.id)
        self._postings = dict(postings)
        self._tokens = tokens_by_id
        self._index_signature = signature

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
        self._ensure_index(memories)
        candidate_ids: set[str] = set()
        for token in query_tokens:
            candidate_ids.update(self._postings.get(token, ()))
        if not candidate_ids:
            return []
        current_project = str(Path(project_path).resolve()) if project_path else None
        scored: list[ScoredMemory] = []
        normalized_query = query.lower()

        for memory in memories:
            if memory.id not in candidate_ids:
                continue
            if memory.status != "active":
                continue
            if memory.type == "procedure" and len(set(memory.verified_sessions)) < 2:
                continue  # Legacy validation counters did not check completion.
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
            memory_tokens = self._tokens.get(memory.id, set())
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

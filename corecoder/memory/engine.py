"""Lifecycle facade for extraction, storage, retrieval and prompt injection."""

from __future__ import annotations

import logging
from pathlib import Path

from .extractor import MemoryExtractor
from .index import MemoryIndex
from .models import ExtractedMemory, Memory, utc_now
from .retriever import MemoryRetriever, tokenize
from .store import MemoryStore, normalize_memory_id

logger = logging.getLogger(__name__)

MEMORY_POLICY_PROMPT = """# Cross-session memory policy
Memory files are managed automatically by MemoryEngine at session end. Do not inspect, edit, script, or directly modify the memory directory unless the user explicitly asks for file-level memory administration. When the user states or updates a preference, acknowledge that it will be considered for saving at session end; do not claim it is already saved and do not use tools to persist it yourself."""


class MemoryEngine:
    def __init__(
        self,
        llm,
        root: Path | str | None = None,
        project_path: Path | str | None = None,
        top_k: int = 5,
        max_prompt_chars: int = 4_000,
    ):
        self.store = MemoryStore(root)
        self.index = MemoryIndex(self.store.root)
        self.extractor = MemoryExtractor(llm)
        self.retriever = MemoryRetriever()
        self.project_path = Path(project_path or Path.cwd()).resolve()
        self.top_k = top_k
        self.max_prompt_chars = max_prompt_chars

    def build_prompt(self, query: str) -> str:
        matches = self.retriever.retrieve(
            query,
            self.store.list(),
            project_path=self.project_path,
            top_k=self.top_k,
        )
        parts = [MEMORY_POLICY_PROMPT]
        used = len(MEMORY_POLICY_PROMPT)
        if not matches:
            return MEMORY_POLICY_PROMPT
        context_header = (
            "\n\n# Relevant cross-session memory\n"
            "Treat these as potentially stale context. The current request and safety rules take precedence.\n"
        )
        if used + len(context_header) > self.max_prompt_chars:
            return MEMORY_POLICY_PROMPT
        parts.append(context_header)
        used += len(context_header)
        for match in matches:
            memory = match.memory
            fragment = f"\n- [{memory.type}:{memory.id}] {memory.title}: {memory.content.strip()}\n"
            if used + len(fragment) > self.max_prompt_chars:
                break
            parts.append(fragment)
            used += len(fragment)
        return "".join(parts).strip()

    def learn(self, messages: list[dict], source_session: str) -> list[Memory]:
        if not self._has_complete_exchange(messages):
            return []
        existing = self.store.list()
        proposals = self.extractor.extract(messages, existing)
        if not proposals:
            return []

        saved: list[Memory] = []
        by_id = {memory.id: memory for memory in existing}
        for proposal in proposals:
            if proposal.action == "ignore" or proposal.confidence < 0.55:
                continue
            target = by_id.get(proposal.target_id or "")
            if target is None:
                target = self._find_duplicate(proposal, list(by_id.values()))
            memory = self._merge(target, proposal, source_session)
            if memory is None:
                continue
            stored = self.store.save(memory)
            by_id[stored.id] = stored
            saved.append(stored)

        if saved:
            self.index.rebuild(list(by_id.values()))
        return saved

    def forget(self, memory_id: str) -> bool:
        deleted = self.store.delete(memory_id)
        if deleted:
            self.index.rebuild(self.store.list())
        return deleted

    def stats(self) -> dict[str, int]:
        memories = self.store.list()
        result = {"total": len(memories), "global": 0, "project": 0}
        for memory in memories:
            result[memory.scope] += 1
        return result

    def _merge(
        self,
        target: Memory | None,
        proposal: ExtractedMemory,
        source_session: str,
    ) -> Memory | None:
        title = proposal.title.strip()[:120]
        description = proposal.description.strip()[:300]
        content = proposal.content.strip()[:4_000]
        if not title or not description or not content:
            return None
        keywords = list(dict.fromkeys(k.strip().lower() for k in proposal.keywords if k.strip()))[:15]
        now = utc_now()
        if target is None:
            return Memory(
                id=normalize_memory_id(title),
                title=title,
                description=description,
                content=content,
                type=proposal.type,
                scope=proposal.scope,
                project_path=str(self.project_path) if proposal.scope == "project" else None,
                keywords=keywords,
                confidence=proposal.confidence,
                source_sessions=[source_session],
                created_at=now,
                updated_at=now,
            )

        sources = list(dict.fromkeys([*target.source_sessions, source_session]))
        return target.model_copy(
            update={
                "title": title,
                "description": description,
                "content": content,
                "type": proposal.type,
                "scope": proposal.scope,
                "project_path": str(self.project_path) if proposal.scope == "project" else None,
                "keywords": list(dict.fromkeys([*target.keywords, *keywords]))[:15],
                "confidence": max(target.confidence, proposal.confidence),
                "source_sessions": sources,
                "updated_at": now,
            }
        )

    def _find_duplicate(self, proposal: ExtractedMemory, memories: list[Memory]) -> Memory | None:
        proposal_tokens = tokenize(f"{proposal.title} {proposal.description} {' '.join(proposal.keywords)}")
        for memory in memories:
            if memory.scope != proposal.scope:
                continue
            if memory.scope == "project" and memory.project_path != str(self.project_path):
                continue
            if memory.title.casefold() == proposal.title.casefold():
                return memory
            existing_tokens = tokenize(f"{memory.title} {memory.description} {' '.join(memory.keywords)}")
            union = proposal_tokens | existing_tokens
            if union and len(proposal_tokens & existing_tokens) / len(union) >= 0.65:
                return memory
        return None

    @staticmethod
    def _has_complete_exchange(messages: list[dict]) -> bool:
        has_user = any(message.get("role") == "user" and message.get("content") for message in messages)
        has_assistant = any(message.get("role") == "assistant" and message.get("content") for message in messages)
        return has_user and has_assistant

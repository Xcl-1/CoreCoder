"""Lifecycle facade for reflection, extraction, storage and retrieval."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .extractor import MemoryExtractor
from .index import MemoryIndex
from .models import ExtractedMemory, Memory, SessionReflection, utc_now
from .reflection import MemoryReflector
from .retriever import MemoryRetriever, ScoredMemory, tokenize
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
        self.reflector = MemoryReflector(llm)
        self.retriever = MemoryRetriever()
        self.project_path = Path(project_path or Path.cwd()).resolve()
        self.top_k = top_k
        self.max_prompt_chars = max_prompt_chars
        self._retrieved_this_session: set[str] = set()
        self.last_learning_error: str | None = None

    # ---- retrieval ----------------------------------------------------

    def build_prompt(self, query: str) -> str:
        matches = self.retriever.retrieve(
            query,
            self.store.list(),
            project_path=self.project_path,
            top_k=self.top_k,
        )
        self._record_retrieval(matches)
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

    def search(self, query: str, top_k: int = 20) -> list[ScoredMemory]:
        """Search active memories without recording a runtime use."""
        return self.retriever.retrieve(
            query,
            self.store.list(),
            project_path=self.project_path,
            top_k=top_k,
            min_score=0.01,
        )

    def _record_retrieval(self, matches: list[ScoredMemory]) -> None:
        new_ids = [match.memory.id for match in matches if match.memory.id not in self._retrieved_this_session]
        if not new_ids:
            return
        now = utc_now()
        with self.store.locked():
            for memory_id in new_ids:
                current = self.store.get(memory_id)
                if current is None or current.status != "active":
                    continue
                self.store.save(current.model_copy(update={
                    "last_used_at": now,
                    "use_count": current.use_count + 1,
                }))
                self._retrieved_this_session.add(memory_id)

    # ---- reflection and learning -------------------------------------

    def learn(
        self,
        messages: list[dict],
        source_session: str,
        replay_path: Path | str | None = None,
        *,
        _structured_retry: bool = False,
        _record_failure: bool = True,
    ) -> list[Memory]:
        self.last_learning_error = None
        if not self._has_complete_exchange(messages):
            return []

        reflection = self.reflector.reflect(messages, replay_path) if replay_path else None
        evidence_source = self.reflector.source_text(messages, replay_path) if reflection else ""
        existing = self.store.list()
        proposals: list[ExtractedMemory] = []
        extraction_completed = False
        fallback_attempted = False

        if _structured_retry and reflection and self._supports_execution_reflection(reflection):
            fallback_attempted = True
            fallback = self.extractor.extract_execution_fallback(
                messages,
                existing,
                reflection,
                evidence_source,
            )
            extraction_completed = self.extractor.last_fallback_succeeded
            self._extend_distinct(proposals, fallback)

        general = self.extractor.extract(
            messages,
            existing,
            reflection=reflection,
            evidence_source=evidence_source,
        )
        extraction_completed = extraction_completed or self.extractor.last_succeeded
        self._extend_distinct(proposals, general)

        missing_execution_asset = bool(
            reflection
            and (
                (
                    self.extractor.supports_procedure(reflection)
                    and not self._contains_execution_type(proposals, "procedure")
                )
                or (
                    self.extractor.supports_episode(reflection)
                    and not self._contains_execution_type(proposals, "episode")
                )
            )
        )
        if reflection and missing_execution_asset and not fallback_attempted:
            fallback = self.extractor.extract_execution_fallback(
                messages,
                existing,
                reflection,
                evidence_source,
                include_procedure=not self._contains_execution_type(proposals, "procedure"),
                include_episode=not self._contains_execution_type(proposals, "episode"),
            )
            extraction_completed = extraction_completed or self.extractor.last_fallback_succeeded
            self._extend_distinct(proposals, fallback)

        if not extraction_completed:
            self.last_learning_error = self.extractor.last_error or "memory extraction did not complete"
            if _record_failure:
                self._record_pending_failure_for_session(source_session, self.last_learning_error)
            return []

        saved: list[Memory] = []
        changed = False
        with self.store.locked():
            latest = self.store.list()
            by_id = {memory.id: memory for memory in latest}
            for proposal in proposals:
                if proposal.action == "ignore" or proposal.confidence < 0.55:
                    continue
                if proposal.action == "archive":
                    archived = self._archive_locked(proposal.target_id or "", by_id)
                    if archived:
                        saved.append(archived)
                        changed = True
                    continue

                target = by_id.get(proposal.target_id or "")
                if target is not None and not self._compatible_target(target, proposal):
                    target = None
                if target is None:
                    target = self._find_duplicate(proposal, list(by_id.values()))
                memory = self._merge(target, proposal, source_session, reflection)
                if memory is None:
                    continue
                stored = self.store.save(memory)
                by_id[stored.id] = stored
                saved.append(stored)
                changed = True

                if (
                    proposal.supersedes
                    and proposal.supersedes != stored.id
                    and self._supersede_locked(proposal.supersedes, stored.id, by_id)
                ):
                    changed = True

            if reflection and self._apply_outcome_locked(reflection, by_id):
                changed = True
            if changed:
                self.index.rebuild(list(by_id.values()))

        self._retrieved_this_session.clear()
        self._clear_pending(source_session)
        return saved

    # ---- crash recovery queue ----------------------------------------

    def checkpoint(
        self,
        messages: list[dict],
        source_session: str,
        replay_path: Path | str | None = None,
    ) -> Path | None:
        """Atomically queue the latest complete exchange for crash recovery."""
        if not self._has_complete_exchange(messages):
            return None
        pending_dir = self.store.root / ".pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        safe_id = normalize_memory_id(source_session)
        path = pending_dir / f"{safe_id}.json"
        temporary = pending_dir / f"{safe_id}.tmp"
        payload = {
            "session_id": source_session,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpointed_at": time.time(),
            "attempts": 0,
            "messages": messages,
            "replay_path": str(replay_path) if replay_path else None,
        }
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        return path

    def recover_pending(
        self,
        limit: int = 10,
        exclude_session: str | None = None,
        force: bool = False,
        min_age_seconds: float = 300.0,
    ) -> int:
        """Learn from checkpoints left by interrupted processes."""
        pending_dir = self.store.root / ".pending"
        if not pending_dir.exists():
            return 0
        recovered = 0
        paths = sorted(pending_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
        for path in paths[:limit]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                session_id = str(payload["session_id"])
                if session_id == exclude_session:
                    continue
                checkpointed_at = float(payload.get("checkpointed_at", path.stat().st_mtime))
                if not force and time.time() - checkpointed_at < min_age_seconds:
                    continue
                messages = payload["messages"]
                if not isinstance(messages, list):
                    raise TypeError("pending messages must be a list")
                self.learn(
                    messages,
                    session_id,
                    payload.get("replay_path"),
                    _structured_retry=True,
                    _record_failure=False,
                )
                if not path.exists():
                    recovered += 1
                else:
                    self._record_pending_failure(path, payload, self.last_learning_error)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                logger.warning("Skipping invalid pending memory checkpoint: %s", path)
        return recovered

    def _record_pending_failure(self, path: Path, payload: dict, error: str | None = None) -> None:
        attempts = int(payload.get("attempts", 0)) + 1
        payload["attempts"] = attempts
        payload["last_attempted_at"] = utc_now()
        payload["last_error"] = (error or "memory extraction did not complete")[:500]
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        if attempts >= 3:
            failed_dir = self.store.root / ".failed"
            failed_dir.mkdir(parents=True, exist_ok=True)
            path.replace(failed_dir / path.name)
            logger.warning("Moved repeatedly failing memory checkpoint to %s", failed_dir / path.name)
            return

    def _record_pending_failure_for_session(self, source_session: str, error: str) -> None:
        path = self.store.root / ".pending" / f"{normalize_memory_id(source_session)}.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._record_pending_failure(path, payload, error)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Could not record pending memory failure: %s", path)

    def pending_count(self) -> int:
        pending_dir = self.store.root / ".pending"
        return len(list(pending_dir.glob("*.json"))) if pending_dir.exists() else 0

    def pending_status(self, limit: int = 10) -> list[dict[str, str | int]]:
        """Return safe, concise retry details for pending reflection checkpoints."""
        pending_dir = self.store.root / ".pending"
        if not pending_dir.exists():
            return []
        statuses: list[dict[str, str | int]] = []
        paths = sorted(pending_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
        for path in paths[: max(0, limit)]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                statuses.append({
                    "session_id": str(payload.get("session_id", path.stem)),
                    "attempts": int(payload.get("attempts", 0)),
                    "last_attempted_at": str(payload.get("last_attempted_at", "-")),
                    "last_error": str(payload.get("last_error", "not retried yet"))[:200],
                })
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                statuses.append({
                    "session_id": path.stem,
                    "attempts": 0,
                    "last_attempted_at": "-",
                    "last_error": "invalid pending checkpoint",
                })
        return statuses

    def _clear_pending(self, source_session: str) -> None:
        path = self.store.root / ".pending" / f"{normalize_memory_id(source_session)}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # ---- lifecycle administration ------------------------------------

    def forget(self, memory_id: str) -> bool:
        with self.store.locked():
            deleted = self.store.delete(memory_id)
            if deleted:
                self.index.rebuild(self.store.list())
        self._retrieved_this_session.discard(memory_id)
        return deleted

    def archive(self, memory_id: str) -> Memory | None:
        with self.store.locked():
            memories = self.store.list()
            by_id = {memory.id: memory for memory in memories}
            archived = self._archive_locked(memory_id, by_id)
            if archived:
                self.index.rebuild(list(by_id.values()))
        self._retrieved_this_session.discard(memory_id)
        return archived

    def approve(self, memory_id: str) -> Memory | None:
        """Explicitly reactivate an archived or superseded memory."""
        with self.store.locked():
            memories = self.store.list()
            by_id = {memory.id: memory for memory in memories}
            target = by_id.get(normalize_memory_id(memory_id))
            if target is None:
                return None
            if target.status == "active":
                return target
            approved = target.model_copy(update={
                "status": "active",
                "version": target.version + 1,
                "updated_at": utc_now(),
            })
            stored = self.store.save(approved)
            by_id[stored.id] = stored
            self.index.rebuild(list(by_id.values()))
            return stored

    def stats(self) -> dict[str, int]:
        memories = self.store.list()
        result = {
            "total": len(memories),
            "global": 0,
            "project": 0,
            "candidate": 0,
            "active": 0,
            "archived": 0,
            "superseded": 0,
        }
        for memory in memories:
            result[memory.scope] += 1
            result[memory.status] += 1
        return result

    def _archive_locked(self, memory_id: str, by_id: dict[str, Memory]) -> Memory | None:
        target = by_id.get(normalize_memory_id(memory_id))
        if target is None:
            return None
        archived = target.model_copy(update={
            "status": "archived",
            "version": target.version + 1,
            "updated_at": utc_now(),
        })
        stored = self.store.save(archived)
        by_id[stored.id] = stored
        return stored

    def _supersede_locked(self, old_id: str, replacement_id: str, by_id: dict[str, Memory]) -> bool:
        target = by_id.get(normalize_memory_id(old_id))
        replacement = by_id.get(replacement_id)
        if target is None or replacement is None or target.scope != replacement.scope:
            return False
        superseded = target.model_copy(update={
            "status": "superseded",
            "version": target.version + 1,
            "updated_at": utc_now(),
        })
        stored = self.store.save(superseded)
        by_id[stored.id] = stored
        return True

    def _apply_outcome_locked(
        self,
        reflection: SessionReflection,
        by_id: dict[str, Memory],
    ) -> bool:
        if reflection.outcome not in ("success", "partial", "failure"):
            return False
        changed = False
        now = utc_now()
        for memory_id in self._retrieved_this_session:
            memory = by_id.get(memory_id)
            if memory is None:
                continue
            update = {"updated_at": now}
            if reflection.outcome == "success":
                update["success_count"] = memory.success_count + 1
                update["confidence"] = min(1.0, memory.confidence + 0.02)
            else:
                update["failure_count"] = memory.failure_count + 1
                update["confidence"] = max(0.1, memory.confidence - 0.05)
            stored = self.store.save(memory.model_copy(update=update))
            by_id[stored.id] = stored
            changed = True
        return changed

    # ---- merge and conflict helpers ----------------------------------

    @staticmethod
    def _contains_execution_type(proposals: list[ExtractedMemory], memory_type: str) -> bool:
        return any(
            proposal.type == memory_type and proposal.action in ("create", "merge")
            for proposal in proposals
        )

    @staticmethod
    def _extend_distinct(
        proposals: list[ExtractedMemory],
        additions: list[ExtractedMemory],
    ) -> None:
        """Keep at most one create/merge proposal for each execution-memory type."""
        for proposal in additions:
            if (
                proposal.type in ("procedure", "episode")
                and proposal.action in ("create", "merge")
                and MemoryEngine._contains_execution_type(proposals, proposal.type)
            ):
                continue
            proposals.append(proposal)

    @staticmethod
    def _supports_execution_reflection(reflection: SessionReflection) -> bool:
        return MemoryExtractor.supports_procedure(reflection) or MemoryExtractor.supports_episode(reflection)

    def _merge(
        self,
        target: Memory | None,
        proposal: ExtractedMemory,
        source_session: str,
        reflection: SessionReflection | None = None,
    ) -> Memory | None:
        title = proposal.title.strip()[:120]
        description = proposal.description.strip()[:300]
        content = proposal.content.strip()[:4_000]
        if not title or not description or not content:
            return None
        keywords = list(dict.fromkeys(k.strip().lower() for k in proposal.keywords if k.strip()))[:15]
        evidence = proposal.evidence.strip()[:500]
        now = utc_now()
        execution_asset = proposal.type in ("procedure", "episode")
        independently_validated = execution_asset and self._supports_execution_asset(proposal, reflection)
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
                evidence=[evidence] if evidence else [],
                source_sessions=[source_session],
                validation_count=1 if independently_validated else 0,
                validated_at=now if independently_validated else None,
                status="candidate" if independently_validated else "active",
                supersedes=proposal.supersedes,
                created_at=now,
                updated_at=now,
            )

        sources = list(dict.fromkeys([*target.source_sessions, source_session]))
        evidence_items = list(dict.fromkeys([*target.evidence, evidence] if evidence else target.evidence))[-20:]
        validation_count = target.validation_count
        validated_at = target.validated_at
        status = "active"
        if independently_validated:
            if source_session not in target.source_sessions:
                validation_count += 1
                validated_at = now
            status = "active" if target.status == "active" or validation_count >= 2 else "candidate"
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
                "evidence": evidence_items,
                "source_sessions": sources,
                "validation_count": validation_count,
                "validated_at": validated_at,
                "status": status,
                "supersedes": proposal.supersedes or target.supersedes,
                "version": target.version + 1,
                "updated_at": now,
            }
        )

    def _find_duplicate(self, proposal: ExtractedMemory, memories: list[Memory]) -> Memory | None:
        proposal_tokens = tokenize(f"{proposal.title} {proposal.description} {' '.join(proposal.keywords)}")
        for memory in memories:
            if memory.status not in ("active", "candidate") or memory.scope != proposal.scope or memory.type != proposal.type:
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

    def _compatible_target(self, target: Memory, proposal: ExtractedMemory) -> bool:
        if target.type != proposal.type or target.scope != proposal.scope:
            return False
        return proposal.scope != "project" or target.project_path == str(self.project_path)

    @staticmethod
    def _supports_execution_asset(
        proposal: ExtractedMemory,
        reflection: SessionReflection | None,
    ) -> bool:
        if proposal.type == "procedure":
            return MemoryExtractor.supports_procedure(reflection)
        return proposal.type == "episode" and MemoryExtractor.supports_episode(reflection)

    @staticmethod
    def _has_complete_exchange(messages: list[dict]) -> bool:
        has_user = any(message.get("role") == "user" and message.get("content") for message in messages)
        has_assistant = any(message.get("role") == "assistant" and message.get("content") for message in messages)
        return has_user and has_assistant

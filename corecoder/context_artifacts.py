"""Content-addressed storage for large tool results.

Large observations are written once and represented in the model transcript by
a small, stable preview.  The full, already-sanitized text can be retrieved by
artifact id without making the prompt prefix churn on every compression pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

CONTEXT_ARTIFACTS_DIR = Path.home() / ".corecoder" / "context-artifacts"
_ARTIFACT_RE = re.compile(r"^artifact://sha256/([0-9a-f]{64})$")
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DIGEST_FILE_RE = re.compile(r"^[0-9a-f]{64}\.txt$")


@dataclass(frozen=True)
class ArtifactPruneResult:
    removed_artifacts: int = 0
    removed_bytes: int = 0


def _safe_session_id(value: str) -> str:
    name = value.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")[:100]
    return name or "session"


class ContextArtifactStore:
    """Per-session, traversal-safe store for sanitized tool observations."""

    def __init__(
        self,
        session_id: str,
        root: str | Path | None = None,
        threshold_chars: int = 12_000,
        preview_chars: int = 1_200,
        ttl_seconds: int = 30 * 24 * 60 * 60,
        max_total_bytes: int = 256 * 1024 * 1024,
    ):
        self.session_id = _safe_session_id(session_id)
        self.root = Path(root or CONTEXT_ARTIFACTS_DIR).expanduser().resolve()
        self.session_dir = (self.root / self.session_id).resolve()
        if self.session_dir.parent != self.root:
            raise ValueError("Invalid context artifact session id")
        self.threshold_chars = max(1_000, threshold_chars)
        self.preview_chars = max(300, preview_chars)
        self.ttl_seconds = max(60, ttl_seconds)
        self.max_total_bytes = max(1_000, max_total_bytes)
        self.externalized_count = 0
        self.externalized_chars = 0
        self.placeholder_chars = 0
        self.stored_count = 0
        self.retrieval_count = 0
        self.retrieved_chars = 0
        self.pruned_count = 0
        self.pruned_bytes = 0

    def externalize(self, content: str, tool_name: str) -> str:
        """Return content unchanged when small, otherwise persist and preview it."""
        if len(content) <= self.threshold_chars or self.is_placeholder(content):
            return content
        artifact_id = self.put(content, tool_name)
        lines = content.count("\n") + 1
        preview = self._preview(content)
        placeholder = (
            "[Tool Artifact]\n"
            f"id: {artifact_id}\n"
            f"tool: {tool_name}\n"
            f"size: {len(content)} chars, {lines} lines\n"
            f"preview:\n{preview}\n"
            "Use retrieve_context with this id to recover exact details."
        )
        self.externalized_count += 1
        self.externalized_chars += len(content)
        self.placeholder_chars += len(placeholder)
        return placeholder

    def put(self, content: str, tool_name: str) -> str:
        """Persist text under its SHA-256 digest and return a stable artifact id."""
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        artifact_id = f"artifact://sha256/{digest}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        text_path = self._path_for_digest(digest)
        metadata_path = text_path.with_suffix(".json")
        if not text_path.exists():
            self._atomic_write(text_path, content)
            self.stored_count += 1
        if not metadata_path.exists():
            metadata = {
                "id": artifact_id,
                "tool": tool_name,
                "chars": len(content),
                "lines": content.count("\n") + 1,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._atomic_write(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2))
        self.prune(protected_ids={artifact_id})
        return artifact_id

    def retrieve(
        self,
        artifact_id: str,
        *,
        query: str = "",
        start_line: int | None = None,
        end_line: int | None = None,
        max_chars: int = 12_000,
    ) -> str:
        """Retrieve a bounded exact range or keyword-focused excerpt."""
        digest = self._digest_from_id(artifact_id)
        path = self._path_for_digest(digest)
        if not path.exists():
            return f"Error: context artifact not found in this session: {artifact_id}"

        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        limit = min(max(500, max_chars), 20_000)

        if query:
            selected = self._query_lines(lines, query)
            label = f"matches for {query!r}"
        elif start_line is not None or end_line is not None:
            start = max(1, start_line or 1)
            end = min(len(lines), end_line or (start + 199))
            if end < start:
                return "Error: end_line must be greater than or equal to start_line"
            selected = [(number, lines[number - 1]) for number in range(start, end + 1)]
            label = f"lines {start}-{end}"
        else:
            selected = list(enumerate(lines[:200], start=1))
            label = "first 200 lines"

        body = "\n".join(f"{number}: {line}" for number, line in selected)
        if not body:
            body = "[no matching lines]"
        truncated = len(body) > limit
        body = body[:limit]
        suffix = "\n[retrieval truncated; request a narrower query or line range]" if truncated else ""
        result = f"[Context Artifact {artifact_id} — {label}]\n{body}{suffix}"
        self.retrieval_count += 1
        self.retrieved_chars += len(result)
        return result

    def prune(
        self,
        *,
        now: float | None = None,
        protected_ids: set[str] | None = None,
    ) -> ArtifactPruneResult:
        """Remove expired artifacts, then oldest artifacts above the byte cap."""
        if not self.root.exists():
            return ArtifactPruneResult()
        protected = {
            self._digest_from_id(artifact_id)
            for artifact_id in (protected_ids or set())
        }
        current_time = time.time() if now is None else now
        candidates: list[tuple[float, str, Path, Path, int]] = []
        for session_dir in self.root.iterdir():
            if (
                not session_dir.is_dir()
                or session_dir.is_symlink()
                or session_dir.parent.resolve() != self.root
            ):
                continue
            for text_path in session_dir.glob("*.txt"):
                if text_path.is_symlink() or not _DIGEST_FILE_RE.fullmatch(text_path.name):
                    continue
                resolved = text_path.resolve()
                if resolved.parent != session_dir.resolve():
                    continue
                digest = text_path.stem
                metadata_path = text_path.with_suffix(".json")
                try:
                    stat = text_path.stat()
                    metadata_size = metadata_path.stat().st_size if metadata_path.exists() else 0
                except OSError:
                    continue
                candidates.append(
                    (stat.st_mtime, digest, text_path, metadata_path, stat.st_size + metadata_size)
                )

        removed_count = 0
        removed_bytes = 0
        survivors: list[tuple[float, str, Path, Path, int]] = []
        for candidate in candidates:
            modified, digest, text_path, metadata_path, size = candidate
            expired = current_time - modified > self.ttl_seconds
            if expired and digest not in protected:
                self._remove_pair(text_path, metadata_path)
                removed_count += 1
                removed_bytes += size
            else:
                survivors.append(candidate)

        total_bytes = sum(candidate[4] for candidate in survivors)
        for _modified, digest, text_path, metadata_path, size in sorted(survivors):
            if total_bytes <= self.max_total_bytes:
                break
            if digest in protected:
                continue
            self._remove_pair(text_path, metadata_path)
            total_bytes -= size
            removed_count += 1
            removed_bytes += size
        self.pruned_count += removed_count
        self.pruned_bytes += removed_bytes
        return ArtifactPruneResult(removed_count, removed_bytes)

    def stats(self) -> dict[str, int]:
        """Return cheap process-local governance and retrieval counters."""
        return {
            "externalized": self.externalized_count,
            "externalized_chars": self.externalized_chars,
            "placeholder_chars": self.placeholder_chars,
            "saved_prompt_chars": max(0, self.externalized_chars - self.placeholder_chars),
            "stored": self.stored_count,
            "retrievals": self.retrieval_count,
            "retrieved_chars": self.retrieved_chars,
            "pruned": self.pruned_count,
            "pruned_bytes": self.pruned_bytes,
        }

    @staticmethod
    def is_placeholder(content: str) -> bool:
        return content.startswith("[Tool Artifact]\nid: artifact://sha256/")

    def _path_for_digest(self, digest: str) -> Path:
        path = (self.session_dir / f"{digest}.txt").resolve()
        if path.parent != self.session_dir:
            raise ValueError("Invalid context artifact id")
        return path

    @staticmethod
    def _digest_from_id(artifact_id: str) -> str:
        match = _ARTIFACT_RE.fullmatch(artifact_id.strip())
        if match is None:
            raise ValueError("artifact_id must use artifact://sha256/<64 hex characters>")
        return match.group(1)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _remove_pair(text_path: Path, metadata_path: Path) -> None:
        for path in (text_path, metadata_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    def _preview(self, content: str) -> str:
        compact = content.strip()
        if len(compact) <= self.preview_chars:
            return compact
        head_size = int(self.preview_chars * 0.65)
        tail_size = self.preview_chars - head_size
        return (
            compact[:head_size].rstrip()
            + "\n... [artifact preview truncated] ...\n"
            + compact[-tail_size:].lstrip()
        )

    @staticmethod
    def _query_lines(lines: list[str], query: str) -> list[tuple[int, str]]:
        needle = query.casefold()
        matches = [index for index, line in enumerate(lines) if needle in line.casefold()]
        wanted: set[int] = set()
        for index in matches[:50]:
            wanted.update(range(max(0, index - 1), min(len(lines), index + 2)))
        return [(index + 1, lines[index]) for index in sorted(wanted)]

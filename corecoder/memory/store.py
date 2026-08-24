"""Markdown-backed storage: one file per memory."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import ValidationError

from ..config import resolve_memory_dir
from .models import Memory

DEFAULT_MEMORY_DIR = resolve_memory_dir("~/.corecoder/memory")
_SAFE_ID_RE = re.compile(r"[^a-z0-9._-]+")


def normalize_memory_id(value: str) -> str:
    """Turn an LLM-provided title or id into a safe, stable filename."""
    normalized = _SAFE_ID_RE.sub("-", value.strip().lower()).strip(".-_")[:80]
    if normalized:
        return normalized
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"memory-{digest}"


class MemoryStore:
    def __init__(self, root: Path | str | None = None):
        self.root = resolve_memory_dir(root)

    def list(self) -> list[Memory]:
        if not self.root.exists():
            return []
        memories: list[Memory] = []
        for path in self.root.glob("*.md"):
            if path.name == "MEMORY.md":
                continue
            memory = self._read(path)
            if memory is not None:
                memories.append(memory)
        return sorted(memories, key=lambda item: item.updated_at, reverse=True)

    def get(self, memory_id: str) -> Memory | None:
        return self._read(self._path(memory_id))

    def save(self, memory: Memory) -> Memory:
        self.root.mkdir(parents=True, exist_ok=True)
        safe_id = normalize_memory_id(memory.id or memory.title)
        stored = memory.model_copy(update={"id": safe_id})
        metadata = stored.model_dump(exclude={"content"})
        text = (
            "---\n"
            + json.dumps(metadata, ensure_ascii=False, indent=2)
            + "\n---\n\n"
            + f"# {stored.title}\n\n{stored.content.strip()}\n"
        )
        path = self._path(safe_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        return stored

    def delete(self, memory_id: str) -> bool:
        path = self._path(memory_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def _path(self, memory_id: str) -> Path:
        safe_id = normalize_memory_id(memory_id)
        path = (self.root / f"{safe_id}.md").resolve()
        if path.parent != self.root.resolve():
            raise ValueError("Invalid memory id")
        return path

    @staticmethod
    def _read(path: Path) -> Memory | None:
        if not path.exists():
            return None
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines or lines[0].strip() != "---":
                return None
            end = lines.index("---", 1)
            metadata = json.loads("\n".join(lines[1:end]))
            body = lines[end + 1 :]
            while body and not body[0].strip():
                body.pop(0)
            if body and body[0].startswith("# "):
                body.pop(0)
            while body and not body[0].strip():
                body.pop(0)
            return Memory.model_validate({**metadata, "content": "\n".join(body).strip()})
        except (OSError, ValueError, json.JSONDecodeError, ValidationError):
            return None

"""Generation of the human-readable MEMORY.md index."""

from __future__ import annotations

from pathlib import Path

from .models import Memory


class MemoryIndex:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def rebuild(self, memories: list[Memory]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        lines = [
            "# CoreCoder Memory",
            "",
            "This file is generated automatically. Edit individual memory files instead.",
            "",
            "| Memory | Type | Scope | Description | Updated |",
            "| --- | --- | --- | --- | --- |",
        ]
        for memory in sorted(memories, key=lambda item: (item.type, item.title.lower())):
            description = memory.description.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| [{memory.title}]({memory.id}.md) | {memory.type} | {memory.scope} | {description} | {memory.updated_at} |"
            )
        path = self.root / "MEMORY.md"
        temporary = self.root / "MEMORY.tmp"
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

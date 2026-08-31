"""File creation / overwrite."""

from pathlib import Path

from ..sandbox import is_write_blocked
from .base import Tool
from .changes import current_change_tracker, missing_parent_dirs


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing one. "
        "For small edits to existing files, prefer edit_file instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the file",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    def _execute_sync(self, file_path: str, content: str) -> str:
        try:
            if is_write_blocked(file_path):
                return f"Error: writing to {file_path} is blocked by sandbox policy"
            p = Path(file_path).expanduser().resolve()
            before = p.read_bytes() if p.is_file() else None
            original_mode = p.stat().st_mode if p.is_file() else None
            created_dirs = missing_parent_dirs(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            current_change_tracker().record(
                p,
                before=before,
                after=p.read_bytes(),
                original_mode=original_mode,
                created_dirs=created_dirs,
            )
            n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Wrote {n_lines} lines to {file_path}"
        except Exception as e:
            return f"Error: {e}"

"""File pattern matching."""

from pathlib import Path

from .base import Tool
from .sensitive import sensitive_path

_SKIP_DIRS = {
    ".git", ".corecoder", ".test_runs", "replays", "node_modules", "__pycache__",
    ".venv", "venv", ".tox", "dist", "build",
}


class GlobTool(Tool):
    name = "glob"
    input_types = ("pattern", "path")
    output_type = "file_paths"
    permission_scope = "filesystem:read"
    side_effect = "none"
    network_access = "none"
    declared_risk = "low"
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: cwd)",
            },
        },
        "required": ["pattern"],
    }

    def _execute_sync(self, pattern: str, path: str = ".") -> str:
        try:
            base = Path(path).expanduser().resolve()
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            hits = [
                hit
                for hit in base.glob(pattern)
                if not sensitive_path(hit)
                and not any(part in _SKIP_DIRS for part in hit.relative_to(base).parts)
            ]
            # sort by mtime, newest first
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            total = len(hits)
            shown = hits[:100]
            lines = [str(h) for h in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return result or "No files matched."
        except Exception as e:
            return f"Error: {e}"

"""Search-and-replace file editing (Claude Code's key innovation).

The core idea: instead of sending whole-file rewrites or line-number patches,
the LLM specifies an *exact* substring to find and its replacement. The
substring must appear exactly once in the file, which eliminates ambiguity
and makes edits safe and reviewable.
"""

from pathlib import Path

from ..sandbox import is_write_blocked
from ._utils import unified_diff
from .base import Tool
from .changes import current_change_tracker, default_change_tracker

# track files changed this session for /diff
_changed_files = default_change_tracker().changed_files


class EditFileTool(Tool):
    name = "edit_file"
    input_types = ("file_path", "text_patch")
    output_type = "diff"
    permission_scope = "filesystem:write"
    side_effect = "local_write"
    description = (
        "Edit a file by replacing an exact string match. "
        "old_string must appear exactly once in the file for safety. "
        "Include enough surrounding context to ensure uniqueness."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find (must be unique in file)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def _execute_sync(self, file_path: str, old_string: str, new_string: str) -> str:
        try:
            if is_write_blocked(file_path):
                return f"Error: writing to {file_path} is blocked by sandbox policy"
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"

            try:
                content = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"Error: {file_path} is not a UTF-8 text file (edit_file only edits text files)"
            occurrences = content.count(old_string)

            if occurrences == 0:
                preview = content[:500] + ("..." if len(content) > 500 else "")
                return (
                    f"Error: old_string not found in {file_path}.\n"
                    f"File starts with:\n{preview}"
                )
            if occurrences > 1:
                return (
                    f"Error: old_string appears {occurrences} times in {file_path}. "
                    f"Include more surrounding lines to make it unique."
                )

            before = p.read_bytes()
            original_mode = p.stat().st_mode
            new_content = content.replace(old_string, new_string, 1)
            p.write_text(new_content, encoding="utf-8")
            current_change_tracker().record(
                p,
                before=before,
                after=p.read_bytes(),
                original_mode=original_mode,
            )

            # generate a unified diff so the user/LLM can see exactly what changed
            diff = unified_diff(content, new_content, str(p))
            return f"Edited {file_path}\n{diff}"
        except Exception as e:
            return f"Error: {e}"

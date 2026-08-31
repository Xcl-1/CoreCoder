"""Tool for undoing all tracked file changes in the current agent session."""

from .base import Tool
from .changes import current_change_tracker


class UndoChangesTool(Tool):
    name = "undo_changes"
    description = (
        "Undo all file changes made through CoreCoder write/edit tools in the current session. "
        "Only use when the user explicitly asks to undo or revert the current changes. "
        "By default, files changed externally after the agent edit are reported as conflicts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": "Overwrite externally changed files. Use only when the user explicitly requests force.",
            },
        },
        "required": [],
    }

    def _execute_sync(self, force: bool = False) -> str:
        tracker = current_change_tracker()
        if not len(tracker):
            return "No tracked file changes to undo."
        result = tracker.undo_all(force=force)
        summary = (
            f"Undo complete: {len(result.restored)} restored, {len(result.deleted)} deleted, "
            f"{len(result.conflicts)} conflicts, {len(result.errors)} errors."
        )
        lines = [summary]
        if result.conflicts:
            lines.append("Conflicts (left unchanged):\n- " + "\n- ".join(result.conflicts))
        if result.errors:
            lines.append("Errors:\n- " + "\n- ".join(result.errors))
        return "\n".join(lines)

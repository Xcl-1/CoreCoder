"""Read-only retrieval tool for externalized context artifacts."""

from __future__ import annotations

from ..context_artifacts import ContextArtifactStore
from .base import Tool


class RetrieveContextTool(Tool):
    name = "retrieve_context"
    input_types = ("artifact_id", "query", "line_range")
    output_type = "artifact_text"
    permission_scope = "context:read"
    side_effect = "none"
    network_access = "none"
    declared_risk = "low"
    description = (
        "Retrieve exact text from a large tool result that was externalized from "
        "the conversation. Use its artifact://sha256/... id, optionally with a "
        "keyword or one-based line range."
    )
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "The artifact://sha256/... id from a tool-result placeholder.",
            },
            "query": {
                "type": "string",
                "description": "Optional case-insensitive keyword; returns matching lines with context.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional one-based first line to retrieve.",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional one-based last line to retrieve.",
            },
            "max_chars": {
                "type": "integer",
                "minimum": 500,
                "maximum": 20000,
                "description": "Maximum returned characters (default 12000).",
            },
        },
        "required": ["artifact_id"],
    }
    def __init__(self, store: ContextArtifactStore):
        self.store = store

    def _execute_sync(
        self,
        artifact_id: str,
        query: str = "",
        start_line: int | None = None,
        end_line: int | None = None,
        max_chars: int = 12_000,
    ) -> str:
        try:
            return self.store.retrieve(
                artifact_id,
                query=query,
                start_line=start_line,
                end_line=end_line,
                max_chars=max_chars,
            )
        except (OSError, ValueError) as exc:
            return f"Error retrieving context artifact: {exc}"

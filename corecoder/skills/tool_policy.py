"""Conservative extraction of explicit tool boundaries from Skill procedures."""

from __future__ import annotations

import re
from dataclasses import dataclass

KNOWN_TOOL_NAMES = frozenset({
    "agent",
    "bash",
    "edit_ast",
    "edit_file",
    "glob",
    "grep",
    "read_file",
    "task_control",
    "undo_changes",
    "write_file",
})

_ALLOW_PATTERNS = (
    re.compile(r"(?:仅允许(?:使用|调用)?|只允许(?:使用|调用)?|允许工具仅限|工具仅限)\s*([^。；;\n]+)"),
    re.compile(
        r"(?:only\s+(?:use|allow)|allowed\s+tools?\s*(?:are|:))\s*([^.;\n]+)",
        re.IGNORECASE,
    ),
)
_FORBID_PATTERNS = (
    re.compile(r"(?:禁止|不得)(?:使用|调用)?\s*([^。；;\n]+)"),
    re.compile(r"(?:do\s+not\s+use|don't\s+use|forbid(?:den)?\s*:?)[ ]*([^.;\n]+)", re.IGNORECASE),
)
_ALIASES = {
    "edit": frozenset({"edit_ast", "edit_file"}),
    "read": frozenset({"read_file"}),
    "shell": frozenset({"bash"}),
    "undo": frozenset({"undo_changes"}),
    "write": frozenset({"write_file"}),
}


@dataclass(frozen=True)
class InferredToolPolicy:
    """Machine-readable boundary explicitly stated in procedure text."""

    required: frozenset[str]
    forbidden: frozenset[str]
    has_allowlist: bool = False

    @property
    def contradictory(self) -> frozenset[str]:
        return self.required & self.forbidden


def infer_tool_policy(text: str) -> InferredToolPolicy:
    """Extract only named tools from explicit allow/forbid clauses."""
    allowed = _names_in_clauses(text, _ALLOW_PATTERNS)
    forbidden = _names_in_clauses(text, _FORBID_PATTERNS)
    has_allowlist = bool(allowed)
    if has_allowlist:
        forbidden.update(KNOWN_TOOL_NAMES - allowed)
    return InferredToolPolicy(
        required=frozenset(allowed),
        forbidden=frozenset(forbidden),
        has_allowlist=has_allowlist,
    )


def _names_in_clauses(text: str, patterns: tuple[re.Pattern[str], ...]) -> set[str]:
    names: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", match.group(1)):
                normalized = token.casefold()
                if normalized in KNOWN_TOOL_NAMES:
                    names.add(normalized)
                else:
                    names.update(_ALIASES.get(normalized, ()))
    return names

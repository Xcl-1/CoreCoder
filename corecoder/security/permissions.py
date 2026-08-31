"""Permission rules — data model and manager.

Rules are loaded from three sources, in priority order (highest first):

1. User-level  ``~/.corecoder/permissions.json``
2. Project-level ``.corecoder/permissions.json``
3. Built-in defaults (see ``defaults.py``)

A rule matches when *both* ``tool_name`` (or ``"*"`` wildcard) and
``pattern`` (regex against the tool's string arguments) agree.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

Action = Literal["allow", "deny", "ask"]


@dataclass
class PermissionRule:
    """A single access-control rule.

    Attributes:
        tool_name: Tool name to match, or ``"*"`` for any tool.
        pattern: Regex tested against the *first* string argument
            of the tool call (``command`` for bash, ``file_path``
            for read/write, etc.).
        action: What to do when this rule matches.
        reason: Human-readable explanation (shown in audit logs
            and confirmation prompts).
        priority: Higher = checked first.  User rules should use
            positive values, project rules 0, built-ins negative.
        source: Where the rule came from — ``"user"``, ``"project"``,
            or ``"builtin"``.
        max_frequency: Maximum calls per minute (``None`` = unlimited).
    """

    tool_name: str
    pattern: str
    action: Action = "ask"
    reason: str = ""
    priority: int = 0
    source: str = "user"
    max_frequency: int | None = None

    _compiled: re.Pattern | None = field(default=None, repr=False, compare=False)

    def compiled(self) -> re.Pattern:
        """Return the compiled regex, caching on first access."""
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)
        return self._compiled

    def matches(self, tool_name: str, arguments: dict) -> bool:
        """Check whether this rule applies to a tool call."""
        # tool name must match exactly, or rule uses wildcard
        if self.tool_name != "*" and self.tool_name != tool_name:
            return False
        # build a searchable string from the arguments
        haystack = _args_to_string(arguments)
        return bool(self.compiled().search(haystack))


def _args_to_string(arguments: dict) -> str:
    """Flatten tool arguments into a single searchable string."""
    parts: list[str] = []
    for value in arguments.values():
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------

USER_PERMISSIONS_DIR = Path.home() / ".corecoder"
USER_PERMISSIONS_PATH = USER_PERMISSIONS_DIR / "permissions.json"
PROJECT_PERMISSIONS_PATH = Path(".corecoder") / "permissions.json"


@dataclass
class PermissionManager:
    """Load, merge and match permission rules from all sources."""

    _user_rules: list[PermissionRule] = field(default_factory=list)
    _project_rules: list[PermissionRule] = field(default_factory=list)
    _builtin_rules: list[PermissionRule] = field(default_factory=list)
    _all_sorted: list[PermissionRule] | None = field(default=None, repr=False)

    def __post_init__(self):
        self.reload()

    # ---- public API -------------------------------------------------------

    def reload(self) -> None:
        """Re-read all rule sources.  Call after editing a config file."""
        from .defaults import builtin_rules

        self._builtin_rules = builtin_rules()
        self._user_rules = _load_json_rules(USER_PERMISSIONS_PATH, "user")
        try:
            same_source = USER_PERMISSIONS_PATH.resolve() == PROJECT_PERMISSIONS_PATH.resolve()
        except OSError:
            same_source = False
        self._project_rules = (
            [] if same_source
            else _load_json_rules(PROJECT_PERMISSIONS_PATH, "project")
        )
        self._all_sorted = None  # invalidate cache

    def match(self, tool_name: str, arguments: dict) -> PermissionRule | None:
        """Return the highest-priority matching rule, or None."""
        for rule in self._sorted():
            if rule.matches(tool_name, arguments):
                return rule
        return None

    def add_user_rule(self, rule: PermissionRule) -> None:
        """Add a rule to the user-level config and persist it."""
        self._user_rules.append(rule)
        self._all_sorted = None
        self._save_user()

    def remove_user_rule(self, index: int) -> bool:
        """Remove a user rule by index.  Returns True if successful."""
        if 0 <= index < len(self._user_rules):
            self._user_rules.pop(index)
            self._all_sorted = None
            self._save_user()
            return True
        return False

    def list_rules(self) -> list[PermissionRule]:
        """Return all rules in priority order (highest first)."""
        return list(self._sorted())

    # ---- internal ---------------------------------------------------------

    def _sorted(self) -> list[PermissionRule]:
        if self._all_sorted is None:
            merged = self._user_rules + self._project_rules + self._builtin_rules
            merged.sort(key=lambda r: r.priority, reverse=True)
            self._all_sorted = merged
        return self._all_sorted

    def _save_user(self) -> None:
        USER_PERMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        data = [_rule_to_dict(r) for r in self._user_rules]
        USER_PERMISSIONS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# JSON persistence helpers
# ---------------------------------------------------------------------------

def _load_json_rules(path: Path, source: str) -> list[PermissionRule]:
    """Load rules from a JSON file.  Returns empty list on any error."""
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [_rule_from_dict(item, source) for item in data]
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        pass
    return []


def _rule_to_dict(rule: PermissionRule) -> dict:
    return {
        "tool_name": rule.tool_name,
        "pattern": rule.pattern,
        "action": rule.action,
        "reason": rule.reason,
        "priority": rule.priority,
        "max_frequency": rule.max_frequency,
    }


def _rule_from_dict(data: dict, source: str) -> PermissionRule:
    return PermissionRule(
        tool_name=data.get("tool_name", "*"),
        pattern=data.get("pattern", ".*"),
        action=data.get("action", "ask"),
        reason=data.get("reason", ""),
        priority=data.get("priority", 0),
        source=source,
        max_frequency=data.get("max_frequency"),
    )

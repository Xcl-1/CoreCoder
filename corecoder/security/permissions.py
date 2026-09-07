"""Permission rules — data model and manager.

Rules are loaded from four sources, in priority order (highest first):

1. Ephemeral session rules (never persisted)
2. User-level  ``~/.corecoder/permissions.json``
3. Project-level ``.corecoder/permissions.json``
4. Built-in defaults (see ``defaults.py``)

A rule matches when *both* ``tool_name`` (or ``"*"`` wildcard) and
``pattern`` (regex against the tool's string arguments) agree.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

Action = Literal["allow", "deny", "ask"]
_NESTED_REPEAT_RE = re.compile(
    r"\((?:[^()]|\\.)*[+*](?:[^()]|\\.)*\)\s*(?:[+*]|\{\d)",
)


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
    hard_boundary: bool = False

    _compiled: re.Pattern | None = field(default=None, repr=False, compare=False)

    @property
    def rule_id(self) -> str:
        """Stable, non-secret identifier suitable for display and revocation."""
        payload = json.dumps(
            [
                self.source,
                self.tool_name,
                self.pattern,
                self.action,
                self.reason,
                self.priority,
                self.max_frequency,
                self.hard_boundary,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        prefix = {
            "session": "ses",
            "user": "usr",
            "project": "prj",
            "builtin": "sys",
        }.get(self.source, "rul")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}-{digest}"

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
    text = " ".join(parts)
    # Normalize compatibility glyphs and strip format controls commonly used
    # to split dangerous keywords without changing their visual appearance.
    text = unicodedata.normalize("NFKC", text)
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


# ---------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------

USER_PERMISSIONS_DIR = Path.home() / ".corecoder"
USER_PERMISSIONS_PATH = USER_PERMISSIONS_DIR / "permissions.json"
PROJECT_PERMISSIONS_PATH = Path(".corecoder") / "permissions.json"


@dataclass
class PermissionManager:
    """Load, merge and match permission rules from all sources."""

    _session_rules: list[PermissionRule] = field(default_factory=list)
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
        # Hard built-in denials are platform safety boundaries, not defaults.
        # User/project allow rules may tune ordinary policy but cannot override
        # a root deletion, raw-disk write, or equivalent invariant.
        for rule in self._builtin_rules:
            if rule.hard_boundary and rule.action == "deny" and rule.matches(tool_name, arguments):
                return rule
        for rule in self._sorted():
            if rule.matches(tool_name, arguments):
                return rule
        return None

    def add_user_rule(self, rule: PermissionRule) -> None:
        """Add a rule to the user-level config and persist it."""
        _validate_rule(rule)
        rule.source = "user"
        rule.hard_boundary = False
        self._user_rules.append(rule)
        self._all_sorted = None
        self._save_user()

    def add_session_rule(self, rule: PermissionRule) -> None:
        """Add an in-memory rule that expires when the process exits."""
        _validate_rule(rule)
        rule.source = "session"
        rule.hard_boundary = False
        self._session_rules.append(rule)
        self._all_sorted = None

    def remove_user_rule(self, index: int | str) -> bool:
        """Remove a user rule by legacy index or stable rule ID."""
        resolved = index
        if isinstance(index, str):
            resolved = next(
                (position for position, rule in enumerate(self._user_rules) if rule.rule_id == index),
                -1,
            )
        if isinstance(resolved, int) and not isinstance(resolved, bool) and 0 <= resolved < len(self._user_rules):
            self._user_rules.pop(resolved)
            self._all_sorted = None
            self._save_user()
            return True
        return False

    def list_rules(self, source: str | None = None) -> list[PermissionRule]:
        """Return rules in priority order, optionally restricted by source."""
        rules = self._sorted()
        if source is None:
            return list(rules)
        if source not in {"session", "user", "project", "builtin"}:
            raise ValueError("permission source must be session, user, project, or builtin")
        return [rule for rule in rules if rule.source == source]

    def find_rule(self, rule_id: str) -> PermissionRule | None:
        """Find a rule from any source by its stable identifier."""
        return next((rule for rule in self._sorted() if rule.rule_id == rule_id), None)

    def revoke_rule(self, rule_id: str) -> PermissionRule | None:
        """Remove a mutable user/session rule; project and built-ins are immutable here."""
        for collection, persistent in ((self._session_rules, False), (self._user_rules, True)):
            for index, rule in enumerate(collection):
                if rule.rule_id != rule_id:
                    continue
                removed = collection.pop(index)
                self._all_sorted = None
                if persistent:
                    self._save_user()
                return removed
        return None

    def clear_session_rules(self) -> list[PermissionRule]:
        """Remove and return every process-local permission rule."""
        removed = list(self._session_rules)
        self._session_rules.clear()
        self._all_sorted = None
        return removed

    # ---- internal ---------------------------------------------------------

    def _sorted(self) -> list[PermissionRule]:
        if self._all_sorted is None:
            merged = self._session_rules + self._user_rules + self._project_rules + self._builtin_rules
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
                rules: list[PermissionRule] = []
                for item in data:
                    try:
                        rule = _rule_from_dict(item, source)
                        _validate_rule(rule)
                        rules.append(rule)
                    except (KeyError, TypeError, ValueError, re.error):
                        continue
                return rules
    except (json.JSONDecodeError, OSError, TypeError):
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
    if not isinstance(data, dict):
        raise TypeError("permission rule must be an object")
    return PermissionRule(
        tool_name=data.get("tool_name", "*"),
        pattern=data.get("pattern", ".*"),
        action=data.get("action", "ask"),
        reason=data.get("reason", ""),
        priority=data.get("priority", 0),
        source=source,
        max_frequency=data.get("max_frequency"),
        # Configuration files cannot mint hard platform boundaries.
        hard_boundary=False,
    )


def _validate_rule(rule: PermissionRule) -> None:
    """Reject malformed configuration instead of accidentally treating it as allow."""
    if not isinstance(rule.tool_name, str) or not rule.tool_name:
        raise ValueError("tool_name must be a non-empty string")
    if not isinstance(rule.pattern, str) or len(rule.pattern) > 4096:
        raise ValueError("pattern must be a string of at most 4096 characters")
    if _NESTED_REPEAT_RE.search(rule.pattern):
        raise ValueError("pattern contains a potentially catastrophic nested repeat")
    if rule.action not in {"allow", "deny", "ask"}:
        raise ValueError("action must be allow, deny, or ask")
    if not isinstance(rule.reason, str):
        raise TypeError("reason must be a string")
    if not isinstance(rule.priority, int) or isinstance(rule.priority, bool):
        raise TypeError("priority must be an integer")
    if (
        rule.max_frequency is not None
        and (
            not isinstance(rule.max_frequency, int)
            or isinstance(rule.max_frequency, bool)
            or rule.max_frequency <= 0
        )
    ):
        raise ValueError("max_frequency must be a positive integer")
    rule.compiled()

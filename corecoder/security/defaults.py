"""Built-in default permission rules.

These are the lowest-priority rules — user and project configs
override them.  The dangerous-command patterns that used to live
in ``tools/bash.py`` are now defined here so they can be
inspected and extended without touching tool code.
"""

import re

# ---------------------------------------------------------------------------
# dangerous shell patterns (migrated from tools/bash.py)
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # recursive delete aimed at root/home (force flag optional)
    (r"\brm\s+(-\w*)?-r\w*\s+(/|~|\$HOME)", "recursive delete on home/root"),
    # recursive (-r/-R) and force (-f) flags together, in any order or spacing
    (r"\brm\b(?=(?:.*\s)?-\w*[rR])(?=(?:.*\s)?-\w*f)", "force recursive delete"),
    # the same, written with long-form flags
    (r"\brm\b.*--recursive\b.*--force\b|\brm\b.*--force\b.*--recursive\b", "force recursive delete"),
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*of=/dev/", "raw disk write"),
    (r">\s*/dev/sd[a-z]", "overwrite block device"),
    (r"\bchmod\s+(-R\s+)?777\s+/", "chmod 777 on root"),
    (r":\(\)\s*\{.*:\|:.*\}", "fork bomb"),
    (r"\bcurl\b.*\|\s*(sudo\s+)?(ba)?sh\b", "pipe curl to shell"),
    (r"\bwget\b.*\|\s*(sudo\s+)?(ba)?sh\b", "pipe wget to shell"),
]

# Guard-only rules. These are intentionally separate from check_dangerous(),
# which is also used by BashTool when no Guard is configured.
_GUARD_ONLY_DENY_PATTERNS: list[tuple[str, str]] = [
    (r"(?:\r|\n|>>?|[|;&])", "shell chaining or redirection"),
]


def check_dangerous(cmd: str) -> str | None:
    """Return a warning string if *cmd* looks destructive, else None.

    This is the canonical dangerous-command check, used by both
    ``BashTool`` (for backward-compatible operation without a Guard)
    and ``Guard`` (as built-in deny rules).
    """
    for pattern, reason in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return reason
    return None

# ---------------------------------------------------------------------------
# safe shell commands — allow without confirmation
# ---------------------------------------------------------------------------

_SAFE_SHELL_PATTERNS: list[str] = [
    r"^(ls|dir)(?:\s+[^<>|;&]*)?$",
    r"^(cat|head|tail|less|more)(?:\s+[^<>|;&]*)?$",
    r"^(pwd|whoami|date|printenv)(?:\s+[^<>|;&]*)?$",
    r"^env\s*$",
    r"^cd(?:\s+[^<>|;&]*)?$",
    (
        r"^git(?:\s+-C\s+\S+)?\s+"
        r"(?!.*(?:--output(?:=|\s)|--ext-diff|--textconv))"
        r"(?:status|diff|log|show|rev-parse|ls-files)(?:\s+[^<>|;&]*)?$"
    ),
    r"^hg\s+(?:status|diff|log|cat|id)(?:\s+[^<>|;&]*)?$",
    r"^svn\s+(?:status|diff|log|info|list|cat)(?:\s+[^<>|;&]*)?$",
    r"^(python|python3|pip|pip3)\s+(--version|-V|--help|-h)\s*$",
    r"^(node|npm|npx|yarn|pnpm)\s+(--version|-v|--help|-h)\s*$",
    r"^(wc|df|du)(?:\s+[^<>|;&]*)?$",
]


def builtin_rules():
    """Return the list of built-in ``PermissionRule`` instances.

    These are the *lowest* priority — user and project configs
    override them.  The list is built on-demand so the import
    side-effect is minimal.
    """
    from .permissions import PermissionRule

    rules: list[PermissionRule] = []

    # ---- dangerous shell commands — always deny ----
    for pattern, reason in _DANGEROUS_PATTERNS + _GUARD_ONLY_DENY_PATTERNS:
        rules.append(PermissionRule(
            tool_name="bash",
            pattern=pattern,
            action="deny",
            reason=reason,
            priority=-10,
            source="builtin",
        ))

    # ---- safe shell commands — allow without asking ----
    for pattern in _SAFE_SHELL_PATTERNS:
        rules.append(PermissionRule(
            tool_name="bash",
            pattern=pattern,
            action="allow",
            reason="safe shell command — no side effects or read-only",
            priority=-10,
            source="builtin",
        ))

    # ---- read tools — always allow ----
    for name in ("read_file", "grep", "glob"):
        rules.append(PermissionRule(
            tool_name=name,
            pattern=r".*",
            action="allow",
            reason="read-only tool — always safe",
            priority=-10,
            source="builtin",
        ))

    # ---- write tools — allow (sandbox checks separately) ----
    for name in ("write_file", "edit_file", "edit_ast"):
        rules.append(PermissionRule(
            tool_name=name,
            pattern=r".*",
            action="allow",
            reason="write tool — sandbox policy applies separately",
            priority=-10,
            source="builtin",
        ))

    # ---- undo — destructive restoration requires explicit confirmation ----
    rules.append(PermissionRule(
        tool_name="undo_changes",
        pattern=r".*",
        action="ask",
        reason="undo restores or deletes files changed during this session",
        priority=-10,
        source="builtin",
    ))

    # ---- agent tool — allow ----
    rules.append(PermissionRule(
        tool_name="agent",
        pattern=r".*",
        action="allow",
        reason="sub-agent — inherits parent guard",
        priority=-10,
        source="builtin",
    ))

    return rules

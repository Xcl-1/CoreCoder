"""Guard — multi-layer security review chain.

Wire this into the agent loop so every tool call passes through::

    1. Static rule matching   (PermissionManager)
    2. User confirmation      (optional callback)
    3. Frequency throttle     (per-tool calls/minute)
    4. Audit logging          (every decision recorded)
    5. Output sanitisation    (redact secrets from results)
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .audit import AuditEntry, AuditLogger
from .permissions import PermissionManager, PermissionRule

# ---------------------------------------------------------------------------
# sensitive-pattern redaction (Layer 5)
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    (r"sk-[a-zA-Z0-9]{20,}", "[OPENAI_KEY_REDACTED]"),
    (r"AKIA[0-9A-Z]{16}", "[AWS_KEY_REDACTED]"),
    (r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+", "[JWT_REDACTED]"),
    (
        r"-----BEGIN .*PRIVATE KEY-----.*?-----END .*PRIVATE KEY-----",
        "[PRIVATE_KEY_REDACTED]",
    ),
    (
        r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"][^'\"]+['\"]",
        "[SECRET_REDACTED]",
    ),
]
_PERSISTENT_PERMISSION_RE = re.compile(
    r"(?:^|[\\/])\.corecoder[\\/]permissions\.json(?:[\"']|\s|$)",
    re.IGNORECASE,
)


@dataclass
class SecurityDecision:
    """Result of the Guard review chain."""

    allowed: bool
    reason: str
    rule: PermissionRule | None = None
    user_confirmed: bool = False


# type alias for the confirm callback
ConfirmCallback = Callable[[str, dict, str], bool | None]
#  callback(tool_name, arguments, reason) → True=allow, False=deny, None=cancel


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


@dataclass
class Guard:
    """Multi-layer security review chain.

    Args:
        permissions: Optional pre-configured PermissionManager.
        audit_logger: Optional pre-configured AuditLogger.
        confirm_callback: Optional callback for interactive confirmation.
            When ``None`` (the default), ``ask`` rules are treated as
            ``deny`` — safe for non-interactive / CI usage.
        max_frequency_window: Time window in seconds for rate limiting
            (default 60 = 1 minute).
    """

    permissions: PermissionManager = field(default_factory=PermissionManager)
    audit: AuditLogger = field(default_factory=AuditLogger)
    confirm_callback: ConfirmCallback | None = None
    max_frequency_window: float = 60.0

    # per-tool call timestamps for frequency throttle (Layer 3)
    _freq_log: dict[str, list[float]] = field(default_factory=dict, repr=False)

    # ---- public API -------------------------------------------------------

    def review(self, tool_name: str, arguments: dict) -> SecurityDecision:
        """Run Layers 1–4 and return a decision.

        The caller should check ``decision.allowed`` before executing
        the tool, and call ``sanitize()`` on the result afterwards.
        """
        rule = self.permissions.match(tool_name, arguments)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        # ---- Layer 1: static rule matching ----
        if rule is None:
            # No rule matched — should not happen (builtins cover everything),
            # but default to deny for safety.
            decision = SecurityDecision(allowed=False, reason="no matching rule")
            self._audit(ts, tool_name, arguments, "deny", "builtin", decision.reason)
            return decision

        # A deny decision always wins, including when a dangerous command also
        # happens to mention a permission file.
        if rule.action == "deny":
            self._audit(ts, tool_name, arguments, "deny", rule.source, rule.reason)
            return SecurityDecision(allowed=False, reason=rule.reason, rule=rule)

        user_confirmed = False
        if _targets_persistent_permissions(tool_name, arguments):
            reason = "persistent permission changes require explicit user approval"
            if self.confirm_callback is None:
                self._audit(ts, tool_name, arguments, "deny", "builtin", f"{reason} — requires confirmation")
                return SecurityDecision(allowed=False, reason=f"{reason} (requires confirmation)", rule=rule)
            choice = self.confirm_callback(tool_name, arguments, reason)
            if choice is not True:
                suffix = "user denied" if choice is False else "user cancelled"
                self._audit(ts, tool_name, arguments, "deny", "builtin", f"{reason} — {suffix}")
                return SecurityDecision(allowed=False, reason=f"{reason} ({suffix})", rule=rule)
            user_confirmed = True

        # ---- Layer 2: user confirmation for 'ask' rules ----
        if rule.action == "ask":
            if self.confirm_callback is not None:
                user_choice = self.confirm_callback(tool_name, arguments, rule.reason)
                if user_choice is True:
                    user_confirmed = True
                elif user_choice is False:
                    self._audit(ts, tool_name, arguments, "deny", rule.source,
                                f"{rule.reason} — user denied confirmation")
                    return SecurityDecision(allowed=False, reason=f"{rule.reason} (user denied)", rule=rule)
                else:
                    # None = cancelled
                    self._audit(ts, tool_name, arguments, "deny", rule.source,
                                "user cancelled confirmation")
                    return SecurityDecision(allowed=False, reason="user cancelled", rule=rule)
            else:
                # no callback — ask degrades to deny in non-interactive mode
                self._audit(ts, tool_name, arguments, "deny", rule.source,
                            f"{rule.reason} — requires confirmation (non-interactive mode)")
                return SecurityDecision(allowed=False, reason=f"{rule.reason} (requires confirmation)", rule=rule)

        # ---- Layer 3: frequency throttle ----
        freq_ok = True
        if rule.max_frequency is not None:
            freq_ok = self._check_frequency(tool_name, rule.max_frequency)

        self._audit(ts, tool_name, arguments, "allow", rule.source, rule.reason,
                    freq_checked=rule.max_frequency is not None, freq_passed=freq_ok,
                    user_confirmed=user_confirmed)

        if not freq_ok:
            return SecurityDecision(
                allowed=False,
                reason=f"rate limit exceeded ({rule.max_frequency}/min)",
                rule=rule,
            )

        # ---- Layer 4: audit already written above ----
        return SecurityDecision(
            allowed=True,
            reason=rule.reason,
            rule=rule,
            user_confirmed=user_confirmed,
        )

    def sanitize(self, text: str) -> str:
        """Layer 5: redact secrets from tool output."""
        for pattern, replacement in _SENSITIVE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def request_confirmation(
        self,
        tool_name: str,
        arguments: dict,
        reason: str,
        source: str = "runtime",
    ) -> SecurityDecision:
        """Request and audit an additional runtime risk confirmation."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        if self.confirm_callback is None:
            self._audit(ts, tool_name, arguments, "deny", source, f"{reason} — requires confirmation")
            return SecurityDecision(False, f"{reason} (requires confirmation)")
        choice = self.confirm_callback(tool_name, arguments, reason)
        if choice is True:
            self._audit(
                ts,
                tool_name,
                arguments,
                "allow",
                source,
                reason,
                user_confirmed=True,
            )
            return SecurityDecision(True, reason, user_confirmed=True)
        suffix = "user denied" if choice is False else "user cancelled"
        self._audit(ts, tool_name, arguments, "deny", source, f"{reason} — {suffix}")
        return SecurityDecision(False, f"{reason} ({suffix})")

    # ---- internal ---------------------------------------------------------

    def _audit(self, timestamp: str, tool_name: str, arguments: dict,
               decision: str, source: str, reason: str,
               freq_checked: bool = False, freq_passed: bool = True,
               user_confirmed: bool = False) -> None:
        self.audit.log(AuditEntry(
            timestamp=timestamp,
            tool_name=tool_name,
            arguments_summary=_summarise(tool_name, arguments),
            decision=decision,
            rule_source=source,
            reason=reason,
            user_confirmed=user_confirmed,
            frequency_checked=freq_checked,
            frequency_passed=freq_passed,
        ))

    def _check_frequency(self, tool_name: str, max_per_minute: int) -> bool:
        """Return True if the call is within the rate limit."""
        now = time.monotonic()
        window_start = now - self.max_frequency_window

        # prune old timestamps
        times = self._freq_log.get(tool_name, [])
        times = [t for t in times if t > window_start]
        self._freq_log[tool_name] = times

        if len(times) >= max_per_minute:
            return False  # rate limit exceeded

        times.append(now)
        return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _summarise(tool_name: str, arguments: dict, max_len: int = 200) -> str:
    """Build a short human-readable summary of a tool call."""
    if tool_name == "bash":
        cmd = arguments.get("command", "")
        return f"bash: {cmd[:max_len]}"
    file_path = arguments.get("file_path", "")
    if file_path:
        return f"{tool_name}: {file_path[:max_len]}"
    # generic fallback
    text = " ".join(str(v)[:80] for v in arguments.values())
    return f"{tool_name}: {text[:max_len]}"


def _targets_persistent_permissions(tool_name: str, arguments: dict) -> bool:
    if tool_name not in ("write_file", "edit_file", "edit_ast", "bash"):
        return False
    return any(
        isinstance(value, str) and _PERSISTENT_PERMISSION_RE.search(value)
        for value in arguments.values()
    )

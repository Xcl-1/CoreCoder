"""Guard — multi-layer security review chain.

Wire this into the agent loop so every tool call passes through::

    1. Static hard boundaries (PermissionManager)
    2. Tool capability check  (declared authority vs. arguments)
    3. Risk classification    (deterministic floor + optional AI reviewer)
    4. Human confirmation     (high-risk and ``ask`` decisions)
    5. Audit and output guard (redaction, provenance, injection signals)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditEntry, AuditLogger
from .capabilities import CapabilityReport, inspect_tool_capabilities
from .content import inspect_content, label_untrusted_content
from .network import NetworkDecision, NetworkPolicy
from .permissions import PermissionManager, PermissionRule
from .risk import (
    RiskAssessment,
    RiskLevel,
    RiskReviewer,
    deterministic_risk,
    merge_reviewer_assessment,
)

# ---------------------------------------------------------------------------
# sensitive-pattern redaction (Layer 5)
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    (r"sk-[a-zA-Z0-9]{20,}", "[OPENAI_KEY_REDACTED]"),
    (r"AKIA[0-9A-Z]{16}", "[AWS_KEY_REDACTED]"),
    (r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+", "[JWT_REDACTED]"),
    (
        r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
        "[PRIVATE_KEY_REDACTED]",
    ),
    (
        r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"][^'\"]+['\"]",
        "[SECRET_REDACTED]",
    ),
    (
        (
            r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*"
            r"(?!\[[A-Z_]+REDACTED\])[^\s,;]+"
        ),
        "[SECRET_REDACTED]",
    ),
    (r"authorization\s*:\s*bearer\s+[^\s,;]+", "[BEARER_TOKEN_REDACTED]"),
    (
        r"(?P<scheme>\b(?:https?|ftp|ssh|git)://)[^/\s:@]+:[^/\s@]+@",
        r"\g<scheme>[URL_CREDENTIALS_REDACTED]@",
    ),
    (r"(?:(?:-u|--user)(?:\s+|=))[^\s]+", "[AUTH_OPTION_REDACTED]"),
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
    risk: RiskAssessment | None = None
    capability: CapabilityReport | None = None
    network: NetworkDecision | None = None


@dataclass(frozen=True)
class ConfirmationContext:
    """Structured facts shown to a human before a guarded operation runs."""

    risk_level: str = ""
    risk_reasons: tuple[str, ...] = ()
    capability_scope: str = ""
    side_effect: str = ""
    declared_risk: str = ""
    rule_source: str = ""
    network_action: str = ""
    network_destinations: tuple[str, ...] = ()
    network_mutating: bool = False
    follows_redirects: bool = False
    carries_credentials: bool = False
    can_remember: bool = False


class _DiscardAudit:
    """Audit sink used for non-executing policy previews."""

    def log(self, _entry: AuditEntry) -> None:
        return


# type alias for the confirm callback
ConfirmCallback = Callable[..., bool | None]
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
        network_policy: Destination allowlist and unknown-egress behavior.
        max_frequency_window: Time window in seconds for rate limiting
            (default 60 = 1 minute).
    """

    permissions: PermissionManager = field(default_factory=PermissionManager)
    audit: AuditLogger = field(default_factory=AuditLogger)
    confirm_callback: ConfirmCallback | None = None
    risk_reviewer: RiskReviewer | None = None
    network_policy: NetworkPolicy = field(default_factory=NetworkPolicy)
    max_frequency_window: float = 60.0
    agent_id: str = ""
    parent_id: str = ""
    task_id: str = ""
    permission_scope: str = ""
    workspace_mode: str = ""

    # per-tool call timestamps for frequency throttle (Layer 3)
    _freq_log: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _content_risk_signals: list[str] = field(default_factory=list, repr=False)

    # ---- public API -------------------------------------------------------

    def begin_turn(self) -> None:
        """Reset transient content taint at the start of a user turn."""
        self._content_risk_signals.clear()

    def for_delegate(
        self,
        *,
        agent_id: str,
        parent_id: str,
        task_id: str,
        permission_scope: str,
        workspace_mode: str,
    ) -> Guard:
        """Return a non-interactive, no-broader-authority child policy view.

        Permission, network and risk policies are inherited by reference.  The
        confirmation callback is intentionally removed: a child can report that
        approval is needed, but only the parent control plane may ask the user.
        Frequency counters remain shared so delegation cannot evade rate limits.
        """
        delegated = Guard(
            permissions=self.permissions,
            audit=self.audit,
            confirm_callback=None,
            risk_reviewer=self.risk_reviewer,
            network_policy=self.network_policy,
            max_frequency_window=self.max_frequency_window,
            agent_id=agent_id,
            parent_id=parent_id,
            task_id=task_id,
            permission_scope=permission_scope,
            workspace_mode=workspace_mode,
        )
        delegated._freq_log = self._freq_log
        delegated._content_risk_signals = list(self._content_risk_signals)
        return delegated

    def review(self, tool_name: str, arguments: dict, tool: Any | None = None) -> SecurityDecision:
        """Run the pre-execution review chain and return a decision.

        The caller should check ``decision.allowed`` before executing
        the tool. Passing the tool instance enables capability self-checks;
        the optional argument keeps the public API backward compatible.
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

        # ---- Layer 2: declared tool capability self-check ----
        capability = inspect_tool_capabilities(tool, arguments) if tool is not None else None
        if capability is not None and not capability.valid:
            self._audit(ts, tool_name, arguments, "deny", "capability", capability.reason,
                        capability=capability)
            return SecurityDecision(False, capability.reason, rule=rule, capability=capability)

        # ---- Layer 3: network intent and destination policy ----
        network = self.network_policy.review(tool_name, arguments, capability)
        if network.action == "deny":
            self._audit(ts, tool_name, arguments, "deny", "network", network.reason,
                        capability=capability, network=network)
            return SecurityDecision(
                False,
                network.reason,
                rule=rule,
                capability=capability,
                network=network,
            )

        # ---- Layer 4: deterministic risk floor + optional semantic reviewer ----
        risk = deterministic_risk(tool_name, arguments, capability)
        if network.intent.mutating and risk.level < RiskLevel.HIGH:
            risk = RiskAssessment(
                RiskLevel.HIGH,
                (*risk.reasons, "network operation may mutate remote state"),
                "deterministic-network",
            )
        if self.risk_reviewer is not None:
            try:
                review_arguments = self._sanitize_value(arguments)
                reviewed = self.risk_reviewer(tool_name, review_arguments, risk)
                if not isinstance(reviewed, RiskAssessment):
                    raise TypeError("risk reviewer must return RiskAssessment")
                if (
                    not isinstance(reviewed.level, RiskLevel)
                    or not isinstance(reviewed.reasons, tuple)
                    or any(not isinstance(item, str) for item in reviewed.reasons)
                ):
                    raise TypeError("risk reviewer returned an invalid assessment")
                risk = merge_reviewer_assessment(risk, reviewed)
            except Exception as exc:  # noqa: BLE001 - third-party/AI reviewer boundary
                if risk.level >= RiskLevel.MEDIUM:
                    risk = RiskAssessment(
                        RiskLevel.HIGH,
                        (*risk.reasons, f"risk reviewer unavailable: {type(exc).__name__}"),
                        "reviewer-fail-closed",
                    )
        if risk.level >= RiskLevel.CRITICAL:
            reason = "; ".join(risk.reasons) or "critical-risk operation"
            self._audit(ts, tool_name, arguments, "deny", "risk", reason,
                        risk=risk, capability=capability, network=network)
            return SecurityDecision(
                False, reason, rule=rule, risk=risk, capability=capability, network=network
            )

        # ---- Layer 5: collect all confirmation causes and prompt only once ----
        confirmation_reasons: list[str] = []
        if _targets_persistent_permissions(tool_name, arguments):
            confirmation_reasons.append("persistent permission changes require explicit user approval")
        if tool_name == "agent" and arguments.get("durable") is True:
            confirmation_reasons.append(
                "durable delegation encrypts and persists the task objective and context"
            )
        if rule.action == "ask":
            confirmation_reasons.append(rule.reason or "permission rule requires confirmation")
        if risk.level >= RiskLevel.HIGH:
            confirmation_reasons.extend(risk.reasons or ("high-risk operation",))
        if network.action == "ask":
            confirmation_reasons.append(network.reason)
        if (
            self._content_risk_signals
            and capability is not None
            and capability.side_effect != "none"
        ):
            findings = ", ".join(dict.fromkeys(self._content_risk_signals))
            confirmation_reasons.append(
                "state-changing action follows prompt-injection signals in untrusted "
                f"content from this turn: {findings}"
            )

        user_confirmed = False
        if confirmation_reasons:
            reason = "; ".join(dict.fromkeys(confirmation_reasons))
            if self.confirm_callback is None:
                final_reason = f"{reason} (requires confirmation)"
                self._audit(ts, tool_name, arguments, "deny", "confirmation", final_reason,
                            risk=risk, capability=capability, network=network)
                return SecurityDecision(
                    False, final_reason, rule, risk=risk, capability=capability, network=network
                )
            context = ConfirmationContext(
                risk_level=risk.level.label,
                risk_reasons=risk.reasons,
                capability_scope=capability.scope if capability else "",
                side_effect=capability.side_effect if capability else "",
                declared_risk=capability.declared_risk if capability else "",
                rule_source=rule.source,
                network_action=network.action,
                network_destinations=network.intent.destinations,
                network_mutating=network.intent.mutating,
                follows_redirects=network.intent.follows_redirects,
                carries_credentials=network.intent.carries_credentials,
                can_remember=(
                    rule.action == "ask"
                    and not _targets_persistent_permissions(tool_name, arguments)
                    and not (tool_name == "agent" and arguments.get("durable") is True)
                    and risk.level < RiskLevel.HIGH
                    and network.action != "ask"
                    and not self._content_risk_signals
                ),
            )
            choice = self._invoke_confirmation(tool_name, arguments, reason, context)
            if choice is not True:
                suffix = "user denied" if choice is False else "user cancelled"
                final_reason = f"{reason} ({suffix})"
                self._audit(ts, tool_name, arguments, "deny", "confirmation", final_reason,
                            risk=risk, capability=capability, network=network)
                return SecurityDecision(
                    False, final_reason, rule, risk=risk, capability=capability, network=network
                )
            user_confirmed = True

        # Frequency throttle applies after approval, immediately before execution.
        freq_ok = True
        if rule.max_frequency is not None:
            freq_ok = self._check_frequency(tool_name, rule.max_frequency)
        if not freq_ok:
            reason = f"rate limit exceeded ({rule.max_frequency}/min)"
            self._audit(ts, tool_name, arguments, "deny", rule.source, reason,
                        freq_checked=True, freq_passed=False, user_confirmed=user_confirmed,
                        risk=risk, capability=capability, network=network)
            return SecurityDecision(
                allowed=False,
                reason=reason,
                rule=rule,
                user_confirmed=user_confirmed,
                risk=risk,
                capability=capability,
                network=network,
            )

        self._audit(ts, tool_name, arguments, "allow", rule.source, rule.reason,
                    freq_checked=rule.max_frequency is not None, freq_passed=True,
                    user_confirmed=user_confirmed, risk=risk, capability=capability,
                    network=network)
        return SecurityDecision(
            allowed=True,
            reason=rule.reason,
            rule=rule,
            user_confirmed=user_confirmed,
            risk=risk,
            capability=capability,
            network=network,
        )

    def sanitize(self, text: str) -> str:
        """Redact common secret formats from tool output and audit summaries."""
        for pattern, replacement in _SENSITIVE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.DOTALL)
        return text

    def _sanitize_value(self, value: Any) -> Any:
        """Recursively redact values before crossing a reviewer boundary."""
        if isinstance(value, str):
            return self.sanitize(value)
        if isinstance(value, dict):
            return {key: self._sanitize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_value(item) for item in value)
        return value

    def inspect_output(self, tool_name: str, text: str, *, untrusted: bool = True) -> str:
        """Sanitize, label provenance, and flag prompt-injection indicators."""
        sanitized = self.sanitize(text)
        inspection = inspect_content(sanitized, tool_name, untrusted=untrusted)
        if inspection.signals:
            self._content_risk_signals.extend(inspection.signals)
            reason = "; ".join(inspection.signals)
            self._audit(
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                tool_name,
                {},
                "flag",
                "content-inspection",
                reason,
                risk=RiskAssessment(RiskLevel.HIGH, inspection.signals, "content-inspection"),
            )
        return label_untrusted_content(sanitized, inspection)

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
        context = ConfirmationContext(
            risk_level="high",
            risk_reasons=(reason,),
            rule_source=source,
            can_remember=False,
        )
        choice = self._invoke_confirmation(tool_name, arguments, reason, context)
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

    def explain(
        self,
        tool_name: str,
        arguments: dict,
        tool: Any | None = None,
    ) -> SecurityDecision:
        """Preview the effective decision without prompting, auditing, or executing."""
        preview = Guard(
            permissions=self.permissions,
            audit=_DiscardAudit(),  # type: ignore[arg-type]
            confirm_callback=None,
            risk_reviewer=self.risk_reviewer,
            network_policy=self.network_policy,
            max_frequency_window=self.max_frequency_window,
        )
        preview._content_risk_signals = list(self._content_risk_signals)
        return preview.review(tool_name, arguments, tool=tool)

    def record_permission_change(
        self,
        operation: str,
        rule: PermissionRule | None = None,
        detail: str = "",
    ) -> None:
        """Append an audit event for an explicit permission-policy mutation."""
        rule_id = rule.rule_id if rule else "multiple"
        description = detail or operation
        if rule is not None:
            description = (
                f"{operation}: {rule.action} {rule.tool_name} rule "
                f"{rule.rule_id} ({rule.source})"
            )
        self.audit.log(AuditEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            tool_name="permissions",
            arguments_summary=f"{operation}: {rule_id}",
            arguments_digest=_arguments_digest({"operation": operation, "rule_id": rule_id}),
            decision="policy",
            rule_source="user-action",
            reason=self.sanitize(description),
            agent_id=self.agent_id,
            parent_id=self.parent_id,
            task_id=self.task_id,
            permission_scope=self.permission_scope,
            workspace_mode=self.workspace_mode,
        ))

    def _invoke_confirmation(
        self,
        tool_name: str,
        arguments: dict,
        reason: str,
        context: ConfirmationContext,
    ) -> bool | None:
        """Call legacy three-argument and structured callbacks compatibly."""
        callback = self.confirm_callback
        if callback is None:
            return None
        try:
            inspect.signature(callback).bind(tool_name, arguments, reason, context)
        except (TypeError, ValueError):
            return callback(tool_name, arguments, reason)
        return callback(tool_name, arguments, reason, context)

    # ---- internal ---------------------------------------------------------

    def _audit(self, timestamp: str, tool_name: str, arguments: dict,
               decision: str, source: str, reason: str,
               freq_checked: bool = False, freq_passed: bool = True,
               user_confirmed: bool = False,
               risk: RiskAssessment | None = None,
               capability: CapabilityReport | None = None,
               network: NetworkDecision | None = None) -> None:
        self.audit.log(AuditEntry(
            timestamp=timestamp,
            tool_name=tool_name,
            arguments_summary=self.sanitize(_summarise(tool_name, arguments)),
            arguments_digest=_arguments_digest(arguments),
            decision=decision,
            rule_source=source,
            reason=reason,
            user_confirmed=user_confirmed,
            frequency_checked=freq_checked,
            frequency_passed=freq_passed,
            risk_level=risk.level.label if risk else "",
            risk_reasons=list(risk.reasons) if risk else [],
            capability_scope=capability.scope if capability else "",
            network_access=capability.network_access if capability else "",
            declared_risk=capability.declared_risk if capability else "",
            network_policy_action=network.action if network else "",
            network_destinations=list(network.intent.destinations) if network else [],
            agent_id=self.agent_id,
            parent_id=self.parent_id,
            task_id=self.task_id,
            permission_scope=self.permission_scope,
            workspace_mode=self.workspace_mode,
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
        cmd = str(arguments.get("command", "")).strip()
        executable = cmd.split(None, 1)[0][:60] if cmd else "(empty)"
        return f"bash executable: {executable}"
    file_path = arguments.get("file_path", "")
    if file_path:
        return f"{tool_name}: {file_path[:max_len]}"
    # Generic payloads can contain arbitrary nested secrets. Preserve shape,
    # not values; the separate digest supports correlation without disclosure.
    keys = ", ".join(sorted(str(key) for key in arguments))
    return f"{tool_name}: arguments=[{keys[:max_len]}]"


def _arguments_digest(arguments: dict) -> str:
    """Stable non-reversible identifier for correlating equivalent calls."""
    try:
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(sorted(arguments))
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def _targets_persistent_permissions(tool_name: str, arguments: dict) -> bool:
    if tool_name not in ("write_file", "edit_file", "edit_ast", "bash"):
        return False
    return any(
        isinstance(value, str) and _PERSISTENT_PERMISSION_RE.search(value)
        for value in arguments.values()
    )

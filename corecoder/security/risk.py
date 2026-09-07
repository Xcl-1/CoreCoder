"""Deterministic risk floor with an optional semantic/AI reviewer hook."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from .capabilities import CapabilityReport


class RiskLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    reasons: tuple[str, ...] = ()
    reviewer: str = "deterministic"


RiskReviewer = Callable[[str, dict, RiskAssessment], RiskAssessment]


_HIGH_RISK_SHELL: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bgit\s+(?:[^\r\n]*\s)?push\b", re.IGNORECASE), "publishes commits to a remote"),
    (re.compile(r"\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*[fd])\b", re.IGNORECASE),
     "can irreversibly discard local work"),
    (re.compile(r"\b(?:npm|pnpm|yarn|twine|cargo)\s+(?:publish|deploy)\b", re.IGNORECASE),
     "publishes an external artifact"),
    (re.compile(r"\b(?:kubectl\s+(?:apply|delete)|terraform\s+(?:apply|destroy))\b", re.IGNORECASE),
     "changes external infrastructure"),
    (re.compile(r"\b(?:curl|wget)\b[^\r\n]*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--data)\b", re.IGNORECASE),
     "sends data or mutates a remote service"),
]


def deterministic_risk(tool_name: str, arguments: dict,
                       capability: CapabilityReport | None = None) -> RiskAssessment:
    """Return a non-bypassable risk floor from declared effects and arguments."""
    side_effect = capability.side_effect if capability else ""
    declared_level = {
        "low": RiskLevel.LOW,
        "medium": RiskLevel.MEDIUM,
        "high": RiskLevel.HIGH,
        "critical": RiskLevel.CRITICAL,
    }.get(capability.declared_risk if capability else "", RiskLevel.LOW)
    if tool_name == "bash":
        command = str(arguments.get("command", ""))
        for pattern, reason in _HIGH_RISK_SHELL:
            if pattern.search(command):
                return RiskAssessment(RiskLevel.HIGH, (reason,))
        if side_effect == "dynamic":
            return RiskAssessment(RiskLevel.MEDIUM, ("process behavior is dynamic",))
    if declared_level >= RiskLevel.HIGH:
        return RiskAssessment(declared_level, (f"tool declares {declared_level.label} baseline risk",))
    if side_effect in {"local_write", "dynamic", "delegated"}:
        return RiskAssessment(RiskLevel.MEDIUM, (f"tool side effect is {side_effect}",))
    return RiskAssessment(RiskLevel.LOW, ("read-only declared capability",))


def merge_reviewer_assessment(base: RiskAssessment, reviewed: RiskAssessment) -> RiskAssessment:
    """Merge a semantic review without allowing it to lower the risk floor."""
    level = max(base.level, reviewed.level)
    reasons = tuple(dict.fromkeys((*base.reasons, *reviewed.reasons)))
    return RiskAssessment(level, reasons, reviewed.reviewer or "semantic-reviewer")

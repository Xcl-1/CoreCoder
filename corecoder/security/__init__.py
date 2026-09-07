"""Security package — permissions, audit, and output sanitisation.

Provides an optional, self-contained security layer that plugs into
the agent loop via a single ``guard`` parameter on ``Agent.__init__``.

Usage::

    from corecoder.security import Guard
    agent = Agent(llm=llm, guard=Guard())
"""

from .audit import AuditEntry, AuditLogger, AuditQueryResult
from .capabilities import CapabilityReport, inspect_tool_capabilities
from .content import ContentInspection, inspect_content
from .gate import ConfirmationContext, Guard, SecurityDecision
from .network import NetworkDecision, NetworkIntent, NetworkPolicy, inspect_network_intent
from .permissions import PermissionManager, PermissionRule
from .risk import RiskAssessment, RiskLevel

__all__ = [
    "AuditEntry",
    "AuditLogger",
    "AuditQueryResult",
    "CapabilityReport",
    "ConfirmationContext",
    "ContentInspection",
    "Guard",
    "NetworkDecision",
    "NetworkIntent",
    "NetworkPolicy",
    "PermissionManager",
    "PermissionRule",
    "RiskAssessment",
    "RiskLevel",
    "SecurityDecision",
    "inspect_content",
    "inspect_network_intent",
    "inspect_tool_capabilities",
]

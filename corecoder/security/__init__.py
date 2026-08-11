"""Security package — permissions, audit, and output sanitisation.

Provides an optional, self-contained security layer that plugs into
the agent loop via a single ``guard`` parameter on ``Agent.__init__``.

Usage::

    from corecoder.security import Guard
    agent = Agent(llm=llm, guard=Guard())
"""

from .audit import AuditEntry, AuditLogger
from .gate import Guard
from .permissions import PermissionManager, PermissionRule

__all__ = [
    "AuditEntry",
    "AuditLogger",
    "Guard",
    "PermissionManager",
    "PermissionRule",
]

"""Skill discovery, routing, and prompt activation."""

from .catalog import CatalogIssue, SkillCatalog
from .evaluation import RoutingCase, RoutingMetrics, evaluate_router
from .lifecycle import allowed_transitions, transition_skill
from .manager import SkillManager
from .models import (
    RouteResult,
    RoutingContext,
    Skill,
    SkillCandidate,
    SkillManifest,
    TaskSignature,
)
from .registry import SkillRegistry, SkillSource
from .router import SkillRouter

__all__ = [
    "CatalogIssue",
    "RouteResult",
    "RoutingCase",
    "RoutingContext",
    "RoutingMetrics",
    "Skill",
    "SkillCandidate",
    "SkillCatalog",
    "SkillManager",
    "SkillManifest",
    "SkillRegistry",
    "SkillRouter",
    "SkillSource",
    "TaskSignature",
    "allowed_transitions",
    "evaluate_router",
    "transition_skill",
]

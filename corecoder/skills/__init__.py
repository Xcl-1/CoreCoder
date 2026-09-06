"""Skill discovery, routing, and prompt activation."""

from .catalog import CatalogIssue, SkillCatalog
from .evaluation import RoutingCase, RoutingMetrics, evaluate_router
from .evolution import SkillEvolutionEngine
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
from .telemetry import SkillTelemetryStore

__all__ = [
    "CatalogIssue",
    "RouteResult",
    "RoutingCase",
    "RoutingContext",
    "RoutingMetrics",
    "Skill",
    "SkillCandidate",
    "SkillCatalog",
    "SkillEvolutionEngine",
    "SkillManager",
    "SkillManifest",
    "SkillRegistry",
    "SkillRouter",
    "SkillSource",
    "SkillTelemetryStore",
    "TaskSignature",
    "allowed_transitions",
    "evaluate_router",
    "transition_skill",
]

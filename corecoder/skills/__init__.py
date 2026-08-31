"""Skill discovery, routing, and prompt activation."""

from .manager import SkillManager
from .models import RouteResult, Skill, SkillCandidate, SkillManifest
from .registry import SkillRegistry, SkillSource
from .router import SkillRouter

__all__ = [
    "RouteResult",
    "Skill",
    "SkillCandidate",
    "SkillManager",
    "SkillManifest",
    "SkillRegistry",
    "SkillRouter",
    "SkillSource",
]

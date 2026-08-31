"""Typed models for reusable, dynamically routed agent skills."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SkillStatus = Literal["draft", "candidate", "active", "disabled", "deprecated"]
SkillScope = Literal["builtin", "user", "project", "custom"]


class SkillTools(BaseModel):
    """Tool requirements and restrictions declared by a skill."""

    required: list[str] = Field(default_factory=list)
    recommended: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_disjoint_sets(self) -> SkillTools:
        groups = {
            "required": set(self.required),
            "recommended": set(self.recommended),
            "forbidden": set(self.forbidden),
        }
        conflicts = (groups["required"] | groups["recommended"]) & groups["forbidden"]
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"tools cannot be both allowed and forbidden: {names}")
        return self


class SkillExamples(BaseModel):
    """Positive and negative routing examples kept in the small manifest."""

    positive: list[str] = Field(default_factory=list, max_length=20)
    negative: list[str] = Field(default_factory=list, max_length=20)


class SkillManifest(BaseModel):
    """The small machine-readable portion loaded during discovery."""

    schema_version: int = Field(default=1, ge=1, le=1)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,79}$")
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(default="1.0.0", min_length=1, max_length=40)
    summary: str = Field(min_length=1, max_length=500)
    category: list[str] = Field(default_factory=list, max_length=8)
    tags: list[str] = Field(default_factory=list, max_length=30)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    intents: list[str] = Field(default_factory=list, max_length=30)
    applies_when: list[str] = Field(default_factory=list, max_length=20)
    not_when: list[str] = Field(default_factory=list, max_length=20)
    examples: SkillExamples = Field(default_factory=SkillExamples)
    tools: SkillTools = Field(default_factory=SkillTools)
    conflicts_with: list[str] = Field(default_factory=list, max_length=10)
    exclusive_group: str | None = Field(default=None, max_length=80)
    token_budget: int = Field(default=1800, ge=200, le=12000)
    priority: int = Field(default=0, ge=-100, le=100)
    status: SkillStatus = "active"


class Skill(BaseModel):
    """A discovered skill and the location of its full instructions."""

    manifest: SkillManifest
    path: Path
    scope: SkillScope
    source_priority: int = 0
    instructions: str = ""

    model_config = {"arbitrary_types_allowed": True}


class SkillCandidate(BaseModel):
    """One routed candidate with human-readable scoring evidence."""

    skill: Skill
    score: float
    reasons: list[str] = Field(default_factory=list)
    explicit: bool = False


class RouteResult(BaseModel):
    """Complete routing trace for the current user turn."""

    query: str
    candidates: list[SkillCandidate] = Field(default_factory=list)
    selected: list[SkillCandidate] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    prompt: str = ""

    @property
    def selected_ids(self) -> list[str]:
        return [item.skill.manifest.id for item in self.selected]

    @property
    def forbidden_tools(self) -> set[str]:
        denied: set[str] = set()
        for item in self.selected:
            denied.update(item.skill.manifest.tools.forbidden)
        return denied

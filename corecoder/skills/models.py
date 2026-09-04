"""Typed models for reusable, dynamically routed agent skills."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SkillStatus = Literal[
    "draft",
    "candidate",
    "shadow",
    "canary",
    "active",
    "disabled",
    "deprecated",
]
SkillScope = Literal["builtin", "user", "project", "custom"]
SkillLayer = Literal["atomic", "workflow", "orchestrator"]
SkillRisk = Literal["low", "medium", "high"]
RouteDecision = Literal["explicit", "auto", "clarify", "abstain"]
TaskIntentMode = Literal["single", "multi", "exploratory"]


class SkillIntentSignature(BaseModel):
    """Structured routing vocabulary for one skill.

    Fields are deliberately optional so schema-v1 manifests remain useful.  A
    catalog can progressively gain precision as maintainers add discriminating
    values instead of rewriting every existing package at once.
    """

    domains: list[str] = Field(default_factory=list, max_length=12)
    actions: list[str] = Field(default_factory=list, max_length=24)
    objects: list[str] = Field(default_factory=list, max_length=24)
    artifacts: list[str] = Field(default_factory=list, max_length=16)
    outputs: list[str] = Field(default_factory=list, max_length=16)
    constraints: list[str] = Field(default_factory=list, max_length=20)
    contexts: list[str] = Field(default_factory=list, max_length=20)

    def dimensions(self) -> dict[str, list[str]]:
        return {
            "domains": self.domains,
            "actions": self.actions,
            "objects": self.objects,
            "artifacts": self.artifacts,
            "outputs": self.outputs,
            "constraints": self.constraints,
            "contexts": self.contexts,
        }


class RoutingContext(BaseModel):
    """Observable context supplied by the host without another model call."""

    attachments: list[str] = Field(default_factory=list, max_length=20)
    artifact_types: set[str] = Field(default_factory=set)
    inputs: set[str] = Field(default_factory=set)
    connected_apps: set[str] = Field(default_factory=set)
    granted_permissions: set[str] = Field(default_factory=set)
    live_app: bool = False
    external_write: bool = False
    risk: SkillRisk = "low"
    intent_mode: TaskIntentMode | None = None
    routing_key: str = ""

    def signals(self) -> set[str]:
        values = set(self.inputs)
        values.update(f"app:{name.lower()}" for name in self.connected_apps)
        if self.attachments:
            values.add("attachment")
        if self.live_app:
            values.add("live_app")
        if self.external_write:
            values.add("external_write")
        return values


class TaskSignature(BaseModel):
    """A compact interpretation of the current request used during ranking."""

    domains: set[str] = Field(default_factory=set)
    actions: set[str] = Field(default_factory=set)
    objects: set[str] = Field(default_factory=set)
    artifacts: set[str] = Field(default_factory=set)
    outputs: set[str] = Field(default_factory=set)
    constraints: set[str] = Field(default_factory=set)
    contexts: set[str] = Field(default_factory=set)
    risk: SkillRisk = "low"
    intent_mode: TaskIntentMode = "single"

    def dimensions(self) -> dict[str, set[str]]:
        return {
            "domains": self.domains,
            "actions": self.actions,
            "objects": self.objects,
            "artifacts": self.artifacts,
            "outputs": self.outputs,
            "constraints": self.constraints,
            "contexts": self.contexts,
        }


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


class SkillContrastiveExample(BaseModel):
    """A boundary example that should route to a specific alternative."""

    query: str = Field(min_length=1, max_length=500)
    expected_skill: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,79}$")
    reason: str = Field(default="", max_length=300)


class SkillExamples(BaseModel):
    """Small, routing-only examples; full examples belong in references."""

    positive: list[str] = Field(default_factory=list, max_length=20)
    negative: list[str] = Field(default_factory=list, max_length=20)
    hard_negative: list[str] = Field(default_factory=list, max_length=20)
    contrastive: list[SkillContrastiveExample] = Field(default_factory=list, max_length=20)


class SkillRelations(BaseModel):
    """Capability-graph edges used for composition and lifecycle governance."""

    dependencies: list[str] = Field(default_factory=list, max_length=10)
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    composes_with: list[str] = Field(default_factory=list, max_length=20)
    supersedes: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_dependency_versions(self) -> SkillRelations:
        undeclared = set(self.dependency_versions) - set(self.dependencies)
        if undeclared:
            raise ValueError(
                "dependency_versions contains undeclared dependencies: "
                + ", ".join(sorted(undeclared))
            )
        return self


class SkillRequirements(BaseModel):
    """Non-tool prerequisites used as hard routing gates."""

    context_any: list[str] = Field(default_factory=list, max_length=20)
    inputs_any: list[str] = Field(default_factory=list, max_length=20)
    permissions: list[str] = Field(default_factory=list, max_length=20)


class SkillResourceMode(BaseModel):
    """Resources loaded only when one of the mode's routing phrases matches."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,79}$")
    when: list[str] = Field(min_length=1, max_length=20)
    references: list[str] = Field(default_factory=list, max_length=20)
    scripts: list[str] = Field(default_factory=list, max_length=20)
    assets: list[str] = Field(default_factory=list, max_length=20)


class SkillRoutingPolicy(BaseModel):
    """Invocation and rollout policy kept out of executable instructions."""

    allow_implicit: bool = True
    risk: SkillRisk = "low"
    rollout_percent: int = Field(default=100, ge=0, le=100)


class SkillLifecycleState(BaseModel):
    """Last reviewed lifecycle transition, retained in the manifest."""

    previous_status: SkillStatus | None = None
    changed_at: str = ""
    reason: str = Field(default="", max_length=500)


class SkillManifest(BaseModel):
    """The small machine-readable portion loaded during discovery."""

    schema_version: int = Field(default=1, ge=1, le=2)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,79}$")
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(default="1.0.0", min_length=1, max_length=40)
    summary: str = Field(min_length=1, max_length=500)
    layer: SkillLayer = "workflow"
    category: list[str] = Field(default_factory=list, max_length=8)
    tags: list[str] = Field(default_factory=list, max_length=30)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    intents: list[str] = Field(default_factory=list, max_length=30)
    signature: SkillIntentSignature = Field(default_factory=SkillIntentSignature)
    applies_when: list[str] = Field(default_factory=list, max_length=20)
    not_when: list[str] = Field(default_factory=list, max_length=20)
    examples: SkillExamples = Field(default_factory=SkillExamples)
    tools: SkillTools = Field(default_factory=SkillTools)
    requires: SkillRequirements = Field(default_factory=SkillRequirements)
    relations: SkillRelations = Field(default_factory=SkillRelations)
    resource_modes: list[SkillResourceMode] = Field(default_factory=list, max_length=20)
    routing: SkillRoutingPolicy = Field(default_factory=SkillRoutingPolicy)
    lifecycle: SkillLifecycleState = Field(default_factory=SkillLifecycleState)
    conflicts_with: list[str] = Field(default_factory=list, max_length=10)
    exclusive_group: str | None = Field(default=None, max_length=80)
    token_budget: int = Field(default=1800, ge=200, le=12000)
    priority: int = Field(default=0, ge=-100, le=100)
    status: SkillStatus = "active"

    @model_validator(mode="after")
    def validate_relationships(self) -> SkillManifest:
        relation_groups = {
            "dependencies": set(self.relations.dependencies),
            "composes_with": set(self.relations.composes_with),
            "supersedes": set(self.relations.supersedes),
            "conflicts_with": set(self.conflicts_with),
        }
        self_references = [name for name, ids in relation_groups.items() if self.id in ids]
        if self_references:
            raise ValueError(f"skill cannot reference itself in {', '.join(self_references)}")
        contradictory = relation_groups["dependencies"] & relation_groups["conflicts_with"]
        if contradictory:
            names = ", ".join(sorted(contradictory))
            raise ValueError(f"dependencies cannot also conflict with this skill: {names}")
        return self


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
    recall_score: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    matched_dimensions: list[str] = Field(default_factory=list)
    explicit: bool = False
    shadow: bool = False


class RouteResult(BaseModel):
    """Complete routing trace for the current user turn."""

    query: str
    candidates: list[SkillCandidate] = Field(default_factory=list)
    selected: list[SkillCandidate] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    prompt: str = ""
    decision: RouteDecision = "abstain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    margin: float = Field(default=0.0, ge=0.0, le=1.0)
    clarification: str = ""
    signature: TaskSignature = Field(default_factory=TaskSignature)

    @property
    def selected_ids(self) -> list[str]:
        return [item.skill.manifest.id for item in self.selected]

    @property
    def forbidden_tools(self) -> set[str]:
        denied: set[str] = set()
        for item in self.selected:
            denied.update(item.skill.manifest.tools.forbidden)
        return denied

    @property
    def needs_clarification(self) -> bool:
        return self.decision == "clarify"

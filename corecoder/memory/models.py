"""Data models shared by the memory pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

MemoryType = Literal[
    "user",
    "feedback",
    "project",
    "reference",
    "procedure",
    "episode",
    "profile",
]
MemoryScope = Literal["global", "project"]
MemoryAction = Literal["create", "merge", "archive", "ignore"]
MemoryStatus = Literal["candidate", "active", "archived", "superseded"]
ReflectionOutcome = Literal["success", "partial", "failure", "unknown"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Memory(BaseModel):
    """A durable fact learned from one or more conversations."""

    id: str
    title: str
    description: str
    content: str
    type: MemoryType = "project"
    scope: MemoryScope = "project"
    project_path: str | None = None
    keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    source_sessions: list[str] = Field(default_factory=list)
    last_used_at: str | None = None
    use_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    validation_count: int = Field(default=0, ge=0)
    validated_at: str | None = None
    status: MemoryStatus = "active"
    supersedes: str | None = None
    version: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class ExtractedMemory(BaseModel):
    """One operation proposed by the LLM memory extractor."""

    action: MemoryAction = "create"
    target_id: str | None = None
    title: str = ""
    description: str = ""
    content: str = ""
    type: MemoryType = "project"
    scope: MemoryScope = "project"
    keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence: str
    supersedes: str | None = None


class SessionReflection(BaseModel):
    """Structured review of one execution session."""

    task_summary: str = ""
    outcome: ReflectionOutcome = "unknown"
    summary: str = ""
    failures: list[str] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    effective_actions: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    reusable_lessons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    tool_executions: int = Field(default=0, ge=0)
    successful_tools: int = Field(default=0, ge=0)
    failed_tools: int = Field(default=0, ge=0)

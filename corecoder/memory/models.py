"""Data models shared by the memory pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

MemoryType = Literal["user", "feedback", "project", "reference"]
MemoryScope = Literal["global", "project"]
MemoryAction = Literal["create", "merge", "ignore"]


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
    source_sessions: list[str] = Field(default_factory=list)
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

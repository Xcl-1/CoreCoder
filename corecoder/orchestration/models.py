"""Shared, backend-neutral workflow protocol models."""

from __future__ import annotations

import enum
import hashlib
import uuid

from pydantic import BaseModel, ConfigDict, Field

from ..delegation import TaskResult, TaskSpec, TaskStatus


def _workflow_id() -> str:
    return f"workflow_{uuid.uuid4().hex[:12]}"


class WorkflowBackend(str, enum.Enum):
    NATIVE = "native"
    LANGGRAPH = "langgraph"


class WorkflowStage(str, enum.Enum):
    CREATED = "created"
    PLAN = "plan"
    APPROVAL = "approval"
    EXECUTE = "execute"
    VERIFY = "verify"
    REVIEW = "review"
    DECIDE = "decide"
    FINISHED = "finished"


class ReviewVerdict(str, enum.Enum):
    PASS = "pass"
    REPLAN = "replan"
    REJECT = "reject"


class FailureType(str, enum.Enum):
    NONE = "none"
    TRANSIENT = "transient"
    IMPLEMENTATION = "implementation"
    PLAN = "plan"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    POLICY = "policy"
    BUDGET = "budget"


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    evidence: tuple[str, ...] = ()
    feedback: str = Field(default="", max_length=2_000)


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ReviewVerdict
    failure_type: FailureType = FailureType.NONE
    feedback: str = Field(default="", max_length=2_000)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=80)
    task_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=1, max_length=2_000)

    @classmethod
    def for_task(cls, task: TaskSpec, reason: str) -> ApprovalRequest:
        digest = hashlib.sha256(task.model_dump_json().encode("utf-8")).hexdigest()
        return cls(
            request_id=f"approval_{digest[:16]}",
            task_digest=digest,
            reason=reason,
        )


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=80)
    task_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved: bool


class WorkflowRequest(BaseModel):
    """One backend-neutral request backed by a fully scoped delegated task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: str = Field(default_factory=_workflow_id, min_length=1, max_length=80)
    task: TaskSpec
    max_replans: int = Field(default=0, ge=0, le=3)
    max_total_tokens: int = Field(default=64_000, ge=512, le=3_000_000)


class WorkflowResult(BaseModel):
    """Bounded workflow result whose task facts still come from the controller."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    backend: WorkflowBackend
    stage: WorkflowStage
    status: TaskStatus
    task_results: list[TaskResult] = Field(default_factory=list, max_length=20)
    verification: VerificationResult | None = None
    review: ReviewResult | None = None
    replan_count: int = Field(default=0, ge=0, le=3)
    pending_approval: ApprovalRequest | None = None
    trace: list[WorkflowStage] = Field(default_factory=list, max_length=100)
    error: str | None = Field(default=None, max_length=2_000)

    @property
    def final_task_result(self) -> TaskResult | None:
        return self.task_results[-1] if self.task_results else None

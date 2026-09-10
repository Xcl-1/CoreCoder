"""Deterministic metrics for comparing orchestration backends."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ..delegation import TaskStatus
from .models import WorkflowBackend, WorkflowResult, WorkflowStage

_PASSING_TEST_STATUSES = frozenset({"pass", "passed", "ok", "success", "successful"})


class WorkflowMetrics(BaseModel):
    """Facts derived from one completed or interrupted workflow result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: str
    backend: WorkflowBackend
    succeeded: bool
    task_attempts: int = Field(ge=0)
    replan_count: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    policy_violations: int = Field(ge=0)
    reported_tests: int = Field(ge=0)
    passed_reported_tests: int = Field(ge=0)
    acceptance_checks: int = Field(ge=0)
    parent_verified_acceptance: int = Field(ge=0)
    approval_stage_seen: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class EvaluationSummary(BaseModel):
    """Aggregate suitable for a Native/LangGraph A/B report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: WorkflowBackend
    workflow_count: int = Field(ge=0)
    successful_workflows: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    task_attempts: int = Field(ge=0)
    replan_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    total_duration_ms: float = Field(ge=0)
    average_duration_ms: float = Field(ge=0)
    policy_violations: int = Field(ge=0)
    reported_tests: int = Field(ge=0)
    passed_reported_tests: int = Field(ge=0)
    reported_test_pass_rate: float | None = Field(default=None, ge=0, le=1)
    parent_verified_acceptance: int = Field(ge=0)


def measure_workflow(result: WorkflowResult) -> WorkflowMetrics:
    """Extract trusted counters without treating model prose as evidence."""

    tasks = result.task_results
    tests = [test for task in tasks for test in task.tests]
    acceptance = [check for task in tasks for check in task.acceptance]
    return WorkflowMetrics(
        workflow_id=result.workflow_id,
        backend=result.backend,
        succeeded=result.status == TaskStatus.COMPLETED,
        task_attempts=len(tasks),
        replan_count=result.replan_count,
        prompt_tokens=sum(task.usage.prompt_tokens for task in tasks),
        completion_tokens=sum(task.usage.completion_tokens for task in tasks),
        tool_calls=sum(task.usage.tool_calls for task in tasks),
        duration_ms=sum(task.usage.duration_ms for task in tasks),
        policy_violations=sum(task.policy_violations for task in tasks),
        reported_tests=len(tests),
        passed_reported_tests=sum(
            test.status.strip().casefold() in _PASSING_TEST_STATUSES for test in tests
        ),
        acceptance_checks=len(acceptance),
        parent_verified_acceptance=sum(
            check.passed is True and check.verified_by_parent for check in acceptance
        ),
        approval_stage_seen=WorkflowStage.APPROVAL in result.trace,
    )


def summarize_workflows(
    backend: WorkflowBackend,
    results: Iterable[WorkflowResult],
) -> EvaluationSummary:
    """Aggregate one backend and reject accidental mixed-backend samples."""

    metrics = [measure_workflow(result) for result in results]
    mismatched = [item.backend.value for item in metrics if item.backend != backend]
    if mismatched:
        raise ValueError(
            f"evaluation for {backend.value!r} contains other backends: {mismatched}"
        )
    count = len(metrics)
    successes = sum(item.succeeded for item in metrics)
    duration = sum(item.duration_ms for item in metrics)
    tests = sum(item.reported_tests for item in metrics)
    passed_tests = sum(item.passed_reported_tests for item in metrics)
    return EvaluationSummary(
        backend=backend,
        workflow_count=count,
        successful_workflows=successes,
        success_rate=successes / count if count else 0.0,
        task_attempts=sum(item.task_attempts for item in metrics),
        replan_count=sum(item.replan_count for item in metrics),
        total_tokens=sum(item.total_tokens for item in metrics),
        tool_calls=sum(item.tool_calls for item in metrics),
        total_duration_ms=duration,
        average_duration_ms=duration / count if count else 0.0,
        policy_violations=sum(item.policy_violations for item in metrics),
        reported_tests=tests,
        passed_reported_tests=passed_tests,
        reported_test_pass_rate=passed_tests / tests if tests else None,
        parent_verified_acceptance=sum(
            item.parent_verified_acceptance for item in metrics
        ),
    )

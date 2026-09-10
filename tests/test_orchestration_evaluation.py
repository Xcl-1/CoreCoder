"""Deterministic orchestration evaluation tests."""

import pytest

from corecoder.delegation import (
    AcceptanceCheck,
    TaskResult,
    TaskRole,
    TaskStatus,
    TaskTestResult,
    TaskUsage,
    WorkspaceMode,
)
from corecoder.orchestration import (
    WorkflowBackend,
    WorkflowResult,
    WorkflowStage,
    measure_workflow,
    summarize_workflows,
)


def _result(
    *,
    backend=WorkflowBackend.LANGGRAPH,
    status=TaskStatus.COMPLETED,
    task_id="task-1",
    policy_violations=0,
):
    task = TaskResult(
        task_id=task_id,
        agent_id="child",
        parent_id="parent",
        role=TaskRole.EXECUTOR,
        execution_mode=WorkspaceMode.FORK,
        status=status,
        policy_violations=policy_violations,
        usage=TaskUsage(
            prompt_tokens=100,
            completion_tokens=25,
            tool_calls=2,
            duration_ms=40,
        ),
        tests=[
            TaskTestResult(name="unit", status="passed"),
            TaskTestResult(name="integration", status="failed"),
        ],
        acceptance=[
            AcceptanceCheck(
                criterion="verified",
                passed=True,
                verified_by_parent=True,
            ),
            AcceptanceCheck(criterion="model-only", passed=True),
        ],
    )
    return WorkflowResult(
        workflow_id=f"workflow-{task_id}",
        backend=backend,
        stage=WorkflowStage.FINISHED,
        status=status,
        task_results=[task],
        replan_count=1,
        trace=[WorkflowStage.APPROVAL, WorkflowStage.FINISHED],
    )


def test_measure_workflow_uses_structured_runtime_facts():
    metrics = measure_workflow(_result(policy_violations=3))

    assert metrics.succeeded is True
    assert metrics.task_attempts == 1
    assert metrics.total_tokens == 125
    assert metrics.tool_calls == 2
    assert metrics.duration_ms == 40
    assert metrics.policy_violations == 3
    assert metrics.reported_tests == 2
    assert metrics.passed_reported_tests == 1
    assert metrics.acceptance_checks == 2
    assert metrics.parent_verified_acceptance == 1
    assert metrics.approval_stage_seen is True


def test_summarize_workflows_produces_comparable_rates():
    completed = _result(task_id="success")
    failed = _result(task_id="failed", status=TaskStatus.FAILED)

    summary = summarize_workflows(WorkflowBackend.LANGGRAPH, [completed, failed])

    assert summary.workflow_count == 2
    assert summary.successful_workflows == 1
    assert summary.success_rate == 0.5
    assert summary.task_attempts == 2
    assert summary.replan_count == 2
    assert summary.total_tokens == 250
    assert summary.average_duration_ms == 40
    assert summary.reported_test_pass_rate == 0.5
    assert summary.parent_verified_acceptance == 2


def test_summarize_empty_sample_is_explicit():
    summary = summarize_workflows(WorkflowBackend.NATIVE, [])

    assert summary.workflow_count == 0
    assert summary.success_rate == 0
    assert summary.reported_test_pass_rate is None


def test_summarize_rejects_mixed_backends():
    with pytest.raises(ValueError, match="other backends"):
        summarize_workflows(WorkflowBackend.NATIVE, [_result()])

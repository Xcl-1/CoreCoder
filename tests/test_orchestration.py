"""Backend-neutral orchestration contract tests."""

import json
from pathlib import Path

import pytest

from corecoder.agent import Agent
from corecoder.delegation import TaskController, TaskResult, TaskSpec, TaskStatus
from corecoder.models import LLMResponse
from corecoder.orchestration import (
    ApprovalDecision,
    ApprovalRequest,
    FailureType,
    LangGraphOrchestrator,
    Orchestrator,
    ReviewResult,
    ReviewVerdict,
    TurnStage,
    TurnStatus,
    VerificationResult,
    WorkflowBackend,
    WorkflowRequest,
    WorkflowStage,
    encrypted_sqlite_checkpointer,
)
from corecoder.orchestration.native import NativeOrchestrator
from corecoder.orchestration.persistence import safe_checkpoint_serializer
from corecoder.tools import ALL_TOOLS


def _completed(spec: TaskSpec, agent_id: str, parent_id: str = "parent") -> TaskResult:
    return TaskResult(
        task_id=spec.task_id,
        agent_id=agent_id,
        parent_id=parent_id,
        role=spec.role,
        execution_mode=spec.execution_mode,
        status=TaskStatus.COMPLETED,
        summary="executed through the controller",
    )


def test_checkpoint_serializer_rejects_non_protocol_types():
    serializer = safe_checkpoint_serializer()
    payload = serializer.dumps_typed(Path("not-allowed"))

    with pytest.raises(ValueError, match="ext_hook failed"):
        serializer.loads_typed(payload)


class _WorkflowLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, on_token=None):
        del messages, tools, on_token
        self.calls += 1
        return LLMResponse(content=json.dumps({
            "summary": "workflow integration completed",
            "evidence": ["real Agent delegation path"],
            "tests": [],
            "acceptance": [],
            "risks": [],
        }))


def test_native_orchestrator_satisfies_protocol():
    async def execute(spec):
        return _completed(spec, "child")

    orchestrator: Orchestrator = NativeOrchestrator(execute)
    assert orchestrator.backend == WorkflowBackend.NATIVE


@pytest.mark.asyncio
async def test_native_orchestrator_uses_real_task_controller_path():
    calls = []

    async def runner(spec, agent_id):
        calls.append((spec, agent_id))
        return _completed(spec, agent_id)

    controller = TaskController(runner, parent_id="parent")
    orchestrator = NativeOrchestrator(controller.execute)
    request = WorkflowRequest(task=TaskSpec(objective="exercise native backend"))

    result = await orchestrator.run(request)

    assert len(calls) == 1
    assert calls[0][0] == request.task
    assert controller.result(request.task.task_id) is result.final_task_result
    assert result.backend == WorkflowBackend.NATIVE
    assert result.stage == WorkflowStage.FINISHED
    assert result.status == TaskStatus.COMPLETED
    assert result.final_task_result.summary == "executed through the controller"


@pytest.mark.asyncio
async def test_native_orchestrator_preserves_controller_failure_facts():
    async def runner(spec, agent_id):
        return TaskResult(
            task_id=spec.task_id,
            agent_id=agent_id,
            parent_id="parent",
            role=spec.role,
            execution_mode=spec.execution_mode,
            status=TaskStatus.FAILED,
            error="trusted runner failure",
            policy_violations=2,
        )

    result = await NativeOrchestrator(
        TaskController(runner, parent_id="parent").execute
    ).run(WorkflowRequest(task=TaskSpec(objective="fail safely")))

    assert result.status == TaskStatus.FAILED
    assert result.error == "trusted runner failure"
    assert result.final_task_result.policy_violations == 2


@pytest.mark.asyncio
async def test_langgraph_runs_planning_execution_verification_and_review():
    pytest.importorskip("langgraph")
    observed = []

    async def runner(spec, agent_id):
        observed.append(("execute", spec.objective))
        return _completed(spec, agent_id)

    async def planner(request, feedback, attempt):
        assert feedback == ""
        assert attempt == 0
        observed.append(("plan", request.task.objective))
        return request.task

    async def verifier(result):
        observed.append(("verify", result.status.value))
        return VerificationResult(passed=True, evidence=("runtime evidence",))

    async def reviewer(result, verification):
        observed.append(("review", verification.passed))
        return ReviewResult(verdict=ReviewVerdict.PASS)

    controller = TaskController(runner, parent_id="parent")
    orchestrator: Orchestrator = LangGraphOrchestrator(
        controller.execute,
        planner=planner,
        verifier=verifier,
        reviewer=reviewer,
    )
    result = await orchestrator.run(
        WorkflowRequest(task=TaskSpec(objective="run the guarded graph"))
    )

    assert observed == [
        ("plan", "run the guarded graph"),
        ("execute", "run the guarded graph"),
        ("verify", "completed"),
        ("review", True),
    ]
    assert result.backend == WorkflowBackend.LANGGRAPH
    assert result.status == TaskStatus.COMPLETED
    assert result.trace == [
        WorkflowStage.PLAN,
        WorkflowStage.EXECUTE,
        WorkflowStage.VERIFY,
        WorkflowStage.REVIEW,
        WorkflowStage.DECIDE,
        WorkflowStage.FINISHED,
    ]


@pytest.mark.asyncio
async def test_langgraph_cannot_turn_failed_verification_into_success():
    pytest.importorskip("langgraph")

    async def runner(spec, agent_id):
        return _completed(spec, agent_id)

    async def verifier(_result):
        return VerificationResult(passed=False, feedback="tests failed")

    async def careless_reviewer(_result, _verification):
        return ReviewResult(verdict=ReviewVerdict.PASS)

    controller = TaskController(runner, parent_id="parent")
    result = await LangGraphOrchestrator(
        controller.execute,
        verifier=verifier,
        reviewer=careless_reviewer,
    ).run(WorkflowRequest(task=TaskSpec(objective="must remain failed")))

    assert result.status == TaskStatus.FAILED
    assert result.error == "tests failed"
    assert controller.result(result.final_task_result.task_id).status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_langgraph_planner_cannot_replace_task_identity():
    pytest.importorskip("langgraph")

    async def execute(_spec):
        raise AssertionError("identity violation must fail before execution")

    async def planner(request, _feedback, _attempt):
        return request.task.model_copy(update={"task_id": "replacement"})

    orchestrator = LangGraphOrchestrator(execute, planner=planner)
    with pytest.raises(ValueError, match="may not replace"):
        await orchestrator.run(
            WorkflowRequest(task=TaskSpec(objective="preserve authority identity"))
        )


@pytest.mark.asyncio
async def test_langgraph_replans_read_only_work_with_fresh_task_identity():
    pytest.importorskip("langgraph")
    executions = []
    plans = []

    async def runner(spec, agent_id):
        executions.append(spec.task_id)
        result = _completed(spec, agent_id)
        result.usage.prompt_tokens = 100
        return result

    async def planner(request, feedback, attempt):
        plans.append((feedback, attempt))
        objective = request.task.objective if not feedback else f"{request.task.objective}; {feedback}"
        return request.task.model_copy(update={"objective": objective})

    async def verifier(_result):
        return VerificationResult(passed=len(executions) > 1, feedback="collect better evidence")

    async def reviewer(_result, verification):
        return ReviewResult(
            verdict=ReviewVerdict.PASS if verification.passed else ReviewVerdict.REPLAN,
            failure_type=(FailureType.NONE if verification.passed else FailureType.INSUFFICIENT_EVIDENCE),
            feedback=verification.feedback,
        )

    request = WorkflowRequest(
        task=TaskSpec(objective="inspect safely"),
        max_replans=1,
        max_total_tokens=1_000,
    )
    controller = TaskController(runner, parent_id="parent")
    result = await LangGraphOrchestrator(
        controller.execute,
        planner=planner,
        verifier=verifier,
        reviewer=reviewer,
    ).run(request)

    assert result.status == TaskStatus.COMPLETED
    assert result.replan_count == 1
    assert len(result.task_results) == 2
    assert executions[0] == request.task.task_id
    assert executions[1] != executions[0]
    assert plans == [("", 0), ("collect better evidence", 1)]


@pytest.mark.asyncio
async def test_langgraph_blocks_shared_tree_write_replay():
    pytest.importorskip("langgraph")
    calls = 0

    async def runner(spec, agent_id):
        nonlocal calls
        calls += 1
        return _completed(spec, agent_id)

    async def reviewer(_result, _verification):
        return ReviewResult(
            verdict=ReviewVerdict.REPLAN,
            failure_type=FailureType.IMPLEMENTATION,
            feedback="try writing again",
        )

    task = TaskSpec(
        objective="unsafe retry",
        allowed_tools=("write_file",),
        write_paths=("output.txt",),
    )
    result = await LangGraphOrchestrator(
        TaskController(runner, parent_id="parent").execute,
        reviewer=reviewer,
    ).run(WorkflowRequest(task=task, max_replans=2))

    assert calls == 1
    assert result.status == TaskStatus.FAILED
    assert result.error == "write retries require worktree isolation"


@pytest.mark.asyncio
async def test_langgraph_blocks_replan_after_policy_violation():
    pytest.importorskip("langgraph")
    calls = 0

    async def runner(spec, agent_id):
        nonlocal calls
        calls += 1
        result = _completed(spec, agent_id)
        result.policy_violations = 1
        return result

    async def reviewer(_result, _verification):
        return ReviewResult(verdict=ReviewVerdict.REPLAN, failure_type=FailureType.POLICY)

    result = await LangGraphOrchestrator(
        TaskController(runner, parent_id="parent").execute,
        reviewer=reviewer,
    ).run(WorkflowRequest(task=TaskSpec(objective="do not retry"), max_replans=2))

    assert calls == 1
    assert result.status == TaskStatus.FAILED
    assert result.error == "policy violations cannot be retried"


@pytest.mark.asyncio
async def test_langgraph_pauses_before_execution_and_resumes_with_bound_approval():
    pytest.importorskip("langgraph")
    calls = 0

    async def runner(spec, agent_id):
        nonlocal calls
        calls += 1
        return _completed(spec, agent_id)

    async def approval_gate(task):
        return ApprovalRequest.for_task(task, "test approval required")

    request = WorkflowRequest(task=TaskSpec(objective="pause safely"))
    orchestrator = LangGraphOrchestrator(
        TaskController(runner, parent_id="parent").execute,
        approval_gate=approval_gate,
    )

    paused = await orchestrator.run(request)

    assert paused.status == TaskStatus.INTERRUPTED
    assert paused.stage == WorkflowStage.APPROVAL
    assert paused.pending_approval is not None
    assert calls == 0

    approval = paused.pending_approval
    resumed = await orchestrator.resume(
        request.workflow_id,
        ApprovalDecision(
            request_id=approval.request_id,
            task_digest=approval.task_digest,
            approved=True,
        ),
    )

    assert calls == 1
    assert resumed.status == TaskStatus.COMPLETED
    assert resumed.pending_approval is None
    assert WorkflowStage.APPROVAL in resumed.trace


@pytest.mark.asyncio
async def test_langgraph_denied_or_mismatched_approval_never_executes():
    pytest.importorskip("langgraph")

    async def execute(_spec):
        raise AssertionError("denied workflow must not execute")

    async def approval_gate(task):
        return ApprovalRequest.for_task(task, "approval required")

    for mutate_digest, expected in (
        (False, "workflow execution was denied"),
        (True, "approval decision does not match the planned task"),
    ):
        request = WorkflowRequest(task=TaskSpec(objective=f"deny {mutate_digest}"))
        orchestrator = LangGraphOrchestrator(execute, approval_gate=approval_gate)
        paused = await orchestrator.run(request)
        approval = paused.pending_approval
        digest = "0" * 64 if mutate_digest else approval.task_digest
        stopped = await orchestrator.resume(
            request.workflow_id,
            ApprovalDecision(
                request_id=approval.request_id,
                task_digest=digest,
                approved=False,
            ),
        )

        assert stopped.status == TaskStatus.REJECTED
        assert stopped.error == expected
        assert stopped.task_results == []


@pytest.mark.asyncio
async def test_langgraph_default_gate_pauses_unscoped_task():
    pytest.importorskip("langgraph")

    async def execute(_spec):
        raise AssertionError("unapproved unscoped task must not execute")

    task = TaskSpec(
        objective="use shell",
        allowed_tools=("bash",),
        allow_unscoped_tools=True,
    )
    result = await LangGraphOrchestrator(execute).run(WorkflowRequest(task=task))

    assert result.status == TaskStatus.INTERRUPTED
    assert "unscoped" in result.pending_approval.reason


@pytest.mark.asyncio
async def test_encrypted_sqlite_checkpoint_resumes_in_new_orchestrator(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    database = tmp_path / "workflow.db"
    key = b"0123456789abcdef0123456789abcdef"
    objective = "SENSITIVE WORKFLOW OBJECTIVE"

    async def approval_gate(task):
        return ApprovalRequest.for_task(task, "persistent approval required")

    request = WorkflowRequest(task=TaskSpec(objective=objective))
    async with encrypted_sqlite_checkpointer(database, key=key) as first_checkpointer:
        first = LangGraphOrchestrator(
            lambda _spec: None,
            approval_gate=approval_gate,
            checkpointer=first_checkpointer,
        )
        paused = await first.run(request)

    assert paused.status == TaskStatus.INTERRUPTED
    assert objective.encode("utf-8") not in database.read_bytes()
    pending = paused.pending_approval
    calls = 0

    async def execute(spec):
        nonlocal calls
        calls += 1
        return _completed(spec, "child")

    async with encrypted_sqlite_checkpointer(database, key=key) as second_checkpointer:
        second = LangGraphOrchestrator(
            execute,
            approval_gate=approval_gate,
            checkpointer=second_checkpointer,
        )
        resumed = await second.resume(
            request.workflow_id,
            ApprovalDecision(
                request_id=pending.request_id,
                task_digest=pending.task_digest,
                approved=True,
            ),
        )

    assert calls == 1
    assert resumed.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_encrypted_sqlite_checkpoint_rejects_weak_key(tmp_path):
    with pytest.raises(ValueError, match="16, 24, or 32"):
        async with encrypted_sqlite_checkpointer(tmp_path / "workflow.db", key=b"weak"):
            pass


@pytest.mark.asyncio
async def test_agent_uses_langgraph_backend_without_changing_tool_contract():
    pytest.importorskip("langgraph")
    llm = _WorkflowLLM()
    agent = Agent(
        llm=llm,
        tools=ALL_TOOLS,
        replay=False,
        context_artifacts_enabled=False,
        agent_id="orchestrated-parent",
    )
    try:
        workflow = await agent.run_workflow(
            WorkflowRequest(task=TaskSpec(objective="run configured workflow"))
        )
        assert workflow.backend == WorkflowBackend.LANGGRAPH
        assert workflow.status == TaskStatus.COMPLETED
        assert workflow.final_task_result.parent_id == "orchestrated-parent"

        payload = await agent._tool_by_name["agent"].execute(
            task="preserve the existing agent tool result",
            allowed_tools=[],
        )
        delegated = TaskResult.model_validate_json(payload)
        assert delegated.status == TaskStatus.COMPLETED
        assert delegated.parent_id == "orchestrated-parent"
        assert llm.calls == 2
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_main_agent_chat_runs_through_langgraph_turn_lifecycle():
    llm = _WorkflowLLM()
    agent = Agent(
        llm=llm,
        tools=[],
        replay=False,
        context_artifacts_enabled=False,
    )
    try:
        answer = await agent.chat("run the main turn through the graph")

        assert "workflow integration completed" in answer
        turn = agent.last_turn_workflow
        assert turn is not None
        assert turn.execution.status == TurnStatus.COMPLETED
        assert turn.verified and turn.reviewed
        assert turn.trace == (
            TurnStage.PLAN,
            TurnStage.EXECUTE,
            TurnStage.VERIFY,
            TurnStage.REVIEW,
            TurnStage.FINISHED,
        )
    finally:
        agent.close()

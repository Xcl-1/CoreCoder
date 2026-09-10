"""Optional LangGraph workflow that delegates execution to CoreCoder."""

from __future__ import annotations

import operator
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from ..delegation import TaskResult, TaskSpec, TaskStatus
from .models import (
    ApprovalDecision,
    ApprovalRequest,
    FailureType,
    ReviewResult,
    ReviewVerdict,
    VerificationResult,
    WorkflowBackend,
    WorkflowRequest,
    WorkflowResult,
    WorkflowStage,
)
from .native import TaskExecutor
from .persistence import safe_checkpoint_serializer

Planner = Callable[[WorkflowRequest, str, int], Awaitable[TaskSpec]]
Verifier = Callable[[TaskResult], Awaitable[VerificationResult]]
Reviewer = Callable[[TaskResult, VerificationResult], Awaitable[ReviewResult]]
ApprovalGate = Callable[[TaskSpec], Awaitable[ApprovalRequest | None]]


class _WorkflowState(TypedDict, total=False):
    request: WorkflowRequest
    planned_task: TaskSpec
    task_results: Annotated[list[TaskResult], operator.add]
    verification: VerificationResult
    review: ReviewResult
    approval_request: ApprovalRequest
    approval_granted: bool
    replan_count: int
    feedback: str
    continue_workflow: bool
    terminal_reason: str
    trace: Annotated[list[WorkflowStage], operator.add]


async def _identity_plan(
    request: WorkflowRequest,
    _feedback: str,
    _replan_count: int,
) -> TaskSpec:
    return request.task


async def _runtime_verifier(result: TaskResult) -> VerificationResult:
    passed = result.status == TaskStatus.COMPLETED and result.policy_violations == 0
    feedback = "" if passed else (result.error or "task did not complete safely")
    evidence = tuple(result.evidence[:20])
    return VerificationResult(passed=passed, evidence=evidence, feedback=feedback)


async def _verification_reviewer(
    result: TaskResult,
    verification: VerificationResult,
) -> ReviewResult:
    passed = (
        verification.passed
        and result.status == TaskStatus.COMPLETED
        and result.policy_violations == 0
    )
    return ReviewResult(
        verdict=ReviewVerdict.PASS if passed else ReviewVerdict.REJECT,
        failure_type=FailureType.NONE if passed else FailureType.IMPLEMENTATION,
        feedback="" if passed else (verification.feedback or "verification failed"),
    )


async def _high_risk_approval_gate(task: TaskSpec) -> ApprovalRequest | None:
    unscoped = set(task.allowed_tools) & {"bash", "undo_changes"}
    if task.allow_unscoped_tools or unscoped:
        return ApprovalRequest.for_task(
            task,
            "unscoped delegated tools require explicit workflow approval",
        )
    return None


class LangGraphUnavailableError(RuntimeError):
    """Raised only when the optional backend is selected without its extra."""


class LangGraphOrchestrator:
    """Plan, execute, verify, and review without bypassing TaskController.

    ``execute_task`` must be a controller-owned entry point such as
    ``TaskController.execute`` or ``Agent.delegate``. The graph never receives
    tools, a filesystem handle, or permission-elevation capability.
    """

    def __init__(
        self,
        execute_task: TaskExecutor,
        *,
        planner: Planner | None = None,
        verifier: Verifier | None = None,
        reviewer: Reviewer | None = None,
        approval_gate: ApprovalGate | None = None,
        checkpointer: Any | None = None,
    ):
        self._execute_task = execute_task
        self._planner = planner or _identity_plan
        self._verifier = verifier or _runtime_verifier
        self._reviewer = reviewer or _verification_reviewer
        self._approval_gate = approval_gate or _high_risk_approval_gate
        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    @property
    def backend(self) -> WorkflowBackend:
        return WorkflowBackend.LANGGRAPH

    def _build_graph(self):
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise LangGraphUnavailableError(
                "CoreCoder installation is missing its LangGraph dependencies"
            ) from exc

        builder = StateGraph(_WorkflowState)
        builder.add_node("plan", self._plan)
        builder.add_node("approval", self._approval)
        builder.add_node("execute", self._execute)
        builder.add_node("verify", self._verify)
        builder.add_node("review", self._review)
        builder.add_node("decide", self._decide)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "approval")
        builder.add_conditional_edges(
            "approval",
            self._approval_next,
            {"execute": "execute", "finish": END},
        )
        builder.add_edge("execute", "verify")
        builder.add_edge("verify", "review")
        builder.add_edge("review", "decide")
        builder.add_conditional_edges(
            "decide",
            self._next_node,
            {"replan": "plan", "finish": END},
        )
        checkpointer = self._checkpointer or InMemorySaver(
            serde=safe_checkpoint_serializer()
        )
        return builder.compile(checkpointer=checkpointer)

    async def _plan(self, state: _WorkflowState) -> dict:
        request = state["request"]
        replan_count = state["replan_count"]
        planned_task = await self._planner(request, state["feedback"], replan_count)
        self._validate_plan(request.task, planned_task)
        if replan_count == 0:
            planned_task = planned_task.model_copy(update={"task_id": request.task.task_id})
        else:
            planned_task = planned_task.model_copy(update={
                "task_id": f"task_{uuid.uuid4().hex[:12]}",
                "max_retries": 0,
                "durable": False,
            })
        return {"planned_task": planned_task, "trace": [WorkflowStage.PLAN]}

    async def _approval(self, state: _WorkflowState) -> dict:
        from langgraph.types import interrupt

        approval = await self._approval_gate(state["planned_task"])
        if approval is None:
            return {"approval_granted": True}
        value = interrupt(approval.model_dump(mode="json"))
        try:
            decision = ApprovalDecision.model_validate(value)
        except (TypeError, ValueError):
            return {
                "approval_request": approval,
                "approval_granted": False,
                "terminal_reason": "invalid approval decision",
                "trace": [WorkflowStage.APPROVAL],
            }
        matches = (
            decision.request_id == approval.request_id
            and decision.task_digest == approval.task_digest
        )
        if not matches:
            reason = "approval decision does not match the planned task"
        elif not decision.approved:
            reason = "workflow execution was denied"
        else:
            reason = ""
        return {
            "approval_request": approval,
            "approval_granted": matches and decision.approved,
            "terminal_reason": reason,
            "trace": [WorkflowStage.APPROVAL],
        }

    @staticmethod
    def _approval_next(state: _WorkflowState) -> str:
        return "execute" if state["approval_granted"] else "finish"

    @staticmethod
    def _validate_plan(original: TaskSpec, planned: TaskSpec) -> None:
        """A planner may refine instructions but never broaden authority."""
        if original.task_id != planned.task_id:
            raise ValueError("planner may not replace the controller task identity")
        protected = (
            "role",
            "execution_mode",
            "allowed_tools",
            "read_paths",
            "write_paths",
            "token_budget",
            "max_tool_calls",
            "max_rounds",
            "timeout_seconds",
            "acceptance_criteria",
            "allow_unscoped_tools",
        )
        changed = [name for name in protected if getattr(original, name) != getattr(planned, name)]
        if changed:
            raise ValueError(f"planner may not change task authority: {', '.join(changed)}")

    async def _execute(self, state: _WorkflowState) -> dict:
        result = await self._execute_task(state["planned_task"])
        return {"task_results": [result], "trace": [WorkflowStage.EXECUTE]}

    async def _verify(self, state: _WorkflowState) -> dict:
        verification = await self._verifier(state["task_results"][-1])
        return {"verification": verification, "trace": [WorkflowStage.VERIFY]}

    async def _review(self, state: _WorkflowState) -> dict:
        review = await self._reviewer(
            state["task_results"][-1],
            state["verification"],
        )
        return {"review": review, "trace": [WorkflowStage.REVIEW]}

    async def _decide(self, state: _WorkflowState) -> dict:
        request = state["request"]
        result = state["task_results"][-1]
        review = state["review"]
        reason = ""
        should_replan = review.verdict == ReviewVerdict.REPLAN
        if should_replan and result.policy_violations:
            should_replan = False
            reason = "policy violations cannot be retried"
        elif should_replan and result.status == TaskStatus.BUDGET_EXCEEDED:
            should_replan = False
            reason = "budget-exceeded tasks cannot be retried"
        elif should_replan and request.task.write_paths and request.task.execution_mode.value != "worktree":
            should_replan = False
            reason = "write retries require worktree isolation"
        elif should_replan and state["replan_count"] >= request.max_replans:
            should_replan = False
            reason = "workflow replan limit reached"
        elif should_replan and self._used_tokens(state) >= request.max_total_tokens:
            should_replan = False
            reason = "workflow token budget reached"
        elif review.verdict == ReviewVerdict.REJECT:
            reason = review.feedback or "workflow review rejected"

        update = {
            "continue_workflow": should_replan,
            "terminal_reason": reason,
            "trace": [WorkflowStage.DECIDE],
        }
        if should_replan:
            update.update({
                "replan_count": state["replan_count"] + 1,
                "feedback": review.feedback,
            })
        return update

    @staticmethod
    def _used_tokens(state: _WorkflowState) -> int:
        return sum(
            result.usage.prompt_tokens + result.usage.completion_tokens
            for result in state["task_results"]
        )

    @staticmethod
    def _next_node(state: _WorkflowState) -> str:
        return "replan" if state["continue_workflow"] else "finish"

    async def run(self, request: WorkflowRequest) -> WorkflowResult:
        state = await self._graph.ainvoke({
            "request": request,
            "task_results": [],
            "replan_count": 0,
            "feedback": "",
            "continue_workflow": False,
            "terminal_reason": "",
            "trace": [],
        }, config=self._config(request.workflow_id))
        return self._result_from_state(request.workflow_id, state)

    async def resume(
        self,
        workflow_id: str,
        decision: ApprovalDecision,
    ) -> WorkflowResult:
        from langgraph.types import Command

        state = await self._graph.ainvoke(
            Command(resume=decision.model_dump(mode="json")),
            config=self._config(workflow_id),
        )
        return self._result_from_state(workflow_id, state)

    @staticmethod
    def _config(workflow_id: str) -> dict:
        return {"configurable": {"thread_id": workflow_id}}

    def _result_from_state(self, workflow_id: str, state: dict) -> WorkflowResult:
        interrupts = state.get("__interrupt__", ())
        if interrupts:
            approval = ApprovalRequest.model_validate(interrupts[0].value)
            return WorkflowResult(
                workflow_id=workflow_id,
                backend=self.backend,
                stage=WorkflowStage.APPROVAL,
                status=TaskStatus.INTERRUPTED,
                task_results=state.get("task_results", []),
                replan_count=state.get("replan_count", 0),
                pending_approval=approval,
                trace=state.get("trace", []),
                error=approval.reason,
            )

        if not state.get("task_results"):
            return WorkflowResult(
                workflow_id=workflow_id,
                backend=self.backend,
                stage=WorkflowStage.FINISHED,
                status=TaskStatus.REJECTED,
                replan_count=state.get("replan_count", 0),
                trace=[*state.get("trace", []), WorkflowStage.FINISHED],
                error=state.get("terminal_reason") or "workflow stopped before execution",
            )
        result = state["task_results"][-1]
        verification = state["verification"]
        review = state["review"]
        passed = (
            result.status == TaskStatus.COMPLETED
            and result.policy_violations == 0
            and verification.passed
            and review.verdict == ReviewVerdict.PASS
        )
        error = None if passed else (
            state["terminal_reason"]
            or review.feedback
            or verification.feedback
            or result.error
            or "workflow review rejected"
        )
        return WorkflowResult(
            workflow_id=workflow_id,
            backend=self.backend,
            stage=WorkflowStage.FINISHED,
            status=TaskStatus.COMPLETED if passed else TaskStatus.FAILED,
            task_results=state["task_results"],
            verification=verification,
            review=review,
            replan_count=state["replan_count"],
            trace=[*state["trace"], WorkflowStage.FINISHED],
            error=error,
        )

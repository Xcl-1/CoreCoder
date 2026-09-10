"""Compatibility orchestration backend for the existing execution path."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..delegation import TaskResult, TaskSpec
from .models import (
    ApprovalDecision,
    WorkflowBackend,
    WorkflowRequest,
    WorkflowResult,
    WorkflowStage,
)

TaskExecutor = Callable[[TaskSpec], Awaitable[TaskResult]]


class NativeOrchestrator:
    """Delegate exactly once through the caller-owned TaskController boundary."""

    def __init__(self, execute_task: TaskExecutor):
        self._execute_task = execute_task

    @property
    def backend(self) -> WorkflowBackend:
        return WorkflowBackend.NATIVE

    async def run(self, request: WorkflowRequest) -> WorkflowResult:
        result = await self._execute_task(request.task)
        return WorkflowResult(
            workflow_id=request.workflow_id,
            backend=self.backend,
            stage=WorkflowStage.FINISHED,
            status=result.status,
            task_results=[result],
            trace=[WorkflowStage.EXECUTE, WorkflowStage.FINISHED],
            error=result.error,
        )

    async def resume(
        self,
        workflow_id: str,
        decision: ApprovalDecision,
    ) -> WorkflowResult:
        del workflow_id, decision
        raise RuntimeError("native workflows do not have resumable checkpoints")

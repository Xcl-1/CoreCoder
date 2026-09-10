"""Interface implemented by every CoreCoder workflow backend."""

from __future__ import annotations

from typing import Protocol

from .models import ApprovalDecision, WorkflowBackend, WorkflowRequest, WorkflowResult


class Orchestrator(Protocol):
    """Run workflows without owning tool permissions or task execution."""

    @property
    def backend(self) -> WorkflowBackend: ...

    async def run(self, request: WorkflowRequest) -> WorkflowResult: ...

    async def resume(
        self,
        workflow_id: str,
        decision: ApprovalDecision,
    ) -> WorkflowResult: ...

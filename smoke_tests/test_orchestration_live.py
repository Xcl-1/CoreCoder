"""Opt-in real-provider smoke tests for orchestration backends."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.delegation import TaskSpec, TaskStatus
from corecoder.llm import LLM, LiteLLM
from corecoder.orchestration import (
    ApprovalDecision,
    ApprovalRequest,
    FailureType,
    LangGraphOrchestrator,
    ReviewResult,
    ReviewVerdict,
    VerificationResult,
    WorkflowRequest,
    encrypted_sqlite_checkpointer,
    summarize_workflows,
)
from corecoder.orchestration.native import NativeOrchestrator


async def main() -> None:
    config = Config.from_env()
    if not config.api_key:
        raise RuntimeError("real orchestration smoke test requires a configured API key")
    provider = LiteLLM if config.provider == "litellm" else LLM
    llm = provider(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_tokens=min(config.max_tokens, 1_024),
    )
    agent = Agent(
        llm=llm,
        tools=[],
        replay=False,
        context_artifacts_enabled=False,
        agent_id="live-configured-orchestrator",
    )
    try:
        results = []
        main_answer = await agent.chat(
            "Compute 9 + 8 and answer with the exact result. Do not use tools."
        )
        assert "17" in main_answer, main_answer
        main_turn = agent.last_turn_workflow
        assert main_turn is not None
        assert main_turn.verified and main_turn.reviewed
        results.append({
            "backend": "langgraph-main-chat",
            "status": main_turn.execution.status.value,
            "answer": main_answer,
            "trace": [stage.value for stage in main_turn.trace],
        })

        request = WorkflowRequest(task=TaskSpec(
            objective=(
                "Compute 17 + 25. Return a concise task report whose summary includes "
                "the exact result. Do not use tools."
            ),
            max_rounds=2,
            token_budget=8_000,
            max_tool_calls=0,
            timeout_seconds=120,
            acceptance_criteria=("summary contains the exact arithmetic result",),
        ))
        comparison_results = []
        for backend, runner in (
            ("native", NativeOrchestrator(agent.delegate).run),
            ("langgraph", agent.run_workflow),
        ):
            backend_request = request.model_copy(update={
                "workflow_id": f"{request.workflow_id}-{backend}",
                "task": request.task.model_copy(update={
                    "task_id": f"{request.task.task_id}-{backend}",
                }),
            })
            result = await runner(backend_request)
            assert result.backend.value == backend
            assert result.status == TaskStatus.COMPLETED, result.model_dump()
            assert result.final_task_result is not None
            assert "42" in result.final_task_result.summary, result.model_dump()
            assert result.final_task_result.parent_id == "live-configured-orchestrator"
            comparison_results.append(result)
            results.append({
                "backend": result.backend.value,
                "workflow_id": result.workflow_id,
                "status": result.status.value,
                "task_status": result.final_task_result.status.value,
                "summary": result.final_task_result.summary,
                "trace": [stage.value for stage in result.trace],
                "usage": result.final_task_result.usage.model_dump(),
            })

        background_spec = TaskSpec(
            objective="Compute 6 * 7 and report the exact result without tools.",
            max_rounds=2,
            token_budget=8_000,
            max_tool_calls=0,
            timeout_seconds=120,
        )
        background_task_id = await agent.submit_task(background_spec)
        background_result = await agent.wait_task(background_task_id, timeout=120)
        assert background_result.status == TaskStatus.COMPLETED
        assert "42" in background_result.summary
        results.append({
            "backend": "langgraph-background",
            "task_id": background_task_id,
            "status": background_result.status.value,
            "summary": background_result.summary,
        })

        verification_attempts = 0

        async def replan(request, feedback, _attempt):
            objective = request.task.objective
            if feedback:
                objective = f"{objective} Address this verification feedback: {feedback}"
            return request.task.model_copy(update={"objective": objective})

        async def verify_replan(_result):
            nonlocal verification_attempts
            verification_attempts += 1
            passed = verification_attempts > 1
            return VerificationResult(
                passed=passed,
                feedback="state the arithmetic result explicitly" if not passed else "",
            )

        async def review_replan(_result, verification):
            return ReviewResult(
                verdict=ReviewVerdict.PASS if verification.passed else ReviewVerdict.REPLAN,
                failure_type=(
                    FailureType.NONE
                    if verification.passed
                    else FailureType.INSUFFICIENT_EVIDENCE
                ),
                feedback=verification.feedback,
            )

        replan_result = await LangGraphOrchestrator(
            agent.delegate,
            planner=replan,
            verifier=verify_replan,
            reviewer=review_replan,
        ).run(WorkflowRequest(
            task=TaskSpec(
                objective="Compute 17 + 25 and report the exact result without tools.",
                max_rounds=2,
                token_budget=8_000,
                max_tool_calls=0,
                timeout_seconds=120,
            ),
            max_replans=1,
            max_total_tokens=20_000,
        ))
        assert replan_result.status == TaskStatus.COMPLETED, replan_result.model_dump()
        assert replan_result.replan_count == 1
        assert len(replan_result.task_results) == 2
        assert replan_result.task_results[0].task_id != replan_result.task_results[1].task_id
        results.append({
            "backend": "langgraph-replan",
            "workflow_id": replan_result.workflow_id,
            "status": replan_result.status.value,
            "task_attempts": len(replan_result.task_results),
            "replan_count": replan_result.replan_count,
            "trace": [stage.value for stage in replan_result.trace],
        })

        async def require_approval(task):
            return ApprovalRequest.for_task(task, "live smoke approval")

        approval_request = WorkflowRequest(task=TaskSpec(
            objective="Compute 17 + 25 and report the exact result without tools.",
            max_rounds=2,
            token_budget=8_000,
            max_tool_calls=0,
            timeout_seconds=120,
        ))
        approval_orchestrator = LangGraphOrchestrator(
            agent.delegate,
            approval_gate=require_approval,
        )
        task_count_before = len(agent.tasks.list_tasks(limit=1_000))
        paused = await approval_orchestrator.run(approval_request)
        assert paused.status == TaskStatus.INTERRUPTED
        assert paused.pending_approval is not None
        assert len(agent.tasks.list_tasks(limit=1_000)) == task_count_before
        pending = paused.pending_approval
        approval_result = await approval_orchestrator.resume(
            approval_request.workflow_id,
            ApprovalDecision(
                request_id=pending.request_id,
                task_digest=pending.task_digest,
                approved=True,
            ),
        )
        assert approval_result.status == TaskStatus.COMPLETED, approval_result.model_dump()
        assert len(agent.tasks.list_tasks(limit=1_000)) == task_count_before + 1
        results.append({
            "backend": "langgraph-approval",
            "workflow_id": approval_result.workflow_id,
            "paused_before_execution": True,
            "status": approval_result.status.value,
            "trace": [stage.value for stage in approval_result.trace],
        })

        persistence_root = Path(tempfile.mkdtemp(
            prefix="workflow-",
            dir=Path(".test_runs"),
        ))
        database = persistence_root / "checkpoints.db"
        checkpoint_key = b"0123456789abcdef0123456789abcdef"
        persistent_request = WorkflowRequest(task=TaskSpec(
            objective="Compute 17 + 25 and report the exact result without tools.",
            max_rounds=2,
            token_budget=8_000,
            max_tool_calls=0,
            timeout_seconds=120,
        ))
        async with encrypted_sqlite_checkpointer(
            database,
            key=checkpoint_key,
        ) as first_checkpointer:
            first_instance = LangGraphOrchestrator(
                agent.delegate,
                approval_gate=require_approval,
                checkpointer=first_checkpointer,
            )
            persistent_pause = await first_instance.run(persistent_request)
        assert persistent_pause.status == TaskStatus.INTERRUPTED
        assert persistent_request.task.objective.encode("utf-8") not in database.read_bytes()

        persistent_pending = persistent_pause.pending_approval
        async with encrypted_sqlite_checkpointer(
            database,
            key=checkpoint_key,
        ) as second_checkpointer:
            second_instance = LangGraphOrchestrator(
                agent.delegate,
                approval_gate=require_approval,
                checkpointer=second_checkpointer,
            )
            persistent_result = await second_instance.resume(
                persistent_request.workflow_id,
                ApprovalDecision(
                    request_id=persistent_pending.request_id,
                    task_digest=persistent_pending.task_digest,
                    approved=True,
                ),
            )
        assert persistent_result.status == TaskStatus.COMPLETED
        results.append({
            "backend": "langgraph-persistent-approval",
            "workflow_id": persistent_result.workflow_id,
            "new_instance_resumed": True,
            "checkpoint_encrypted": True,
            "status": persistent_result.status.value,
        })
        comparison = [
            summarize_workflows(result.backend, [result]).model_dump(mode="json")
            for result in comparison_results
        ]
        print(json.dumps(
            {"runs": results, "comparison": comparison},
            ensure_ascii=False,
            indent=2,
        ))
    finally:
        agent.close()
        llm.close()


if __name__ == "__main__":
    asyncio.run(main())

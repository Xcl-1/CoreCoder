"""Encrypted durable delegated-task queue tests."""

import json

import pytest
from cryptography.fernet import Fernet

from corecoder.agent import Agent
from corecoder.delegation import TaskEvent, TaskEventKind, TaskRole, TaskSpec, TaskStatus, WorkspaceMode
from corecoder.models import LLMResponse
from corecoder.task_queue import DurableTaskQueue


class _ResultLLM:
    def chat(self, messages, tools=None, on_token=None):
        return LLMResponse(content=json.dumps({"summary": "recovered durable work"}))


def test_durable_queue_encrypts_prompt_and_rejects_tampering(tmp_path):
    key = Fernet.generate_key()
    queue = DurableTaskQueue(tmp_path, "workspace", key=key)
    spec = TaskSpec(
        objective="PRIVATE DURABLE OBJECTIVE",
        context="PRIVATE DURABLE CONTEXT",
        durable=True,
    )

    path = queue.enqueue(spec)
    raw = path.read_bytes()
    assert b"PRIVATE DURABLE" not in raw
    assert queue.load().specs == (spec,)

    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    loaded = queue.load()
    assert loaded.specs == ()
    assert loaded.invalid_items == 1


def test_durable_queue_enforces_capacity(tmp_path):
    queue = DurableTaskQueue(tmp_path, "bounded", max_items=1)
    queue.enqueue(TaskSpec(objective="first", durable=True))
    with pytest.raises(RuntimeError, match="queue is full"):
        queue.enqueue(TaskSpec(objective="second", durable=True))


@pytest.mark.asyncio
async def test_agent_recovers_authenticated_durable_task_after_restart(tmp_path):
    state_dir = tmp_path / "tasks"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    first = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        agent_id="before-crash",
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    spec = TaskSpec(objective="resume after crash", durable=True)
    submitted = TaskEvent(
        sequence=1,
        timestamp="2026-09-08T00:00:00+00:00",
        event=TaskEventKind.SUBMITTED,
        task_id=spec.task_id,
        agent_id="old-child",
        parent_id=first.agent_id,
        role=TaskRole.EXECUTOR,
        execution_mode=WorkspaceMode.FORK,
        status=TaskStatus.PENDING,
    )
    started = submitted.model_copy(update={
        "sequence": 2,
        "event": TaskEventKind.STARTED,
        "status": TaskStatus.RUNNING,
    })
    first._durable_queue.enqueue(spec)
    first._task_journal.record(submitted)
    first._task_journal.record(started)
    first._task_lease.close()  # Simulate process loss without graceful task cancellation.

    recovered = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        agent_id="after-crash",
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    try:
        task_ids = await recovered.recover_durable_tasks()
        result = await recovered.wait_task(spec.task_id, timeout=1)

        assert task_ids == (spec.task_id,)
        assert result.status == TaskStatus.COMPLETED
        assert result.summary == "recovered durable work"
        assert not recovered._durable_queue.contains(spec.task_id)
    finally:
        first.close()
        recovered.close()


@pytest.mark.asyncio
async def test_durable_submission_requires_persistence():
    agent = Agent(llm=_ResultLLM(), tools=[], replay=False)
    with pytest.raises(RuntimeError, match="require task persistence"):
        await agent.submit_task(TaskSpec(objective="cannot persist", durable=True))


def test_explicit_cancel_removes_durable_item_but_shutdown_preserves_it(tmp_path):
    agent = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        task_state_dir=tmp_path,
    )
    spec = TaskSpec(objective="cancel semantics", durable=True)
    event = TaskEvent(
        sequence=1,
        timestamp="2026-09-08T00:00:00+00:00",
        event=TaskEventKind.CANCEL_REQUESTED,
        task_id=spec.task_id,
        agent_id="child",
        parent_id=agent.agent_id,
        role=spec.role,
        execution_mode=spec.execution_mode,
        status=TaskStatus.RUNNING,
    )
    try:
        agent._durable_queue.enqueue(spec)
        agent._audit_task_event(event)
        assert not agent._durable_queue.contains(spec.task_id)

        agent._durable_queue.enqueue(spec)
        agent._closing = True
        agent._audit_task_event(event)
        assert agent._durable_queue.contains(spec.task_id)
    finally:
        agent.close()

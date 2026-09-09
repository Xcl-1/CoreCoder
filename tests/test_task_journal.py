"""Durable delegated-task journal and restart recovery tests."""

import json
import subprocess
import sys
import time

import pytest

from corecoder.agent import Agent
from corecoder.delegation import (
    TaskEvent,
    TaskEventKind,
    TaskResult,
    TaskRole,
    TaskSpec,
    TaskStatus,
    WorkspaceMode,
)
from corecoder.models import LLMResponse
from corecoder.task_journal import TaskJournal, TaskWorkspaceLease


class _ResultLLM:
    def chat(self, messages, tools=None, on_token=None):
        return LLMResponse(
            content=json.dumps({"summary": "durable result", "evidence": ["verified"]}),
            prompt_tokens=20,
            completion_tokens=5,
        )


@pytest.mark.asyncio
async def test_agent_restores_completed_result_without_persisting_objective(tmp_path):
    state_dir = tmp_path / "tasks"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    first = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        session_id="session-before-restart",
        agent_id="parent-before-restart",
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    spec = TaskSpec(objective="PRIVATE OBJECTIVE MUST NOT BE JOURNALED")

    original = await first.delegate(spec)
    journal_path = next(state_dir.glob("tasks_*.jsonl"))
    assert "PRIVATE OBJECTIVE" not in journal_path.read_text(encoding="utf-8")

    restored_agent = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        session_id="session-after-restart",
        agent_id="parent-after-restart",
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    restored = restored_agent.tasks.result(spec.task_id)
    snapshot = restored_agent.tasks.snapshot(spec.task_id)

    assert restored == original
    assert snapshot.status == TaskStatus.COMPLETED
    assert snapshot.parent_id == "parent-before-restart"
    assert await restored_agent.wait_task(spec.task_id) == original
    events = restored_agent.tasks.event_batch(task_id=spec.task_id).events
    assert all(event.sequence > 0 for event in events)
    assert [event.sequence for event in events] == sorted({event.sequence for event in events})


def test_restart_marks_unfinished_journal_task_interrupted_once(tmp_path):
    journal = TaskJournal(tmp_path, "interrupted-session")
    task_id = "task_interrupted"
    submitted = TaskEvent(
        timestamp="2026-09-08T00:00:00+00:00",
        event=TaskEventKind.SUBMITTED,
        task_id=task_id,
        agent_id="child-before-restart",
        parent_id="parent-before-restart",
        role=TaskRole.EXECUTOR,
        execution_mode=WorkspaceMode.WORKTREE,
        status=TaskStatus.PENDING,
    )
    started = submitted.model_copy(update={
        "timestamp": "2026-09-08T00:00:01+00:00",
        "event": TaskEventKind.STARTED,
        "status": TaskStatus.RUNNING,
    })
    journal.record(submitted)
    journal.record(started)

    recovered = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        session_id="interrupted-session",
        task_state_dir=tmp_path,
        task_journal_id="interrupted-session",
    )
    result = recovered.tasks.result(task_id)

    assert result.status == TaskStatus.INTERRUPTED
    assert "not automatically resumed" in result.risks[0]
    assert recovered.tasks.events(task_id=task_id)[-1].event == TaskEventKind.INTERRUPTED

    loaded_again = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        session_id="interrupted-session",
        task_state_dir=tmp_path,
        task_journal_id="interrupted-session",
    )
    events = loaded_again.tasks.events(task_id=task_id)
    assert sum(event.event == TaskEventKind.INTERRUPTED for event in events) == 1


def test_task_journal_is_traversal_safe_bounded_and_corruption_tolerant(tmp_path):
    journal = TaskJournal(tmp_path, "../../outside", max_records=8)
    event = TaskEvent(
        timestamp="2026-09-08T00:00:00+00:00",
        event=TaskEventKind.SUBMITTED,
        task_id="task_safe",
        agent_id="child",
        parent_id="parent",
        role=TaskRole.RESEARCHER,
        execution_mode=WorkspaceMode.FORK,
        status=TaskStatus.PENDING,
    )
    for _ in range(108):
        journal.record(event)
    with journal.path.open("a", encoding="utf-8") as stream:
        stream.write("{partial\n")

    loaded = journal.load()

    assert journal.path.parent == tmp_path.resolve()
    assert len(loaded.records) == 8
    assert loaded.invalid_lines == 1
    assert loaded.total_lines <= 16


def test_workspace_lease_has_single_owner_and_safe_release(tmp_path):
    first = TaskWorkspaceLease(
        tmp_path,
        "workspace",
        "agent-one",
        heartbeat_interval=0.05,
    )
    second = TaskWorkspaceLease(
        tmp_path,
        "workspace",
        "agent-two",
        heartbeat_interval=0.05,
    )
    try:
        assert first.acquire()
        assert first.owns
        assert not second.acquire()
        assert second.owner().agent_id == "agent-one"

        first.close()
        assert second.acquire()
        assert second.owns
        first.close()  # A former owner must not remove the successor's lease.
        assert second.owns
    finally:
        first.close()
        second.close()


def test_workspace_lease_blocks_live_process_and_reclaims_dead_owner(tmp_path):
    ready = tmp_path / "ready"
    helper = (
        "import pathlib,sys,time\n"
        "from corecoder.task_journal import TaskWorkspaceLease\n"
        "lease=TaskWorkspaceLease(sys.argv[1],'cross-process','helper',heartbeat_interval=.05)\n"
        "assert lease.acquire()\n"
        "pathlib.Path(sys.argv[2]).write_text('ready',encoding='ascii')\n"
        "time.sleep(30)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", helper, str(tmp_path), str(ready)])
    contender = TaskWorkspaceLease(
        tmp_path,
        "cross-process",
        "contender",
        heartbeat_interval=0.05,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        assert not contender.acquire()
        assert contender.owner().process_id == process.pid

        process.kill()
        process.wait(timeout=5)
        assert contender.acquire()
        assert contender.owns
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        contender.close()


@pytest.mark.asyncio
async def test_agent_observes_live_owner_then_explicitly_claims_released_lease(tmp_path):
    state_dir = tmp_path / "tasks"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    owner = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        agent_id="lease-owner",
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    task_id = "task_owned_elsewhere"
    submitted = TaskEvent(
        sequence=1,
        timestamp="2026-09-08T00:00:00+00:00",
        event=TaskEventKind.SUBMITTED,
        task_id=task_id,
        agent_id="child-owner",
        parent_id=owner.agent_id,
        role=TaskRole.EXECUTOR,
        execution_mode=WorkspaceMode.FORK,
        status=TaskStatus.PENDING,
    )
    started = submitted.model_copy(update={
        "sequence": 2,
        "event": TaskEventKind.STARTED,
        "status": TaskStatus.RUNNING,
    })
    owner._task_journal.record(submitted)
    owner._task_journal.record(started)

    observer = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        agent_id="lease-observer",
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    try:
        assert owner.owns_task_scheduler
        assert not observer.owns_task_scheduler
        assert observer.tasks.status(task_id) == TaskStatus.RUNNING
        assert observer.tasks.result(task_id) is None
        assert not observer.cancel_task(task_id)
        with pytest.raises(RuntimeError, match="owned by another process"):
            await observer.delegate(TaskSpec(objective="must not run"))

        completed_result = TaskResult(
            task_id=task_id,
            agent_id="child-owner",
            parent_id=owner.agent_id,
            role=TaskRole.EXECUTOR,
            execution_mode=WorkspaceMode.FORK,
            status=TaskStatus.COMPLETED,
            summary="finished by lease owner",
        )
        completed = started.model_copy(update={
            "sequence": 3,
            "event": TaskEventKind.COMPLETED,
            "status": TaskStatus.COMPLETED,
        })
        owner._task_journal.record(completed, completed_result)
        observer.refresh_task_state()
        assert observer.tasks.result(task_id) == completed_result

        abandoned_id = "task_abandoned_after_release"
        abandoned = submitted.model_copy(update={
            "sequence": 4,
            "task_id": abandoned_id,
            "status": TaskStatus.PENDING,
        })
        owner._task_journal.record(abandoned)
        observer.refresh_task_state()
        assert observer.tasks.status(abandoned_id) == TaskStatus.PENDING

        owner.close()
        assert observer.claim_task_scheduler()
        assert observer.owns_task_scheduler
        assert observer.tasks.result(task_id).status == TaskStatus.COMPLETED
        assert observer.tasks.result(abandoned_id).status == TaskStatus.INTERRUPTED
    finally:
        owner.close()
        observer.close()

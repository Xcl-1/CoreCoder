"""Standalone durable queue worker tests."""

import asyncio
import json
import threading
import time

import pytest

import corecoder.cli as cli_module
from corecoder.agent import Agent
from corecoder.delegation import TaskSpec, TaskStatus
from corecoder.models import LLMResponse
from corecoder.tools.changes import ChangeTracker
from corecoder.worker import DurableTaskWorker, DurableTaskWorkerPool


class _ResultLLM:
    def chat(self, messages, tools=None, on_token=None):
        return LLMResponse(content=json.dumps({"summary": "worker finished"}))


class _FailedLLM:
    def chat(self, messages, tools=None, on_token=None):
        return LLMResponse(content="Error: simulated child failure")


class _ConcurrencyProbeLLM:
    def __init__(self, probe):
        self.probe = probe

    def chat(self, messages, tools=None, on_token=None):
        with self.probe["lock"]:
            self.probe["active"] += 1
            self.probe["maximum"] = max(self.probe["maximum"], self.probe["active"])
        time.sleep(0.05)
        with self.probe["lock"]:
            self.probe["active"] -= 1
        return LLMResponse(content=json.dumps({"summary": "bounded"}))


class _BlockingLLM:
    def __init__(self, started: threading.Event, release: threading.Event):
        self.started = started
        self.release = release

    def chat(self, messages, tools=None, on_token=None):
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test did not release worker")
        return LLMResponse(content=json.dumps({"summary": "released"}))


@pytest.mark.asyncio
async def test_worker_consumes_task_enqueued_by_observer_process(tmp_path):
    state_dir = tmp_path / "tasks"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    owner = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        agent_id="worker-owner",
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    producer = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        agent_id="queue-producer",
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    spec = TaskSpec(objective="queued from another process", durable=True)
    try:
        task_id = await producer.submit_task(spec)
        assert task_id == spec.task_id
        assert producer.tasks.snapshot(task_id) is None

        stats = await DurableTaskWorker(owner, poll_interval=0.05).run(once=True)
        observed = await producer.wait_task(task_id, timeout=1)

        assert stats.scans == 1
        assert stats.scheduled == 1
        assert stats.completed == 1
        assert stats.succeeded == 1
        assert stats.failed == 0
        assert observed.status == TaskStatus.COMPLETED
        assert observed.summary == "worker finished"
    finally:
        producer.close()
        owner.close()


@pytest.mark.asyncio
async def test_worker_requires_persistence_and_scheduler_ownership(tmp_path):
    disabled = Agent(llm=_ResultLLM(), tools=[], replay=False)
    with pytest.raises(RuntimeError, match="persistence is disabled"):
        await DurableTaskWorker(disabled).run(once=True)

    workspace = tmp_path / "repo"
    workspace.mkdir()
    owner = Agent(
        llm=_ResultLLM(), tools=[], replay=False,
        task_state_dir=tmp_path / "tasks", workspace_root=workspace,
    )
    observer = Agent(
        llm=_ResultLLM(), tools=[], replay=False,
        task_state_dir=tmp_path / "tasks", workspace_root=workspace,
    )
    try:
        with pytest.raises(RuntimeError, match="owned by another process"):
            await DurableTaskWorker(observer).run(once=True)
    finally:
        observer.close()
        owner.close()


def test_once_worker_reports_failed_task_and_returns_nonzero(tmp_path):
    state_dir = tmp_path / "tasks"
    workspace = tmp_path / "repo"
    workspace.mkdir()
    owner = Agent(
        llm=_FailedLLM(),
        tools=[],
        replay=False,
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    producer = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        task_state_dir=state_dir,
        workspace_root=workspace,
    )
    spec = TaskSpec(objective="fail in worker", durable=True)
    try:
        asyncio.run(producer.submit_task(spec))
        exit_code = cli_module._run_worker(owner, once=True, poll_interval=0.05)

        assert exit_code == 1
        result = asyncio.run(producer.wait_task(spec.task_id, timeout=1))
        assert result.status == TaskStatus.FAILED
    finally:
        producer.close()
        owner.close()


def test_worker_poll_interval_is_bounded():
    with pytest.raises(ValueError, match="poll_interval"):
        DurableTaskWorker(None, poll_interval=0.01)


@pytest.mark.asyncio
async def test_worker_cancellation_marks_shutdown_before_loop_teardown(tmp_path):
    agent = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        task_state_dir=tmp_path,
    )
    execution = asyncio.create_task(DurableTaskWorker(agent, poll_interval=1).run())
    await asyncio.sleep(0)
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution
    try:
        assert agent._closing is True
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_worker_pool_drains_independent_workspaces_fairly(tmp_path):
    state_dir = tmp_path / "tasks"
    workspaces = [tmp_path / "repo-a", tmp_path / "repo-b"]
    for workspace in workspaces:
        workspace.mkdir()
    workers = [
        Agent(
            llm=_ResultLLM(),
            replay=False,
            task_state_dir=state_dir,
            workspace_root=workspace,
        )
        for workspace in workspaces
    ]
    producers = [
        Agent(
            llm=_ResultLLM(),
            tools=[],
            replay=False,
            task_state_dir=state_dir,
            workspace_root=workspace,
        )
        for workspace in workspaces
    ]
    specs = [
        TaskSpec(objective=f"work in {workspace.name}", durable=True)
        for workspace in workspaces
    ]
    try:
        for producer, spec in zip(producers, specs, strict=True):
            await producer.submit_task(spec)

        stats = await DurableTaskWorkerPool(workers, poll_interval=0.05).run(once=True)

        assert stats.scans == 1
        assert stats.workspace_scans == 2
        assert stats.scheduled == 2
        assert stats.completed == 2
        assert stats.succeeded == 2
        assert stats.failed == 0
        assert stats.claimed_workspaces == 2
        assert stats.skipped_workspaces == 0
        assert stats.errors == 0
        for producer, spec in zip(producers, specs, strict=True):
            result = await producer.wait_task(spec.task_id, timeout=1)
            assert result.status == TaskStatus.COMPLETED
    finally:
        for producer in producers:
            producer.close()
        for worker in workers:
            worker.close()


@pytest.mark.asyncio
async def test_worker_pool_skips_workspace_claimed_by_another_worker(tmp_path):
    state_dir = tmp_path / "tasks"
    busy_workspace = tmp_path / "busy"
    available_workspace = tmp_path / "available"
    busy_workspace.mkdir()
    available_workspace.mkdir()
    existing_owner = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        task_state_dir=state_dir,
        workspace_root=busy_workspace,
    )
    candidates = [
        Agent(
            llm=_ResultLLM(),
            tools=[],
            replay=False,
            task_state_dir=state_dir,
            workspace_root=workspace,
        )
        for workspace in (busy_workspace, available_workspace)
    ]
    producer = Agent(
        llm=_ResultLLM(),
        tools=[],
        replay=False,
        task_state_dir=state_dir,
        workspace_root=available_workspace,
    )
    spec = TaskSpec(objective="claimed by pool", durable=True)
    try:
        await producer.submit_task(spec)
        stats = await DurableTaskWorkerPool(candidates).run(once=True)

        assert stats.claimed_workspaces == 1
        assert stats.skipped_workspaces == 1
        assert stats.scheduled == 1
        assert (await producer.wait_task(spec.task_id, timeout=1)).status == TaskStatus.COMPLETED
    finally:
        producer.close()
        for candidate in candidates:
            candidate.close()
        existing_owner.close()


@pytest.mark.asyncio
async def test_worker_pool_counts_failed_results_separately(tmp_path):
    state_dir = tmp_path / "tasks"
    workers = []
    producers = []
    try:
        for index, llm in enumerate((_ResultLLM(), _FailedLLM())):
            workspace = tmp_path / f"result-{index}"
            workspace.mkdir()
            workers.append(Agent(
                llm=llm,
                tools=[],
                replay=False,
                task_state_dir=state_dir,
                workspace_root=workspace,
            ))
            producers.append(Agent(
                llm=_ResultLLM(),
                tools=[],
                replay=False,
                task_state_dir=state_dir,
                workspace_root=workspace,
            ))
            await producers[-1].submit_task(
                TaskSpec(objective=f"pool result {index}", durable=True)
            )

        stats = await DurableTaskWorkerPool(workers).run(once=True)

        assert stats.completed == 2
        assert stats.succeeded == 1
        assert stats.failed == 1
        assert stats.errors == 0
    finally:
        for producer in producers:
            producer.close()
        for worker in workers:
            worker.close()


def test_worker_pool_rejects_duplicate_workspaces(tmp_path):
    agent = Agent(llm=_ResultLLM(), tools=[], replay=False, workspace_root=tmp_path)
    try:
        with pytest.raises(ValueError, match="unique"):
            DurableTaskWorkerPool([agent, agent])
    finally:
        agent.close()


def test_top_level_agents_receive_isolated_tool_registries(tmp_path):
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    tracker = ChangeTracker()
    first = Agent(
        llm=_ResultLLM(),
        replay=False,
        changes=tracker,
        workspace_root=first_workspace,
    )
    second = Agent(llm=_ResultLLM(), replay=False, workspace_root=second_workspace)
    try:
        assert first.changes is tracker
        assert f"- Working directory: {first_workspace.resolve()}" in first._system
        assert first._tool_by_name["agent"] is not second._tool_by_name["agent"]
        assert first._tool_by_name["agent"]._parent_agent is first
        assert second._tool_by_name["agent"]._parent_agent is second
    finally:
        second.close()
        first.close()


@pytest.mark.asyncio
async def test_worker_pool_cancellation_preserves_every_workspace(tmp_path):
    agents = []
    for name in ("a", "b"):
        workspace = tmp_path / name
        workspace.mkdir()
        agents.append(Agent(
            llm=_ResultLLM(),
            tools=[],
            replay=False,
            task_state_dir=tmp_path / "tasks",
            workspace_root=workspace,
        ))
    execution = asyncio.create_task(
        DurableTaskWorkerPool(agents, poll_interval=1).run()
    )
    await asyncio.sleep(0)
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution
    try:
        assert all(agent._closing for agent in agents)
    finally:
        for agent in agents:
            agent.close()


@pytest.mark.asyncio
async def test_worker_pool_enforces_global_execution_limit(tmp_path):
    probe = {"lock": threading.Lock(), "active": 0, "maximum": 0}
    state_dir = tmp_path / "tasks"
    workers = []
    producers = []
    try:
        for index in range(3):
            workspace = tmp_path / f"repo-{index}"
            workspace.mkdir()
            workers.append(Agent(
                llm=_ConcurrencyProbeLLM(probe),
                tools=[],
                replay=False,
                task_state_dir=state_dir,
                workspace_root=workspace,
            ))
            producers.append(Agent(
                llm=_ResultLLM(),
                tools=[],
                replay=False,
                task_state_dir=state_dir,
                workspace_root=workspace,
            ))
            await producers[-1].submit_task(
                TaskSpec(objective=f"bounded task {index}", durable=True)
            )

        stats = await DurableTaskWorkerPool(
            workers,
            max_concurrency=1,
        ).run(once=True)

        assert stats.completed == 3
        assert probe["maximum"] == 1
    finally:
        for producer in producers:
            producer.close()
        for worker in workers:
            worker.close()


@pytest.mark.asyncio
async def test_pool_limited_task_stays_pending_until_execution_slot_is_available(tmp_path):
    started = threading.Event()
    release = threading.Event()
    state_dir = tmp_path / "tasks"
    workers = []
    producers = []
    specs = []
    try:
        for index in range(2):
            workspace = tmp_path / f"pending-{index}"
            workspace.mkdir()
            workers.append(Agent(
                llm=_BlockingLLM(started, release),
                tools=[],
                replay=False,
                task_state_dir=state_dir,
                workspace_root=workspace,
            ))
            producers.append(Agent(
                llm=_ResultLLM(),
                tools=[],
                replay=False,
                task_state_dir=state_dir,
                workspace_root=workspace,
            ))
            spec = TaskSpec(objective=f"pending task {index}", durable=True)
            specs.append(spec)
            await producers[-1].submit_task(spec)

        execution = asyncio.create_task(
            DurableTaskWorkerPool(workers, max_concurrency=1).run(once=True)
        )
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.sleep(0)

        statuses = [worker.tasks.status(spec.task_id) for worker, spec in zip(
            workers, specs, strict=True
        )]
        assert sorted(status.value for status in statuses if status is not None) == [
            "pending",
            "running",
        ]

        release.set()
        assert (await execution).completed == 2
    finally:
        release.set()
        for producer in producers:
            producer.close()
        for worker in workers:
            worker.close()

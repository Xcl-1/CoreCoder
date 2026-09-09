"""Local fault-injection and multi-workspace worker soak acceptance."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from corecoder.agent import Agent
from corecoder.delegation import TaskSpec, TaskStatus
from corecoder.models import LLMResponse
from corecoder.task_queue import DurableTaskQueue
from corecoder.worker import DurableTaskWorker, DurableTaskWorkerPool

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / ".test_runs" / "worker-resilience"


class _SuccessLLM:
    def __init__(self, probe: dict | None = None):
        self.probe = probe

    def chat(self, messages, tools=None, on_token=None):
        if self.probe is not None:
            with self.probe["lock"]:
                self.probe["active"] += 1
                self.probe["peak"] = max(self.probe["peak"], self.probe["active"])
            time.sleep(0.02)
            with self.probe["lock"]:
                self.probe["active"] -= 1
        return LLMResponse(content='{"summary":"local resilience task completed"}')


class _BlockingLLM:
    def __init__(self, marker: Path):
        self.marker = marker

    def chat(self, messages, tools=None, on_token=None):
        self.marker.write_text("started\n", encoding="utf-8")
        while True:
            time.sleep(1)


def _agent(state: Path, workspace: Path, llm, agent_id: str) -> Agent:
    return Agent(
        llm=llm,
        tools=[],
        replay=False,
        agent_id=agent_id,
        task_state_dir=state,
        workspace_root=workspace,
        task_lease_stale_seconds=5,
    )


async def _run_blocking_worker(state: Path, workspace: Path, marker: Path) -> None:
    agent = _agent(state, workspace, _BlockingLLM(marker), "crash-owner")
    try:
        await DurableTaskWorker(agent, poll_interval=0.05).run(once=True)
    finally:
        agent.close()


async def _run_contender(state: Path, workspace: Path) -> int:
    agent = _agent(state, workspace, _SuccessLLM(), "lease-contender")
    try:
        try:
            await DurableTaskWorker(agent).run(once=True)
        except RuntimeError as exc:
            return 0 if "owned by another process" in str(exc) else 3
        return 2
    finally:
        agent.close()


async def _wait_for(path: Path, process: asyncio.subprocess.Process, timeout: float = 10) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if path.exists():
            return
        if process.returncode is not None:
            raise RuntimeError(f"blocking worker exited early with {process.returncode}")
        await asyncio.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


async def _crash_and_recover(run_root: Path) -> dict:
    state = run_root / "crash-state"
    workspace = run_root / "crash-workspace"
    workspace.mkdir()
    marker = run_root / "child-started.marker"
    spec = TaskSpec(
        task_id="task_crash_recovery",
        objective="survive a forcefully terminated worker",
        durable=True,
    )
    queue = DurableTaskQueue(state, str(workspace.resolve()))
    queue.enqueue(spec)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "blocking-worker",
        str(state),
        str(workspace),
        str(marker),
    ]
    process = await asyncio.create_subprocess_exec(*command, cwd=ROOT)
    try:
        await _wait_for(marker, process)
        contender = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            "contender",
            str(state),
            str(workspace),
            cwd=ROOT,
        )
        assert await contender.wait() == 0
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=10)
    finally:
        if process.returncode is None:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=10)

    assert queue.contains(spec.task_id)
    successor = _agent(state, workspace, _SuccessLLM(), "crash-successor")
    try:
        stats = await DurableTaskWorker(successor).run(once=True)
        result = await successor.wait_task(spec.task_id, timeout=2)
        assert stats.succeeded == 1 and stats.failed == 0
        assert result.status == TaskStatus.COMPLETED
        assert not queue.contains(spec.task_id)
    finally:
        successor.close()
    assert not list(state.glob("*.lease"))
    return {"contender_rejected": True, "recovered": True}


async def _multi_workspace_soak(run_root: Path) -> dict:
    state = run_root / "soak-state"
    workspaces = []
    task_count = 30
    for index in range(5):
        workspace = run_root / f"workspace-{index}"
        workspace.mkdir()
        workspaces.append(workspace)
        queue = DurableTaskQueue(state, str(workspace.resolve()))
        for task_index in range(6):
            queue.enqueue(TaskSpec(
                task_id=f"task_soak_{index}_{task_index}",
                objective=f"local soak task {index}/{task_index}",
                durable=True,
            ))

    probe = {"lock": threading.Lock(), "active": 0, "peak": 0}
    agents = [
        _agent(state, workspace, _SuccessLLM(probe), f"soak-worker-{index}")
        for index, workspace in enumerate(workspaces)
    ]
    try:
        stats = await DurableTaskWorkerPool(
            agents,
            poll_interval=0.05,
            max_concurrency=3,
        ).run(once=True)
        assert stats.scheduled == task_count
        assert stats.completed == task_count
        assert stats.succeeded == task_count
        assert stats.failed == 0 and stats.errors == 0
        assert probe["peak"] == 3
        for workspace in workspaces:
            assert not DurableTaskQueue(state, str(workspace.resolve())).load().specs
    finally:
        for agent in agents:
            agent.close()
    assert not list(state.glob("*.lease"))
    return {"tasks": task_count, "workspaces": len(workspaces), "peak_concurrency": probe["peak"]}


async def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="run-", dir=RUNS))
    crash = await _crash_and_recover(run_root)
    soak = await _multi_workspace_soak(run_root)
    print(json.dumps({"crash_recovery": crash, "soak": soak}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "blocking-worker":
        asyncio.run(_run_blocking_worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])))
    elif len(sys.argv) > 1 and sys.argv[1] == "contender":
        raise SystemExit(asyncio.run(_run_contender(Path(sys.argv[2]), Path(sys.argv[3]))))
    else:
        asyncio.run(main())

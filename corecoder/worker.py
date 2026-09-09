"""Standalone consumer for encrypted durable delegated-task queues."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

from .agent import Agent
from .delegation import TaskStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerStats:
    scans: int = 0
    scheduled: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0


@dataclass(frozen=True)
class WorkerPoolStats:
    """Bounded aggregate statistics for a multi-workspace worker run."""

    scans: int = 0
    workspace_scans: int = 0
    scheduled: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    claimed_workspaces: int = 0
    skipped_workspaces: int = 0
    errors: int = 0


class DurableTaskWorker:
    """Poll one workspace queue and execute it through the normal Agent controller."""

    def __init__(self, agent: Agent, *, poll_interval: float = 1.0):
        if not 0.05 <= poll_interval <= 300:
            raise ValueError("worker poll_interval must be between 0.05 and 300 seconds")
        self.agent = agent
        self.poll_interval = poll_interval

    async def run(self, *, once: bool = False) -> WorkerStats:
        if not self.agent.durable_task_queue_enabled:
            raise RuntimeError("durable task persistence is disabled")
        if not self.agent.owns_task_scheduler:
            raise RuntimeError("workspace task scheduler lease is owned by another process")
        scans = scheduled = completed = succeeded = failed = 0
        try:
            while True:
                task_ids = await self.agent.recover_durable_tasks()
                scans += 1
                scheduled += len(task_ids)
                if once:
                    for task_id in task_ids:
                        result = await self.agent.wait_task(task_id)
                        completed += 1
                        if result.status == TaskStatus.COMPLETED:
                            succeeded += 1
                        else:
                            failed += 1
                    return WorkerStats(
                        scans=scans,
                        scheduled=scheduled,
                        completed=completed,
                        succeeded=succeeded,
                        failed=failed,
                    )
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            self.agent.prepare_shutdown()
            raise


class DurableTaskWorkerPool:
    """Fairly scan independent workspace queues under per-workspace leases.

    Every workspace retains its own Agent, controller, security boundary and
    lease. Scans happen concurrently, preventing a busy workspace from
    delaying admission in another one. A lease held by another process is
    skipped, which also provides the cross-process task-claim mechanism.
    """

    def __init__(
        self,
        agents: Iterable[Agent],
        *,
        poll_interval: float = 1.0,
        max_concurrency: int = 4,
    ):
        if not 0.05 <= poll_interval <= 300:
            raise ValueError("worker poll_interval must be between 0.05 and 300 seconds")
        self.agents = tuple(agents)
        if not self.agents:
            raise ValueError("worker pool requires at least one workspace agent")
        if not 1 <= max_concurrency <= 32:
            raise ValueError("worker pool max_concurrency must be between 1 and 32")
        identities = [os.path.normcase(str(agent.workspace_root)) for agent in self.agents]
        if len(set(identities)) != len(identities):
            raise ValueError("worker pool workspace roots must be unique")
        self.poll_interval = poll_interval
        self.max_concurrency = max_concurrency
        self._execution_limiter = asyncio.Semaphore(max_concurrency)
        for agent in self.agents:
            agent.set_task_execution_limiter(self._execution_limiter)

    async def run(self, *, once: bool = False) -> WorkerPoolStats:
        disabled = [
            str(agent.workspace_root)
            for agent in self.agents
            if not agent.durable_task_queue_enabled
        ]
        if disabled:
            raise RuntimeError(
                "durable task persistence is disabled for workspace(s): "
                + ", ".join(disabled)
            )

        claimed = self._claim_available()
        if not claimed:
            raise RuntimeError("all workspace task scheduler leases are owned by other processes")

        scans = workspace_scans = scheduled = completed = succeeded = failed = errors = 0
        ever_claimed = {str(agent.workspace_root) for agent in claimed}
        try:
            while True:
                claimed = self._claim_available()
                ever_claimed.update(str(agent.workspace_root) for agent in claimed)
                if not claimed:
                    raise RuntimeError("all workspace task scheduler leases were lost")

                batches = await asyncio.gather(
                    *(agent.recover_durable_tasks() for agent in claimed),
                    return_exceptions=True,
                )
                scans += 1
                workspace_scans += len(claimed)
                pending: list[tuple[Agent, str]] = []
                for agent, batch in zip(claimed, batches, strict=True):
                    if isinstance(batch, BaseException):
                        errors += 1
                        logger.warning(
                            "Failed to scan durable queue for %s",
                            agent.workspace_root,
                            exc_info=(type(batch), batch, batch.__traceback__),
                        )
                        continue
                    scheduled += len(batch)
                    pending.extend((agent, task_id) for task_id in batch)

                if once:
                    results = await asyncio.gather(
                        *(agent.wait_task(task_id) for agent, task_id in pending),
                        return_exceptions=True,
                    )
                    finished = [
                        result for result in results if not isinstance(result, BaseException)
                    ]
                    completed += len(finished)
                    succeeded += sum(
                        result.status == TaskStatus.COMPLETED for result in finished
                    )
                    failed += sum(
                        result.status != TaskStatus.COMPLETED for result in finished
                    )
                    errors += sum(isinstance(result, BaseException) for result in results)
                    return WorkerPoolStats(
                        scans=scans,
                        workspace_scans=workspace_scans,
                        scheduled=scheduled,
                        completed=completed,
                        succeeded=succeeded,
                        failed=failed,
                        claimed_workspaces=len(ever_claimed),
                        skipped_workspaces=len(self.agents) - len(ever_claimed),
                        errors=errors,
                    )
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            for agent in self.agents:
                agent.prepare_shutdown()
            raise

    def _claim_available(self) -> tuple[Agent, ...]:
        return tuple(
            agent
            for agent in self.agents
            if agent.owns_task_scheduler or agent.claim_task_scheduler()
        )

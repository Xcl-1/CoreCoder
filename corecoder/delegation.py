"""Centralised, least-authority sub-agent task orchestration.

The controller in this module deliberately knows nothing about prompts or LLMs.
It owns lifecycle, concurrency, timeout, cancellation and result validation; the
parent :class:`corecoder.agent.Agent` remains the only component that can create
and accept a child agent's work.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class TaskRole(str, enum.Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"


class WorkspaceMode(str, enum.Enum):
    """Execution backends understood by the protocol.

    Keeping both backends in the protocol prevents worktree/team support from
    becoming a second orchestration API.
    """

    FORK = "fork"
    WORKTREE = "worktree"


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"
    BUDGET_EXCEEDED = "budget_exceeded"


class TaskEventKind(str, enum.Enum):
    """Trusted lifecycle transitions emitted by the task control plane."""

    SUBMITTED = "submitted"
    STARTED = "started"
    RETRYING = "retrying"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"
    BUDGET_EXCEEDED = "budget_exceeded"
    CIRCUIT_OPENED = "circuit_opened"
    ACCEPTED = "accepted"
    WORKSPACE_READY = "workspace_ready"
    TOOL_STARTED = "tool_started"
    REPORT_RECEIVED = "report_received"
    MERGE_STARTED = "merge_started"


_READ_TOOLS = frozenset({"read_file", "grep", "glob", "retrieve_context"})
_PATH_READ_TOOLS = frozenset({"read_file", "grep", "glob"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "edit_ast"})
_UNSCOPED_TOOLS = frozenset({"bash", "undo_changes", "agent"})
_PROGRESS_EVENTS = frozenset({
    TaskEventKind.WORKSPACE_READY,
    TaskEventKind.TOOL_STARTED,
    TaskEventKind.REPORT_RECEIVED,
    TaskEventKind.MERGE_STARTED,
})
_TERMINAL_STATUSES = frozenset({
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.TIMED_OUT,
    TaskStatus.CANCELLED,
    TaskStatus.INTERRUPTED,
    TaskStatus.REJECTED,
    TaskStatus.BUDGET_EXCEEDED,
})


def _task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"


class TaskSpec(BaseModel):
    """Complete authority and resource envelope for one delegated task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(default_factory=_task_id, min_length=1, max_length=80)
    objective: str = Field(min_length=1, max_length=16_000)
    role: TaskRole = TaskRole.EXECUTOR
    execution_mode: WorkspaceMode = WorkspaceMode.FORK
    context: str = Field(default="", max_length=16_000)
    allowed_tools: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    token_budget: int = Field(default=16_000, ge=256, le=1_000_000)
    max_tool_calls: int = Field(default=30, ge=0, le=1_000)
    max_rounds: int = Field(default=12, ge=1, le=50)
    timeout_seconds: float = Field(default=300.0, gt=0, le=3_600)
    max_retries: int = Field(default=0, ge=0, le=3)
    acceptance_criteria: tuple[str, ...] = ()
    # Shell and undo cannot be confined by file-path argument checks.  They
    # require an explicit opt-in and are intentionally not exposed by AgentTool.
    allow_unscoped_tools: bool = False
    durable: bool = False

    @model_validator(mode="after")
    def validate_authority_envelope(self) -> TaskSpec:
        tools = set(self.allowed_tools)
        if len(tools) != len(self.allowed_tools):
            raise ValueError("allowed_tools must not contain duplicates")
        if "agent" in tools:
            raise ValueError("delegated tasks may not use the agent tool")
        if "task_control" in tools:
            raise ValueError("delegated tasks may not use the task control tool")
        unscoped = tools & _UNSCOPED_TOOLS
        if unscoped and not self.allow_unscoped_tools:
            names = ", ".join(sorted(unscoped))
            raise ValueError(f"unscoped tools require explicit opt-in: {names}")
        if tools & _READ_TOOLS and not self.read_paths:
            raise ValueError("read_paths is required when read tools are allowed")
        if tools & _WRITE_TOOLS and not self.write_paths:
            raise ValueError("write_paths is required when write tools are allowed")
        if any(not value.strip() for value in (*self.read_paths, *self.write_paths)):
            raise ValueError("task paths must be non-empty")
        if any(not item.strip() for item in self.acceptance_criteria):
            raise ValueError("acceptance criteria must be non-empty")
        if self.max_retries and tools & (_WRITE_TOOLS | _UNSCOPED_TOOLS):
            raise ValueError("automatic retries are limited to read-only scoped tasks")
        if self.execution_mode == WorkspaceMode.WORKTREE:
            if self.allow_unscoped_tools or tools & _UNSCOPED_TOOLS:
                raise ValueError("worktree tasks cannot use unscoped tools")
            if any(Path(value).expanduser().is_absolute() for value in (*self.read_paths, *self.write_paths)):
                raise ValueError("worktree task paths must be repository-relative")
        return self


class TaskTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=500)
    status: str = Field(default="not_run", max_length=40)
    details: str = Field(default="", max_length=2_000)


class AcceptanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(min_length=1, max_length=1_000)
    passed: bool | None = None
    evidence: str = Field(default="", max_length=2_000)
    verified_by_parent: bool = False


class TaskUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0


class TaskEvent(BaseModel):
    """Small, non-sensitive lifecycle record produced by the controller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(default=0, ge=0)
    timestamp: str
    event: TaskEventKind
    task_id: str
    agent_id: str
    parent_id: str
    role: TaskRole
    execution_mode: WorkspaceMode
    status: TaskStatus
    attempt: int = Field(default=1, ge=1, le=4)
    permission_scope: str = Field(default="", max_length=2_000)
    message: str = Field(default="", max_length=2_000)
    tool_name: str = Field(default="", max_length=120)


class TaskEventBatch(BaseModel):
    """Cursor-based bounded event response for polling or streaming clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    events: tuple[TaskEvent, ...] = ()
    next_sequence: int = Field(default=0, ge=0)
    history_truncated: bool = False
    has_more: bool = False
    terminal: bool = False
    timed_out: bool = False


class TaskSnapshot(BaseModel):
    """Queryable control-plane state without exposing task prompt/context text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    agent_id: str
    parent_id: str
    role: TaskRole
    execution_mode: WorkspaceMode
    status: TaskStatus
    submitted_at: str
    started_at: str = ""
    finished_at: str = ""
    attempts: int = Field(default=1, ge=1, le=4)
    usage: TaskUsage = Field(default_factory=TaskUsage)
    accepted: bool = False
    requires_parent_review: bool = True
    error: str | None = Field(default=None, max_length=2_000)


class TaskResult(BaseModel):
    """Bounded child output.  ``completed`` never means centrally accepted."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    agent_id: str
    parent_id: str
    role: TaskRole
    execution_mode: WorkspaceMode
    status: TaskStatus
    summary: str = Field(default="", max_length=5_000)
    evidence: list[str] = Field(default_factory=list, max_length=50)
    modifications: list[str] = Field(default_factory=list, max_length=200)
    tests: list[TaskTestResult] = Field(default_factory=list, max_length=50)
    acceptance: list[AcceptanceCheck] = Field(default_factory=list, max_length=50)
    risks: list[str] = Field(default_factory=list, max_length=50)
    error: str | None = Field(default=None, max_length=2_000)
    policy_violations: int = 0
    usage: TaskUsage = Field(default_factory=TaskUsage)
    attempts: int = Field(default=1, ge=1, le=4)
    workspace_path: str = ""
    merge_status: str = "not_applicable"
    accepted: bool = False
    requires_parent_review: bool = True

    def to_legacy_text(self) -> str:
        """Compact text adapter for callers of the pre-protocol ``spawn`` API."""
        if self.status == TaskStatus.COMPLETED:
            return self.summary
        detail = self.error or self.summary or "task did not complete"
        return f"Sub-agent ({self.role.value}) {self.status.value}: {detail}"


class TaskReport(BaseModel):
    """The small JSON document requested from the child model.

    File modifications, usage, policy violations and terminal status are never
    accepted from this report; the controller derives them from runtime facts.
    """

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(default="", max_length=5_000)
    evidence: list[str] = Field(default_factory=list, max_length=50)
    tests: list[TaskTestResult] = Field(default_factory=list, max_length=50)
    acceptance: list[AcceptanceCheck] = Field(default_factory=list, max_length=50)
    risks: list[str] = Field(default_factory=list, max_length=50)


class TeamMemberTemplate(BaseModel):
    """Reusable role/authority preset within an Agent Team."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=80)
    role: TaskRole
    stage: int = Field(default=0, ge=0, le=20)
    allowed_tools: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    token_budget: int = Field(default=8_000, ge=256, le=1_000_000)
    max_tool_calls: int = Field(default=20, ge=0, le=1_000)
    max_rounds: int = Field(default=8, ge=1, le=50)
    timeout_seconds: float = Field(default=180.0, gt=0, le=3_600)
    execution_mode: WorkspaceMode = WorkspaceMode.FORK


class AgentTeamTemplate(BaseModel):
    """A staged role template; members never communicate directly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=80)
    members: tuple[TeamMemberTemplate, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def unique_member_names(self) -> AgentTeamTemplate:
        names = [member.name for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("team member names must be unique")
        return self


class AgentTeamResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team: str
    results: dict[str, TaskResult]
    completed: bool
    requires_parent_review: bool = True


CODING_TEAM = AgentTeamTemplate(
    name="coding",
    members=(
        TeamMemberTemplate(
            name="researcher",
            role=TaskRole.RESEARCHER,
            stage=0,
            allowed_tools=("read_file", "grep", "glob"),
            read_paths=(".",),
        ),
        TeamMemberTemplate(
            name="executor",
            role=TaskRole.EXECUTOR,
            stage=1,
            allowed_tools=("read_file", "grep", "glob", "write_file", "edit_file", "edit_ast"),
            read_paths=(".",),
            write_paths=(".",),
            token_budget=16_000,
            max_tool_calls=30,
            max_rounds=12,
        ),
        TeamMemberTemplate(
            name="reviewer",
            role=TaskRole.REVIEWER,
            stage=2,
            allowed_tools=("read_file", "grep", "glob"),
            read_paths=(".",),
        ),
    ),
)


class TaskBoundary:
    """Deterministic tool and path checks applied before the normal Guard."""

    def __init__(
        self,
        spec: TaskSpec,
        *,
        base_path: Path | None = None,
        ownership_check: Callable[[], str | None] | None = None,
    ):
        self.spec = spec
        self.base_path = (base_path or Path.cwd()).resolve()
        self.ownership_check = ownership_check
        self.allowed_tools = frozenset(spec.allowed_tools)
        self.read_roots = self._resolve_roots(spec.read_paths)
        self.write_roots = self._resolve_roots(spec.write_paths)

    def _resolve_roots(self, values: tuple[str, ...]) -> tuple[Path, ...]:
        roots: list[Path] = []
        for value in values:
            candidate = Path(value).expanduser()
            resolved = (candidate if candidate.is_absolute() else self.base_path / candidate).resolve()
            if resolved not in roots:
                roots.append(resolved)
        return tuple(roots)

    @staticmethod
    def _inside(target: Path, roots: tuple[Path, ...]) -> bool:
        return any(target == root or target.is_relative_to(root) for root in roots)

    def check(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if self.ownership_check is not None and (reason := self.ownership_check()):
            return reason
        if tool_name not in self.allowed_tools:
            return f"task does not allow tool '{tool_name}'"
        if tool_name in _UNSCOPED_TOOLS and not self.spec.allow_unscoped_tools:
            return f"task does not allow unscoped tool '{tool_name}'"

        roots: tuple[Path, ...]
        raw_path: Any
        if tool_name == "retrieve_context":
            return None
        if tool_name in _PATH_READ_TOOLS:
            roots = self.read_roots
            raw_path = arguments.get("file_path") if tool_name == "read_file" else arguments.get("path", ".")
        elif tool_name in _WRITE_TOOLS:
            roots = self.write_roots
            raw_path = arguments.get("file_path")
        else:
            return None

        if raw_path is None:
            return f"task cannot determine the path targeted by '{tool_name}'"
        candidate = Path(str(raw_path)).expanduser()
        target = (candidate if candidate.is_absolute() else self.base_path / candidate).resolve()
        if self._inside(target, roots):
            return None
        allowed = ", ".join(str(root) for root in roots) or "(none)"
        return f"path '{target}' is outside task scope: {allowed}"

    def resolve_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Resolve relative file-tool paths against this task's workspace."""
        resolved = dict(arguments)
        key = "file_path" if tool_name in {"read_file", *_WRITE_TOOLS} else "path"
        if tool_name not in _PATH_READ_TOOLS | _WRITE_TOOLS:
            return resolved
        if key not in resolved:
            if tool_name in {"grep", "glob"}:
                resolved[key] = str(self.base_path)
            return resolved
        candidate = Path(str(resolved[key])).expanduser()
        if not candidate.is_absolute():
            resolved[key] = str((self.base_path / candidate).resolve())
        return resolved


TaskRunner = Callable[[TaskSpec, str], Awaitable[TaskResult]]
TaskEventSink = Callable[[TaskEvent], None]


class _ExecutionPermit:
    """Acquire local and optional pool capacity as one cancellation-safe permit."""

    def __init__(
        self,
        local: asyncio.Semaphore,
        shared: asyncio.Semaphore | None,
    ):
        self.local = local
        self.shared = shared
        self.local_acquired = False
        self.shared_acquired = False

    async def __aenter__(self) -> None:
        await self.local.acquire()
        self.local_acquired = True
        try:
            if self.shared is not None:
                await self.shared.acquire()
                self.shared_acquired = True
        except BaseException:
            self.local.release()
            self.local_acquired = False
            raise

    async def __aexit__(self, *_exc_info) -> None:
        if self.shared_acquired and self.shared is not None:
            self.shared.release()
            self.shared_acquired = False
        if self.local_acquired:
            self.local.release()
            self.local_acquired = False


class TaskController:
    """Central task state machine with bounded concurrency and cancellation."""

    def __init__(
        self,
        runner: TaskRunner,
        *,
        parent_id: str,
        max_concurrency: int = 1,
        failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 60.0,
        event_sink: TaskEventSink | None = None,
        admission_check: Callable[[], str | None] | None = None,
        history_limit: int = 1_000,
        execution_limiter: asyncio.Semaphore | None = None,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if failure_threshold < 1 or circuit_cooldown_seconds <= 0:
            raise ValueError("circuit breaker settings must be positive")
        if not 1 <= history_limit <= 10_000:
            raise ValueError("history_limit must be between 1 and 10000")
        self._runner = runner
        self.parent_id = parent_id
        self.max_concurrency = max_concurrency
        self.history_limit = history_limit
        self._event_sink = event_sink
        self._admission_check = admission_check
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._execution_limiter = execution_limiter
        self._statuses: dict[str, TaskStatus] = {}
        self._active: dict[str, asyncio.Task[TaskResult]] = {}
        self._background: dict[str, asyncio.Task[TaskResult]] = {}
        self._results: dict[str, TaskResult] = {}
        self._specs: dict[str, TaskSpec] = {}
        self._agent_ids: dict[str, str] = {}
        self._submitted_at: dict[str, str] = {}
        self._started_at: dict[str, str] = {}
        self._finished_at: dict[str, str] = {}
        self._events: deque[TaskEvent] = deque(maxlen=history_limit * 8)
        self._event_sequence = 0
        self._last_pruned_sequence = 0
        self._last_pruned_by_task: dict[str, int] = {}
        self._event_waiters: dict[asyncio.Future[None], str | None] = {}
        self._cancel_requested: set[str] = set()
        self._consecutive_failures = 0
        self._failure_threshold = failure_threshold
        self._circuit_cooldown_seconds = circuit_cooldown_seconds
        self._circuit_open_until = 0.0

    def set_execution_limiter(self, limiter: asyncio.Semaphore | None) -> None:
        """Bind a shared pool limit before this controller starts new work."""
        if self._active or self._background:
            raise RuntimeError("cannot change execution limiter while tasks are active")
        self._execution_limiter = limiter

    def status(self, task_id: str) -> TaskStatus | None:
        return self._statuses.get(task_id)

    def result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    def snapshot(self, task_id: str) -> TaskSnapshot | None:
        """Return trusted current state without task objective or context."""
        spec = self._specs.get(task_id)
        status = self._statuses.get(task_id)
        agent_id = self._agent_ids.get(task_id)
        if spec is None or status is None or agent_id is None:
            return None
        result = self._results.get(task_id)
        return TaskSnapshot(
            task_id=task_id,
            agent_id=agent_id,
            parent_id=result.parent_id if result else self.parent_id,
            role=spec.role,
            execution_mode=spec.execution_mode,
            status=status,
            submitted_at=self._submitted_at[task_id],
            started_at=self._started_at.get(task_id, ""),
            finished_at=self._finished_at.get(task_id, ""),
            attempts=result.attempts if result else 1,
            usage=result.usage if result else TaskUsage(),
            accepted=result.accepted if result else False,
            requires_parent_review=result.requires_parent_review if result else True,
            error=result.error if result else None,
        )

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 100,
    ) -> tuple[TaskSnapshot, ...]:
        """Return the most recently submitted matching task snapshots."""
        if not 1 <= limit <= 1_000:
            raise ValueError("task query limit must be between 1 and 1000")
        snapshots = (
            snapshot
            for task_id in reversed(self._statuses)
            if (snapshot := self.snapshot(task_id)) is not None
        )
        return tuple(
            snapshot
            for snapshot in snapshots
            if status is None or snapshot.status == status
        )[:limit]

    def events(
        self,
        *,
        task_id: str | None = None,
        limit: int = 100,
    ) -> tuple[TaskEvent, ...]:
        """Return the most recent in-memory lifecycle events, oldest first."""
        if not 1 <= limit <= 1_000:
            raise ValueError("task event limit must be between 1 and 1000")
        matching = [event for event in self._events if task_id is None or event.task_id == task_id]
        return tuple(matching[-limit:])

    def event_batch(
        self,
        *,
        task_id: str | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> TaskEventBatch:
        """Return the oldest unseen bounded events plus a resumable cursor."""
        if after_sequence < 0:
            raise ValueError("event cursor must not be negative")
        if not 1 <= limit <= 1_000:
            raise ValueError("task event limit must be between 1 and 1000")
        matching = [
            event
            for event in self._events
            if event.sequence > after_sequence
            and (task_id is None or event.task_id == task_id)
        ]
        selected = tuple(matching[:limit])
        next_sequence = selected[-1].sequence if selected else max(
            after_sequence,
            self._event_sequence,
        )
        pruned_sequence = (
            self._last_pruned_sequence
            if task_id is None
            else self._last_pruned_by_task.get(task_id, 0)
        )
        return TaskEventBatch(
            events=selected,
            next_sequence=next_sequence,
            history_truncated=pruned_sequence > after_sequence,
            has_more=len(matching) > len(selected),
            terminal=task_id is not None and task_id in self._results,
        )

    async def wait_events(
        self,
        *,
        task_id: str | None = None,
        after_sequence: int = 0,
        timeout: float | None = None,
        limit: int = 100,
    ) -> TaskEventBatch:
        """Long-poll event history without taking cancellation ownership."""
        if timeout is not None and timeout <= 0:
            raise ValueError("event wait timeout must be positive")
        if task_id is not None and task_id not in self._statuses:
            raise KeyError(task_id)
        batch = self.event_batch(
            task_id=task_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        if batch.events or batch.has_more or batch.terminal:
            return batch

        waiter = asyncio.get_running_loop().create_future()
        self._event_waiters[waiter] = task_id
        try:
            # Recheck after registering so an event cannot fall into the gap
            # between the initial read and waiter creation.
            batch = self.event_batch(
                task_id=task_id,
                after_sequence=after_sequence,
                limit=limit,
            )
            if batch.events or batch.has_more or batch.terminal:
                return batch
            try:
                await asyncio.wait_for(asyncio.shield(waiter), timeout=timeout)
            except TimeoutError:
                return self.event_batch(
                    task_id=task_id,
                    after_sequence=after_sequence,
                    limit=limit,
                ).model_copy(update={"timed_out": True})
            return self.event_batch(
                task_id=task_id,
                after_sequence=after_sequence,
                limit=limit,
            )
        finally:
            self._event_waiters.pop(waiter, None)
            if not waiter.done():
                waiter.cancel()

    def report_progress(
        self,
        task_id: str,
        event: TaskEventKind,
        *,
        agent_id: str = "",
        tool_name: str = "",
    ) -> bool:
        """Record one controller-approved, non-sensitive runtime milestone."""
        if event not in _PROGRESS_EVENTS:
            raise ValueError(f"not a progress event: {event.value}")
        spec = self._specs.get(task_id)
        if spec is None:
            raise KeyError(task_id)
        if self._statuses.get(task_id) != TaskStatus.RUNNING:
            return False
        self._emit(
            spec,
            agent_id or self._agent_ids[task_id],
            event,
            TaskStatus.RUNNING,
            tool_name=tool_name[:120],
        )
        return True

    def restore(
        self,
        records: Iterable[tuple[TaskEvent, TaskResult | None]],
        *,
        interrupt_unfinished: bool = True,
    ) -> tuple[str, ...]:
        """Restore durable terminal state and mark unfinished process work interrupted."""
        grouped: dict[str, list[tuple[TaskEvent, TaskResult | None]]] = {}
        for event, result in records:
            sequence = event.sequence
            if sequence <= self._event_sequence:
                sequence = self._event_sequence + 1
                event = event.model_copy(update={"sequence": sequence})
            self._event_sequence = sequence
            grouped.setdefault(event.task_id, []).append((event, result))
            self._append_event(event)

        restored: list[str] = []
        for task_id, task_records in grouped.items():
            if task_id in self._statuses:
                continue
            events = [item[0] for item in task_records]
            latest = events[-1]
            result = next(
                (candidate for _, candidate in reversed(task_records) if candidate is not None),
                None,
            )
            valid_result = (
                result is not None
                and result.task_id == task_id
                and result.role == latest.role
                and result.execution_mode == latest.execution_mode
                and result.status == latest.status
            )
            spec = TaskSpec(
                task_id=task_id,
                objective="Recovered task metadata; original objective was not persisted",
                role=latest.role,
                execution_mode=latest.execution_mode,
            )
            self._specs[task_id] = spec
            self._agent_ids[task_id] = result.agent_id if valid_result else latest.agent_id
            self._submitted_at[task_id] = next(
                (event.timestamp for event in events if event.event == TaskEventKind.SUBMITTED),
                events[0].timestamp,
            )
            started_at = next(
                (event.timestamp for event in events if event.event == TaskEventKind.STARTED),
                "",
            )
            if started_at:
                self._started_at[task_id] = started_at

            if valid_result and latest.status in _TERMINAL_STATUSES:
                self._statuses[task_id] = result.status
                self._results[task_id] = result
                self._finished_at[task_id] = latest.timestamp
                restored.append(task_id)
                continue

            if not interrupt_unfinished:
                self._statuses[task_id] = latest.status
                restored.append(task_id)
                continue

            interrupted = TaskResult(
                task_id=task_id,
                agent_id=latest.agent_id,
                parent_id=latest.parent_id,
                role=latest.role,
                execution_mode=latest.execution_mode,
                status=TaskStatus.INTERRUPTED,
                error="process ended before a durable terminal task result was recorded",
                risks=["task was not automatically resumed or retried"],
            )
            self._store_terminal(spec, interrupted)
            restored.append(task_id)

        self._prune_history()
        return tuple(restored)

    def accept(
        self,
        task_id: str,
        checks: list[AcceptanceCheck] | None = None,
    ) -> TaskResult:
        """Record the control plane's explicit acceptance of a child result."""
        result = self._results.get(task_id)
        if result is None:
            raise KeyError(task_id)
        if result.status != TaskStatus.COMPLETED:
            raise ValueError("only completed tasks can be accepted")
        verified = checks if checks is not None else result.acceptance
        expected = [item.criterion for item in result.acceptance]
        supplied = [item.criterion for item in verified]
        if supplied != expected:
            raise ValueError("parent checks must match the task acceptance criteria")
        if any(item.passed is not True or not item.verified_by_parent for item in verified):
            raise ValueError("every acceptance check must be passed and parent-verified")
        accepted = result.model_copy(update={
            "acceptance": verified,
            "accepted": True,
            "requires_parent_review": False,
        })
        self._results[task_id] = accepted
        spec = self._specs[task_id]
        self._emit(
            spec,
            accepted.agent_id,
            TaskEventKind.ACCEPTED,
            accepted.status,
            attempt=accepted.attempts,
            message="parent verified all acceptance criteria",
        )
        return accepted

    def cancel(self, task_id: str) -> bool:
        task = self._active.get(task_id) or self._background.get(task_id)
        if task is None or task.done():
            return False
        self._cancel_requested.add(task_id)
        spec = self._specs[task_id]
        self._emit(
            spec,
            self._agent_ids[task_id],
            TaskEventKind.CANCEL_REQUESTED,
            self._statuses[task_id],
            message="cancellation requested by the parent control plane",
        )
        task.cancel()
        return True

    def cancel_all(self) -> tuple[str, ...]:
        """Request cancellation for every currently queued or running task."""
        task_ids = tuple(dict.fromkeys((*self._active, *self._background)))
        cancelled = [task_id for task_id in task_ids if self.cancel(task_id)]
        return tuple(cancelled)

    def prepare_durable_resume(self, task_id: str) -> bool:
        """Forget only a safely resumable crash/shutdown terminal state."""
        if self._statuses.get(task_id) not in {TaskStatus.INTERRUPTED, TaskStatus.CANCELLED}:
            return False
        for mapping in (
            self._statuses,
            self._results,
            self._specs,
            self._agent_ids,
            self._submitted_at,
            self._started_at,
            self._finished_at,
        ):
            mapping.pop(task_id, None)
        return True

    async def submit(self, spec: TaskSpec) -> str:
        """Schedule a task in the background and return its id after registration."""
        self._check_admission()
        if spec.task_id in self._statuses or spec.task_id in self._background:
            raise ValueError(f"duplicate task_id: {spec.task_id}")

        task = asyncio.create_task(self.execute(spec), name=f"submit:{spec.task_id}")
        self._background[spec.task_id] = task

        def discard(done: asyncio.Task[TaskResult]) -> None:
            if self._background.get(spec.task_id) is done:
                self._background.pop(spec.task_id, None)
            try:
                done.result()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    "Background task %s failed outside its isolation boundary",
                    spec.task_id,
                )

        task.add_done_callback(discard)
        # ``execute`` registers PENDING synchronously before its first await.
        # One loop turn therefore makes snapshot/status immediately queryable.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            if spec.task_id in self._statuses:
                self.cancel(spec.task_id)
            else:
                task.cancel()
            raise
        if task.done() and not task.cancelled():
            error = task.exception()
            if error is not None:
                raise error
        return spec.task_id

    async def submit_many(self, specs: list[TaskSpec]) -> tuple[str, ...]:
        """Schedule independent tasks without waiting for their results."""
        task_ids = [spec.task_id for spec in specs]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("batch contains duplicate task_id values")
        submitted: list[str] = []
        try:
            for spec in specs:
                submitted.append(await self.submit(spec))
        except BaseException:
            for task_id in submitted:
                self.cancel(task_id)
            raise
        return tuple(submitted)

    async def wait(self, task_id: str, *, timeout: float | None = None) -> TaskResult:
        """Wait for a background result without transferring cancellation ownership."""
        if timeout is not None and timeout <= 0:
            raise ValueError("wait timeout must be positive")
        result = self._results.get(task_id)
        if result is not None:
            return result
        task = self._background.get(task_id)
        if task is None:
            # Close the small race between background cleanup and result storage.
            result = self._results.get(task_id)
            if result is not None:
                return result
            if task_id not in self._statuses:
                raise KeyError(task_id)
            raise RuntimeError("task is not owned by the background scheduler")
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def execute(self, spec: TaskSpec) -> TaskResult:
        self._check_admission()
        if spec.task_id in self._statuses:
            raise ValueError(f"duplicate task_id: {spec.task_id}")
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        submitted_at = self._timestamp()
        self._statuses[spec.task_id] = TaskStatus.PENDING
        self._specs[spec.task_id] = spec
        self._agent_ids[spec.task_id] = agent_id
        self._submitted_at[spec.task_id] = submitted_at
        self._emit(
            spec,
            agent_id,
            TaskEventKind.SUBMITTED,
            TaskStatus.PENDING,
            timestamp=submitted_at,
        )
        started = time.monotonic()

        if started < self._circuit_open_until:
            result = self._terminal_result(
                spec,
                agent_id,
                TaskStatus.REJECTED,
                started,
                "task controller circuit breaker is open",
            )
            self._store_terminal(spec, result)
            return result

        async def controlled() -> TaskResult:
            async with _ExecutionPermit(self._semaphore, self._execution_limiter):
                if time.monotonic() < self._circuit_open_until:
                    return self._terminal_result(
                        spec,
                        agent_id,
                        TaskStatus.REJECTED,
                        started,
                        "task controller circuit breaker is open",
                    )
                self._statuses[spec.task_id] = TaskStatus.RUNNING
                started_at = self._timestamp()
                self._started_at[spec.task_id] = started_at
                self._emit(
                    spec,
                    agent_id,
                    TaskEventKind.STARTED,
                    TaskStatus.RUNNING,
                    timestamp=started_at,
                )
                prompt_tokens = completion_tokens = tool_calls = 0
                for attempt in range(1, spec.max_retries + 2):
                    remaining_tokens = spec.token_budget - prompt_tokens - completion_tokens
                    remaining_calls = spec.max_tool_calls - tool_calls
                    if remaining_tokens < 256 or remaining_calls < 0:
                        return self._terminal_result(
                            spec,
                            agent_id,
                            TaskStatus.BUDGET_EXCEEDED,
                            started,
                            "task budget exhausted before retry",
                            attempts=attempt,
                        )
                    attempt_spec = spec.model_copy(update={
                        "token_budget": remaining_tokens,
                        "max_tool_calls": remaining_calls,
                    })
                    attempt_agent_id = agent_id if attempt == 1 else f"{agent_id}_r{attempt - 1}"
                    try:
                        result = await self._runner(attempt_spec, attempt_agent_id)
                    except Exception as exc:  # noqa: BLE001 - child boundary
                        result = self._terminal_result(
                            spec,
                            attempt_agent_id,
                            TaskStatus.FAILED,
                            started,
                            f"{type(exc).__name__}: {exc}",
                            attempts=attempt,
                        )
                    prompt_tokens += result.usage.prompt_tokens
                    completion_tokens += result.usage.completion_tokens
                    tool_calls += result.usage.tool_calls
                    result = result.model_copy(update={
                        "attempts": attempt,
                        "usage": TaskUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            tool_calls=tool_calls,
                            duration_ms=(time.monotonic() - started) * 1000,
                        ),
                    })
                    retryable = (
                        result.status == TaskStatus.FAILED
                        and not result.policy_violations
                        and not result.modifications
                    )
                    if not retryable or attempt > spec.max_retries:
                        return result
                    self._emit(
                        spec,
                        attempt_agent_id,
                        TaskEventKind.RETRYING,
                        TaskStatus.RUNNING,
                        attempt=attempt + 1,
                        message="read-only attempt failed without modifications",
                    )
                raise RuntimeError("unreachable retry state")

        task = asyncio.create_task(controlled(), name=spec.task_id)
        self._active[spec.task_id] = task
        try:
            result = await asyncio.wait_for(task, timeout=spec.timeout_seconds)
        except asyncio.TimeoutError:
            result = self._terminal_result(
                spec, agent_id, TaskStatus.TIMED_OUT, started,
                f"task exceeded {spec.timeout_seconds:g}s timeout",
            )
        except asyncio.CancelledError:
            # Explicit controller cancellation is a task result. Cancellation
            # of the caller itself (for example Ctrl+C at the parent) must keep
            # propagating so the parent agent can perform interrupt cleanup.
            if spec.task_id not in self._cancel_requested:
                result = self._terminal_result(
                    spec,
                    agent_id,
                    TaskStatus.CANCELLED,
                    started,
                    "caller cancelled task execution",
                )
                self._store_terminal(spec, result)
                raise
            result = self._terminal_result(
                spec, agent_id, TaskStatus.CANCELLED, started, "task was cancelled",
            )
        except Exception as exc:  # noqa: BLE001 - runner/LLM isolation boundary
            result = self._terminal_result(
                spec,
                agent_id,
                TaskStatus.FAILED,
                started,
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._active.pop(spec.task_id, None)
            self._cancel_requested.discard(spec.task_id)
            self._prune_history()

        # Re-validate even results created by the trusted runner.  This makes
        # custom controller runners obey the same result protocol.
        result = TaskResult.model_validate(result)
        if result.task_id != spec.task_id or result.parent_id != self.parent_id:
            result = self._terminal_result(
                spec, agent_id, TaskStatus.REJECTED, started,
                "runner returned mismatched task or parent identity",
            )
        if result.status in {TaskStatus.FAILED, TaskStatus.TIMED_OUT}:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._circuit_open_until = time.monotonic() + self._circuit_cooldown_seconds
                self._emit(
                    spec,
                    result.agent_id,
                    TaskEventKind.CIRCUIT_OPENED,
                    result.status,
                    attempt=result.attempts,
                    message="consecutive failure threshold reached",
                )
        elif result.status == TaskStatus.COMPLETED:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
        self._store_terminal(spec, result)
        return result

    async def execute_many(self, specs: list[TaskSpec]) -> list[TaskResult]:
        """Run independent tasks under the controller's concurrency limit."""
        task_ids = [spec.task_id for spec in specs]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("batch contains duplicate task_id values")
        return list(await asyncio.gather(*(self.execute(spec) for spec in specs)))

    def _check_admission(self) -> None:
        if self._admission_check is None:
            return
        reason = self._admission_check()
        if reason:
            raise RuntimeError(reason)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def _emit(
        self,
        spec: TaskSpec,
        agent_id: str,
        event: TaskEventKind,
        status: TaskStatus,
        *,
        attempt: int = 1,
        message: str = "",
        tool_name: str = "",
        timestamp: str | None = None,
    ) -> None:
        self._event_sequence += 1
        record = TaskEvent(
            sequence=self._event_sequence,
            timestamp=timestamp or self._timestamp(),
            event=event,
            task_id=spec.task_id,
            agent_id=agent_id,
            parent_id=self.parent_id,
            role=spec.role,
            execution_mode=spec.execution_mode,
            status=status,
            attempt=attempt,
            permission_scope=",".join(spec.allowed_tools),
            message=message,
            tool_name=tool_name,
        )
        self._append_event(record)
        for waiter, watched_task_id in tuple(self._event_waiters.items()):
            if watched_task_id in (None, spec.task_id) and not waiter.done():
                waiter.set_result(None)
        if self._event_sink is not None:
            try:
                self._event_sink(record)
            except Exception:
                logger.warning("Task lifecycle event sink failed", exc_info=True)

    def _append_event(self, event: TaskEvent) -> None:
        if len(self._events) == self._events.maxlen:
            pruned = self._events[0]
            self._last_pruned_sequence = max(self._last_pruned_sequence, pruned.sequence)
            self._last_pruned_by_task[pruned.task_id] = max(
                self._last_pruned_by_task.get(pruned.task_id, 0),
                pruned.sequence,
            )
        self._events.append(event)

    def _store_terminal(self, spec: TaskSpec, result: TaskResult) -> None:
        self._statuses[spec.task_id] = result.status
        self._results[spec.task_id] = result
        self._agent_ids[spec.task_id] = result.agent_id
        finished_at = self._timestamp()
        self._finished_at[spec.task_id] = finished_at
        self._emit(
            spec,
            result.agent_id,
            TaskEventKind(result.status.value),
            result.status,
            attempt=result.attempts,
            timestamp=finished_at,
        )
        self._prune_history()

    def _prune_history(self) -> None:
        terminal_ids = [
            task_id for task_id in self._statuses
            if task_id not in self._active and task_id in self._results
        ]
        for task_id in terminal_ids[:-self.history_limit]:
            self._statuses.pop(task_id, None)
            self._results.pop(task_id, None)
            self._specs.pop(task_id, None)
            self._agent_ids.pop(task_id, None)
            self._submitted_at.pop(task_id, None)
            self._started_at.pop(task_id, None)
            self._finished_at.pop(task_id, None)

    def _terminal_result(
        self,
        spec: TaskSpec,
        agent_id: str,
        status: TaskStatus,
        started: float,
        error: str,
        attempts: int = 1,
    ) -> TaskResult:
        return TaskResult(
            task_id=spec.task_id,
            agent_id=agent_id,
            parent_id=self.parent_id,
            role=spec.role,
            execution_mode=spec.execution_mode,
            status=status,
            error=error,
            risks=[error],
            usage=TaskUsage(duration_ms=(time.monotonic() - started) * 1000),
            attempts=attempts,
        )

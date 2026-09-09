"""CoreCoder - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.4.0"

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.context_artifacts import ContextArtifactStore
from corecoder.delegation import (
    CODING_TEAM,
    AcceptanceCheck,
    AgentTeamResult,
    AgentTeamTemplate,
    TaskController,
    TaskEvent,
    TaskEventBatch,
    TaskEventKind,
    TaskResult,
    TaskRole,
    TaskSnapshot,
    TaskSpec,
    TaskStatus,
    TaskTestResult,
    TaskUsage,
    TeamMemberTemplate,
    WorkspaceMode,
)
from corecoder.llm import LLM
from corecoder.memory import Memory, MemoryEngine
from corecoder.models import LLMResponse, PlanRecord, PlanStep, StepRecord, ToolCall, ToolExecRecord
from corecoder.skills import SkillManager, SkillManifest, SkillRegistry, SkillRouter
from corecoder.task_journal import (
    TaskJournal,
    TaskJournalLoad,
    TaskJournalRecord,
    TaskLeaseRecord,
    TaskWorkspaceLease,
)
from corecoder.task_queue import DurableQueueLoad, DurableTaskEnvelope, DurableTaskQueue
from corecoder.tools import ALL_TOOLS
from corecoder.tools.changes import ChangeTracker, UndoResult
from corecoder.worker import DurableTaskWorker, DurableTaskWorkerPool, WorkerPoolStats, WorkerStats
from corecoder.workspaces import WorktreeError, WorktreeSession

__all__ = [
    "ALL_TOOLS",
    "CODING_TEAM",
    "LLM",
    "AcceptanceCheck",
    "Agent",
    "AgentTeamResult",
    "AgentTeamTemplate",
    "ChangeTracker",
    "Config",
    "ContextArtifactStore",
    "DurableQueueLoad",
    "DurableTaskEnvelope",
    "DurableTaskQueue",
    "DurableTaskWorker",
    "DurableTaskWorkerPool",
    "LLMResponse",
    "Memory",
    "MemoryEngine",
    "PlanRecord",
    "PlanStep",
    "SkillManager",
    "SkillManifest",
    "SkillRegistry",
    "SkillRouter",
    "StepRecord",
    "TaskController",
    "TaskEvent",
    "TaskEventBatch",
    "TaskEventKind",
    "TaskJournal",
    "TaskJournalLoad",
    "TaskJournalRecord",
    "TaskLeaseRecord",
    "TaskResult",
    "TaskRole",
    "TaskSnapshot",
    "TaskSpec",
    "TaskStatus",
    "TaskTestResult",
    "TaskUsage",
    "TaskWorkspaceLease",
    "TeamMemberTemplate",
    "ToolCall",
    "ToolExecRecord",
    "UndoResult",
    "WorkerPoolStats",
    "WorkerStats",
    "WorkspaceMode",
    "WorktreeError",
    "WorktreeSession",
    "__version__",
]

"""CoreCoder - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.4.0"

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.llm import LLM
from corecoder.memory import Memory, MemoryEngine
from corecoder.models import LLMResponse, PlanRecord, PlanStep, StepRecord, ToolCall, ToolExecRecord
from corecoder.skills import SkillManager, SkillManifest, SkillRegistry, SkillRouter
from corecoder.tools import ALL_TOOLS
from corecoder.tools.changes import ChangeTracker, UndoResult

__all__ = [
    "ALL_TOOLS",
    "LLM",
    "Agent",
    "ChangeTracker",
    "Config",
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
    "ToolCall",
    "ToolExecRecord",
    "UndoResult",
    "__version__",
]

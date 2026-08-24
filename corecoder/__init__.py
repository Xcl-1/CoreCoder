"""CoreCoder - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.4.0"

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.llm import LLM
from corecoder.memory import Memory, MemoryEngine
from corecoder.models import LLMResponse, PlanRecord, PlanStep, StepRecord, ToolCall, ToolExecRecord
from corecoder.tools import ALL_TOOLS

__all__ = [
    "ALL_TOOLS",
    "LLM",
    "Agent",
    "Config",
    "LLMResponse",
    "Memory",
    "MemoryEngine",
    "PlanRecord",
    "PlanStep",
    "StepRecord",
    "ToolCall",
    "ToolExecRecord",
    "__version__",
]

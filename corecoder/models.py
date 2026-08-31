"""Pydantic data models for CoreCoder.

All structured data that flows through the agent loop lives here:
Tool calls, LLM responses, and replay log records.
"""

import json
import time

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool-call decision made by the LLM."""

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """A complete LLM response — text, optional tool calls, and usage stats."""

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def message(self) -> dict:
        """Convert to OpenAI chat message format for appending to history.

        This is a *view* of the data, not data itself, so it's a property —
        Pydantic skips properties during serialization by default.
        """
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.reasoning_content:
            # DeepSeek thinking-mode tool calls require the reasoning from the
            # preceding assistant message to be sent back on the next request.
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


class ToolExecRecord(BaseModel):
    """Record of a single tool execution within a step."""

    name: str
    arguments: dict = Field(default_factory=dict)
    result: str = ""  # truncated to max ~5000 chars by the logger
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None


class StepRecord(BaseModel):
    """One complete think→act→observe cycle (one line in the replay JSONL)."""

    step: int
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    messages_count: int = 0
    estimated_input_tokens: int = 0
    llm_response: LLMResponse = Field(default_factory=LLMResponse)
    tool_executions: list[ToolExecRecord] = Field(default_factory=list)
    skill_route: dict = Field(default_factory=dict)
    step_duration_ms: float = 0.0


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    id: int
    action: str  # human-readable description
    tool: str = ""  # suggested tool, or empty for manual reasoning
    expected: str = ""  # what success looks like
    status: str = "pending"  # pending | in_progress | done | failed


class PlanRecord(BaseModel):
    """A structured execution plan produced by the LLM."""

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)

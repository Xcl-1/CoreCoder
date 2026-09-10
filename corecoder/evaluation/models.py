"""Schemas for end-to-end Agent capability evaluations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CheckKind = Literal[
    "answer_contains",
    "answer_equals",
    "answer_regex",
    "file_exists",
    "file_not_exists",
    "file_contains",
    "file_not_contains",
    "file_equals",
    "command_exit",
]


class EvaluationCheck(BaseModel):
    """One deterministic assertion evaluated outside the Agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=200)
    kind: CheckKind
    value: str = ""
    path: str = ""
    command: str = ""
    expected_exit_code: int = 0
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    case_sensitive: bool = True
    required: bool = True
    weight: float = Field(default=1.0, gt=0, le=100)

    @model_validator(mode="after")
    def validate_arguments(self) -> EvaluationCheck:
        if self.kind.startswith("answer_") and not self.value:
            raise ValueError(f"{self.kind} requires value")
        if self.kind.startswith("file_") and not self.path:
            raise ValueError(f"{self.kind} requires path")
        if self.kind in {"file_contains", "file_not_contains", "file_equals"} and not self.value:
            raise ValueError(f"{self.kind} requires value")
        if self.kind == "command_exit" and not self.command:
            raise ValueError("command_exit requires command")
        return self


class EvaluationTask(BaseModel):
    """A reproducible task, its workspace seed, and external acceptance checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    category: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=16_000)
    follow_up_prompts: tuple[str, ...] = Field(default=(), max_length=20)
    description: str = Field(default="", max_length=2_000)
    files: dict[str, str] = Field(default_factory=dict)
    fixture: str = ""
    checks: tuple[EvaluationCheck, ...] = Field(min_length=1)
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=180.0, gt=0, le=3_600)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_duration_seconds: float | None = Field(default=None, gt=0)
    max_policy_violations: int = Field(default=0, ge=0)
    weight: float = Field(default=1.0, gt=0, le=100)

    @model_validator(mode="after")
    def validate_tools(self) -> EvaluationTask:
        if any(not prompt.strip() for prompt in self.follow_up_prompts):
            raise ValueError("follow_up_prompts must be non-empty")
        overlap = set(self.required_tools) & set(self.forbidden_tools)
        if overlap:
            raise ValueError(
                "tools cannot be both required and forbidden: "
                + ", ".join(sorted(overlap))
            )
        return self


class EvaluationSuite(BaseModel):
    """Versioned collection of Agent tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=4_000)
    pass_threshold: float = Field(default=0.8, ge=0, le=1)
    tasks: tuple[EvaluationTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_task_ids(self) -> EvaluationSuite:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation task ids must be unique")
        return self


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: str
    passed: bool
    required: bool = True
    weight: float = 1.0
    evidence: str = Field(default="", max_length=4_000)


class EvaluationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    category: str
    attempt: int = Field(ge=1)
    passed: bool
    status: Literal["completed", "partial", "failed", "timeout", "error"]
    score: float = Field(ge=0, le=1)
    correctness_score: float = Field(ge=0, le=1)
    completion_score: float = Field(ge=0, le=1)
    safety_score: float = Field(ge=0, le=1)
    efficiency_score: float = Field(ge=0, le=1)
    duration_ms: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    tool_names: tuple[str, ...] = ()
    policy_violations: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    answer_excerpt: str = Field(default="", max_length=2_000)
    answer_sha256: str = ""
    error: str = Field(default="", max_length=2_000)
    checks: tuple[CheckResult, ...] = ()
    workspace: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class DimensionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempts: int = Field(ge=0)
    passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    average_score: float = Field(ge=0, le=1)


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempts: int = Field(ge=0)
    passed_attempts: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    correctness_score: float = Field(ge=0, le=1)
    completion_score: float = Field(ge=0, le=1)
    safety_score: float = Field(ge=0, le=1)
    efficiency_score: float = Field(ge=0, le=1)
    reliability_rate: float = Field(ge=0, le=1)
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    total_policy_violations: int = Field(ge=0)
    total_estimated_cost_usd: float | None = Field(default=None, ge=0)
    average_duration_ms: float = Field(ge=0)
    p95_duration_ms: float = Field(ge=0)
    dimensions: dict[str, DimensionSummary] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    suite_name: str
    suite_version: str
    model: str
    started_at: str
    finished_at: str
    repeat: int = Field(ge=1)
    pass_threshold: float = Field(ge=0, le=1)
    passed: bool
    summary: EvaluationSummary
    attempts: tuple[EvaluationAttempt, ...]

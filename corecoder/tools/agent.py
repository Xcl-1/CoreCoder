"""Centralised sub-agent task submission.

The idea: for complex sub-tasks, spawn an independent agent with its own
conversation history and tool access. This lets the main agent delegate
work like "go research this codebase and report back" without polluting
its own context window.

v1.0: supports role-based spawning (executor / reviewer / researcher)
and optional reviewer pass after executor completion.
"""

import json

from ..delegation import (
    CODING_TEAM,
    AgentTeamTemplate,
    TaskRole,
    TaskSpec,
    TaskStatus,
    WorkspaceMode,
)
from .base import Tool


class AgentTool(Tool):
    name = "agent"
    input_types = ("task", "role")
    output_type = "agent_result"
    permission_scope = "agent:delegate"
    side_effect = "delegated"
    network_access = "delegated"
    declared_risk = "medium"
    description = (
        "Spawn a sub-agent to handle a complex sub-task independently. "
        "The parent task controller gives the child an independent context, "
        "bounded tools, paths, tokens, calls and runtime. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window. "
        "Set background=true to receive a task id immediately, then use "
        "task_control to inspect, wait for, or cancel it. "
        "Set the 'role' to 'researcher' for read-only exploration, "
        "'executor' for making changes, or 'reviewer' to check changes. "
        "Set 'review' to true to have a reviewer check executor output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
            "role": {
                "type": "string",
                "enum": ["executor", "researcher", "reviewer"],
                "description": "Sub-agent role. executor=make changes, researcher=explore only, reviewer=check code. Default: executor.",
            },
            "mode": {
                "type": "string",
                "enum": ["single", "coding_team"],
                "description": "Run one scoped child or the staged researcher/executor/reviewer team.",
            },
            "team_objectives": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Optional objectives keyed by researcher, executor, and reviewer.",
            },
            "review": {
                "type": "boolean",
                "description": "If true and role is executor, a reviewer sub-agent will check the output. Default: false.",
            },
            "background": {
                "type": "boolean",
                "description": "Schedule one task and return its id without waiting. Not supported for coding_team or review.",
            },
            "durable": {
                "type": "boolean",
                "description": "Encrypt and persist this background TaskSpec for crash recovery.",
            },
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact child tool allowlist. Unscoped agent/bash/undo tools are rejected.",
            },
            "read_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Directories or files the child may read.",
            },
            "write_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Directories or files the child may modify.",
            },
            "context": {
                "type": "string",
                "description": "Minimal task-specific context; do not copy the full conversation.",
            },
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete conditions the parent will independently verify.",
            },
            "token_budget": {
                "type": "integer",
                "minimum": 256,
                "maximum": 1000000,
                "description": "Maximum reported tokens before the child is stopped.",
            },
            "max_tool_calls": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Maximum child tool calls.",
            },
            "timeout_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 3600,
                "description": "Wall-clock timeout for the task.",
            },
            "execution_mode": {
                "type": "string",
                "enum": ["fork", "worktree"],
                "description": "Shared fork workspace or isolated Git worktree.",
            },
            "max_retries": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": "Automatic retries; allowed only for read-only scoped tasks.",
            },
        },
        "required": ["task"],
    }

    # set by Agent.__init__ after construction
    _parent_agent = None

    async def execute(
        self,
        task: str,
        role: str = "executor",
        mode: str = "single",
        team_objectives: dict[str, str] | None = None,
        review: bool = False,
        background: bool = False,
        durable: bool = False,
        allowed_tools: list[str] | None = None,
        read_paths: list[str] | None = None,
        write_paths: list[str] | None = None,
        context: str = "",
        acceptance_criteria: list[str] | None = None,
        token_budget: int = 16_000,
        max_tool_calls: int = 30,
        timeout_seconds: float = 300,
        execution_mode: str = "fork",
        max_retries: int = 0,
    ) -> str:
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        # import here to avoid circular dep
        role_map = {
            "executor": TaskRole.EXECUTOR,
            "researcher": TaskRole.RESEARCHER,
            "reviewer": TaskRole.REVIEWER,
            "planner": TaskRole.PLANNER,
        }
        if role not in role_map:
            return f"Sub-agent error: unsupported role '{role}'"
        if mode not in {"single", "coding_team"}:
            return f"Sub-agent error: unsupported mode '{mode}'"
        if durable and (not background or mode != "single"):
            return "Sub-agent error: durable mode requires background=true and mode=single"
        try:
            workspace_mode = WorkspaceMode(execution_mode)
        except ValueError:
            return f"Sub-agent error: unsupported execution mode '{execution_mode}'"
        task_role = role_map[role]
        if allowed_tools is None:
            allowed_tools = {
                TaskRole.PLANNER: [],
                TaskRole.RESEARCHER: ["read_file", "grep", "glob"],
                TaskRole.REVIEWER: ["read_file", "grep", "glob"],
                TaskRole.EXECUTOR: [
                    "read_file", "grep", "glob", "write_file", "edit_file", "edit_ast",
                ],
            }[task_role]
        read_tools = {"read_file", "grep", "glob", "retrieve_context"}
        write_tools = {"write_file", "edit_file", "edit_ast"}
        if read_paths is None:
            read_paths = ["."] if set(allowed_tools) & read_tools else []
        if write_paths is None:
            write_paths = ["."] if set(allowed_tools) & write_tools else []

        try:
            if mode == "coding_team":
                if background:
                    return "Sub-agent error: background mode currently supports only single tasks"
                members = []
                for member in CODING_TEAM.members:
                    updates = {
                        "read_paths": tuple(read_paths or member.read_paths),
                        "write_paths": (
                            tuple(write_paths or member.write_paths)
                            if member.role == TaskRole.EXECUTOR else ()
                        ),
                        "execution_mode": (
                            workspace_mode
                            if member.role == TaskRole.EXECUTOR else WorkspaceMode.FORK
                        ),
                    }
                    members.append(member.model_copy(update=updates))
                template = AgentTeamTemplate(name=CODING_TEAM.name, members=tuple(members))
                objectives = team_objectives or {
                    "researcher": f"Research the code relevant to: {task}",
                    "executor": f"Implement the requested change: {task}",
                    "reviewer": f"Review the implementation for: {task}",
                }
                team = await self._parent_agent.run_team(
                    objectives,
                    template=template,
                    context=context,
                    acceptance_criteria=tuple(acceptance_criteria or ()),
                )
                return team.model_dump_json()
            spec = TaskSpec(
                objective=task,
                role=task_role,
                execution_mode=workspace_mode,
                context=context,
                allowed_tools=tuple(allowed_tools),
                read_paths=tuple(read_paths),
                write_paths=tuple(write_paths),
                acceptance_criteria=tuple(acceptance_criteria or ()),
                token_budget=token_budget,
                max_tool_calls=max_tool_calls,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                durable=durable,
            )
            if background:
                if review:
                    return "Sub-agent error: background tasks cannot request an inline reviewer"
                task_id = await self._parent_agent.submit_task(spec)
                snapshot = self._parent_agent.tasks.snapshot(task_id)
                return json.dumps({
                    "submitted": True,
                    "queued": snapshot is None,
                    "task_id": task_id,
                    "task": snapshot.model_dump(mode="json") if snapshot else None,
                }, ensure_ascii=False, separators=(",", ":"))
            result = await self._parent_agent.delegate(spec)
            if review and task_role == TaskRole.EXECUTOR and result.status == TaskStatus.COMPLETED:
                review_text = await self._parent_agent._review(result.summary, task)
                result = result.model_copy(
                    update={"evidence": [*result.evidence, f"Parent-scheduled review: {review_text}"]}
                )
            return result.model_dump_json()
        except Exception as e:
            return f"Sub-agent error: {e}"

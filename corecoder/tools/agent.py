"""Centralised sub-agent task submission.

The idea: for complex sub-tasks, spawn an independent agent with its own
conversation history and tool access. This lets the main agent delegate
work like "go research this codebase and report back" without polluting
its own context window.

The model-facing API has one primitive: delegate one bounded task. The main
agent chooses zero, one, or several calls and schedules dependent phases later.
"""

import json

from ..delegation import TaskRole, TaskSpec, WorkspaceMode
from ..orchestration import WorkflowRequest
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
        "Delegate one substantial independent sub-task with bounded tools, paths, "
        "tokens, calls, and runtime. Issue multiple independent sub-tasks together; "
        "wait before delegating dependent work. Keep simple or tightly coupled work local. "
        "Use background=true with task_control for asynchronous work."
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
                "description": "executor edits; researcher/reviewer are read-only.",
            },
            "background": {
                "type": "boolean",
                "description": "Return a task id without waiting.",
            },
            "durable": {
                "type": "boolean",
                "description": "Persist an encrypted background task for recovery.",
            },
            "allowed_tools": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "read_file",
                        "grep",
                        "glob",
                        "retrieve_context",
                        "write_file",
                        "edit_file",
                        "edit_ast",
                    ],
                },
                "description": (
                    "Exact allowlist; bash, agent, task_control and undo are invalid."
                ),
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
                "description": "Minimal task-specific context.",
            },
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conditions the parent will verify.",
            },
            "token_budget": {
                "type": "integer",
                "minimum": 8000,
                "maximum": 1000000,
                "description": (
                    "Child token limit. Omit for the 16000 default; use at least 8000 "
                    "even for one bounded read."
                ),
            },
            "max_tool_calls": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Child tool-call limit.",
            },
            "timeout_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 3600,
                "description": "Task timeout.",
            },
            "execution_mode": {
                "type": "string",
                "enum": ["fork", "worktree"],
                "description": "Shared fork or isolated Git worktree.",
            },
            "max_retries": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": "Retries for read-only tasks.",
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
        if durable and not background:
            return "Sub-agent error: durable mode requires background=true"
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
                task_id = await self._parent_agent.submit_task(spec)
                snapshot = self._parent_agent.tasks.snapshot(task_id)
                return json.dumps({
                    "submitted": True,
                    "queued": snapshot is None,
                    "task_id": task_id,
                    "task": snapshot.model_dump(mode="json") if snapshot else None,
                }, ensure_ascii=False, separators=(",", ":"))
            workflow = await self._parent_agent.run_workflow(WorkflowRequest(task=spec))
            result = workflow.final_task_result
            if result is None:
                return workflow.model_dump_json()
            return result.model_dump_json()
        except Exception as e:
            return f"Sub-agent error: {e}"

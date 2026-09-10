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
        "Delegate one substantial, well-scoped sub-task to an independent child agent. "
        "The main agent decides whether delegation is useful and how many children are needed: "
        "do not delegate simple work; for multiple independent sub-tasks, issue multiple agent "
        "tool calls in the same response so they can run concurrently. Submit dependent work "
        "in a later response after its prerequisites return. "
        "The parent task controller gives the child an independent context, "
        "bounded tools, paths, tokens, calls and runtime. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window. "
        "Set background=true to receive a task id immediately, then use "
        "task_control to inspect, wait for, or cancel it. "
        "Set the 'role' to 'researcher' for read-only exploration, "
        "'executor' for making changes, or 'reviewer' to check changes. "
        "Delegate a reviewer explicitly after the implementation result is available."
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
            "background": {
                "type": "boolean",
                "description": "Schedule this child and return its task id without waiting.",
            },
            "durable": {
                "type": "boolean",
                "description": "Encrypt and persist this background TaskSpec for crash recovery.",
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
                    "Exact child tool allowlist. Never include bash, agent, task_control, "
                    "or undo_changes; delegated children cannot use unscoped tools."
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

"""Sub-agent spawning (inspired by Claude Code's AgentTool, 1397 lines).

The idea: for complex sub-tasks, spawn an independent agent with its own
conversation history and tool access. This lets the main agent delegate
work like "go research this codebase and report back" without polluting
its own context window.

v1.0: supports role-based spawning (executor / reviewer / researcher)
and optional reviewer pass after executor completion.
"""

from .base import Tool


class AgentTool(Tool):
    name = "agent"
    description = (
        "Spawn a sub-agent to handle a complex sub-task independently. "
        "The sub-agent has its own context and tool access. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window. "
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
            "review": {
                "type": "boolean",
                "description": "If true and role is executor, a reviewer sub-agent will check the output. Default: false.",
            },
        },
        "required": ["task"],
    }

    # set by Agent.__init__ after construction
    _parent_agent = None

    async def execute(self, task: str, role: str = "executor", review: bool = False) -> str:
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        # import here to avoid circular dep
        from ..agent import AgentRole

        role_map = {
            "executor": AgentRole.EXECUTOR,
            "researcher": AgentRole.RESEARCHER,
            "reviewer": AgentRole.REVIEWER,
        }
        agent_role = role_map.get(role, AgentRole.EXECUTOR)

        try:
            result = await self._parent_agent.spawn(
                task=task,
                role=agent_role,
                reviewer=review and agent_role == AgentRole.EXECUTOR,
            )
            return f"[Sub-agent ({role}) completed]\n{result}"
        except Exception as e:
            return f"Sub-agent error: {e}"

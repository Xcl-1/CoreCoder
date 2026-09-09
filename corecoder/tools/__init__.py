"""Tool registry."""

from .agent import AgentTool
from .bash import BashTool
from .edit import EditFileTool
from .edit_ast import EditASTTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .read import ReadFileTool
from .task_control import TaskControlTool
from .undo import UndoChangesTool
from .write import WriteFileTool


def create_tools():
    """Create an isolated tool registry for one top-level agent."""
    return [
        BashTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        EditASTTool(),
        GlobTool(),
        GrepTool(),
        UndoChangesTool(),
        AgentTool(),
        TaskControlTool(),
    ]


# Retain the public registry for discovery and backwards compatibility. Agents
# create their own instances so mutable parent bindings never cross workspaces.
ALL_TOOLS = create_tools()


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None

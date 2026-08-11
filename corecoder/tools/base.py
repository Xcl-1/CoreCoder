"""Base class for all tools."""

import asyncio
from abc import ABC


class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities.

    For most tools, override ``_execute_sync(**kwargs) -> str`` — the
    default ``execute()`` delegates to it via ``asyncio.to_thread`` so the
    event loop stays free for other work.

    Tools that need genuine async I/O (e.g. sub-agent spawning or
    context-var management) can override ``execute()`` directly instead.
    """

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args

    async def execute(self, **kwargs) -> str:
        """Run the tool and return a text result.

        The default implementation delegates to ``_execute_sync`` in a
        thread so the event loop is not blocked by synchronous I/O.
        """
        return await asyncio.to_thread(self._execute_sync, **kwargs)

    def _execute_sync(self, **kwargs) -> str:
        """Synchronous tool body.  Override this for simple tools."""
        raise NotImplementedError(
            f"{self.name}: override execute() (async) or _execute_sync() (sync)"
        )

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

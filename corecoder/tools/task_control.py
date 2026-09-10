"""Parent-only inspection and cancellation of delegated background tasks."""

from __future__ import annotations

import asyncio
import json

from .base import Tool


class TaskControlTool(Tool):
    """Expose bounded controller state to the main agent, never to children."""

    name = "task_control"
    input_types = ("action", "task_id", "after_sequence")
    output_type = "task_control_result"
    permission_scope = "agent:delegate"
    side_effect = "delegated"
    network_access = "none"
    declared_risk = "medium"
    description = (
        "List, inspect, wait for, or cancel this parent's background tasks. "
        "events returns cursor-based progress; wait timeouts do not stop the task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "status", "result", "events", "wait", "cancel"],
                "description": "Operation.",
            },
            "task_id": {
                "type": "string",
                "description": "Required except for list.",
            },
            "timeout_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 300,
                "description": "Wait/events timeout; does not stop the task.",
            },
            "after_sequence": {
                "type": "integer",
                "minimum": 0,
                "description": "Events cursor.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum list size.",
            },
        },
        "required": ["action"],
    }

    _parent_agent = None

    @staticmethod
    def _json(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    async def execute(
        self,
        action: str,
        task_id: str = "",
        timeout_seconds: float = 30,
        limit: int = 20,
        after_sequence: int = 0,
    ) -> str:
        if self._parent_agent is None:
            return self._json({"ok": False, "error": "task control tool is not initialized"})

        self._parent_agent.refresh_task_state()
        controller = self._parent_agent.tasks
        if action == "list":
            snapshots = [item.model_dump(mode="json") for item in controller.list_tasks(limit=limit)]
            return self._json({"ok": True, "tasks": snapshots, "count": len(snapshots)})
        if not task_id:
            return self._json({"ok": False, "error": f"task_id is required for {action}"})

        snapshot = controller.snapshot(task_id)
        queued = self._parent_agent.is_task_queued(task_id)
        if snapshot is None and not queued:
            return self._json({"ok": False, "task_id": task_id, "error": "task not found"})

        if snapshot is None and action in {"status", "result"}:
            return self._json({
                "ok": True,
                "ready": False,
                "queued": True,
                "task": {"task_id": task_id, "status": "queued"},
            })

        if action == "status":
            return self._json({"ok": True, "task": snapshot.model_dump(mode="json")})
        if action == "result":
            result = controller.result(task_id)
            return self._json({
                "ok": True,
                "ready": result is not None,
                "task": (result or snapshot).model_dump(mode="json"),
            })
        if action == "events":
            batch = await self._parent_agent.wait_task_events(
                task_id,
                after_sequence=after_sequence,
                timeout=timeout_seconds,
                limit=limit,
            )
            return self._json({"ok": True, **batch.model_dump(mode="json")})
        if action == "cancel":
            cancelled = self._parent_agent.cancel_task(task_id)
            current = controller.snapshot(task_id) or snapshot
            if current is None:
                return self._json({
                    "ok": False,
                    "cancel_requested": False,
                    "task": {"task_id": task_id, "status": "queued"},
                })
            return self._json({
                "ok": cancelled,
                "cancel_requested": cancelled,
                "task": current.model_dump(mode="json"),
            })
        if action == "wait":
            try:
                result = await self._parent_agent.wait_task(task_id, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                current = controller.snapshot(task_id) or snapshot
                return self._json({
                    "ok": True,
                    "ready": False,
                    "wait_timed_out": True,
                    "task": current.model_dump(mode="json"),
                })
            except RuntimeError as exc:
                return self._json({"ok": False, "task_id": task_id, "error": str(exc)})
            return self._json({"ok": True, "ready": True, "task": result.model_dump(mode="json")})
        return self._json({"ok": False, "error": f"unsupported action: {action}"})

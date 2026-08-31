"""Tests for session-scoped, conflict-aware file undo."""

from __future__ import annotations

import pytest

from corecoder.agent import Agent
from corecoder.llm import LLM
from corecoder.security import PermissionManager
from corecoder.tools import get_tool


class _TC:
    def __init__(self, name: str, arguments: dict):
        self.id = f"call-{name}"
        self.name = name
        self.arguments = arguments


def _agent(*tool_names: str) -> Agent:
    return Agent(
        llm=LLM.__new__(LLM),
        tools=[get_tool(name) for name in tool_names],
        replay=False,
    )


@pytest.mark.asyncio
async def test_undo_restores_original_after_multiple_edits(tmp_path):
    path = tmp_path / "sample.py"
    path.write_bytes(b"alpha\nbeta\n")
    agent = _agent("edit_file")

    first = await agent._exec_tool(_TC("edit_file", {
        "file_path": str(path), "old_string": "alpha", "new_string": "one",
    }))
    second = await agent._exec_tool(_TC("edit_file", {
        "file_path": str(path), "old_string": "beta", "new_string": "two",
    }))

    assert first[2] and second[2]
    assert path.read_text() == "one\ntwo\n"
    assert len(agent.changes) == 1
    result = agent.changes.undo_all()
    assert result.restored == [str(path.resolve())]
    assert path.read_bytes() == b"alpha\nbeta\n"
    assert not agent.changes.changed_files


@pytest.mark.asyncio
async def test_undo_deletes_new_file_and_empty_parent_directories(tmp_path):
    path = tmp_path / "new" / "nested" / "created.txt"
    agent = _agent("write_file")
    result = await agent._exec_tool(_TC("write_file", {
        "file_path": str(path), "content": "created\n",
    }))
    assert result[2]
    assert path.exists()

    undone = agent.changes.undo_all()
    assert undone.deleted == [str(path.resolve())]
    assert not path.exists()
    assert not (tmp_path / "new").exists()


@pytest.mark.asyncio
async def test_undo_restores_overwritten_file(tmp_path):
    path = tmp_path / "existing.txt"
    path.write_bytes(b"original\r\nbytes\x00")
    agent = _agent("write_file")
    await agent._exec_tool(_TC("write_file", {
        "file_path": str(path), "content": "replacement\n",
    }))

    result = agent.changes.undo_all()
    assert not result.conflicts
    assert path.read_bytes() == b"original\r\nbytes\x00"


@pytest.mark.asyncio
async def test_undo_refuses_external_change_then_force_restores(tmp_path):
    path = tmp_path / "conflict.txt"
    path.write_text("original", encoding="utf-8")
    agent = _agent("write_file")
    await agent._exec_tool(_TC("write_file", {
        "file_path": str(path), "content": "agent version",
    }))
    path.write_text("external version", encoding="utf-8")

    safe = agent.changes.undo_all()
    assert safe.conflicts == [str(path.resolve())]
    assert path.read_text(encoding="utf-8") == "external version"
    assert len(agent.changes) == 1

    forced = agent.changes.undo_all(force=True)
    assert forced.restored == [str(path.resolve())]
    assert path.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_change_tracking_is_isolated_between_agents(tmp_path):
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("first", encoding="utf-8")
    second_path.write_text("second", encoding="utf-8")
    first_agent = _agent("write_file")
    second_agent = _agent("write_file")

    await first_agent._exec_tool(_TC("write_file", {
        "file_path": str(first_path), "content": "changed first",
    }))
    await second_agent._exec_tool(_TC("write_file", {
        "file_path": str(second_path), "content": "changed second",
    }))
    first_agent.changes.undo_all()

    assert first_path.read_text(encoding="utf-8") == "first"
    assert second_path.read_text(encoding="utf-8") == "changed second"
    assert len(second_agent.changes) == 1


@pytest.mark.asyncio
async def test_undo_tool_uses_current_agent_tracker(tmp_path):
    path = tmp_path / "tool.txt"
    path.write_text("before", encoding="utf-8")
    agent = _agent("write_file", "undo_changes")
    await agent._exec_tool(_TC("write_file", {
        "file_path": str(path), "content": "after",
    }))

    text, _, success = await agent._exec_tool(_TC("undo_changes", {}))
    assert success
    assert "1 restored" in text
    assert path.read_text(encoding="utf-8") == "before"


@pytest.mark.asyncio
async def test_reset_preserves_process_undo_history(tmp_path):
    path = tmp_path / "reset.txt"
    path.write_text("before", encoding="utf-8")
    agent = _agent("write_file")
    await agent._exec_tool(_TC("write_file", {
        "file_path": str(path), "content": "after",
    }))

    agent.reset()
    assert len(agent.changes) == 1
    assert path.read_text(encoding="utf-8") == "after"
    agent.changes.undo_all()
    assert path.read_text(encoding="utf-8") == "before"


def test_undo_requires_confirmation_by_default():
    rule = PermissionManager().match("undo_changes", {})
    assert rule is not None
    assert rule.action == "ask"

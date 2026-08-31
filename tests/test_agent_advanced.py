"""Tests for advanced agent features: role system, spawn, parallel scheduling."""


import pytest

from corecoder.agent import Agent, AgentRole, role_prompt, role_tools
from corecoder.llm import LLM
from corecoder.models import LLMResponse
from corecoder.tools import ALL_TOOLS

# --- Role system ---------------------------------------------------------

def test_all_roles_have_prompts():
    for role in AgentRole:
        prompt = role_prompt(role)
        assert len(prompt) > 0
        assert isinstance(prompt, str)


def test_planner_has_no_tools():
    """Planner should not get any tools — it only thinks."""
    tools = role_tools(AgentRole.PLANNER, ALL_TOOLS)
    assert tools == []


def test_reviewer_is_read_only():
    """Reviewer should only get read-only tools."""
    tools = role_tools(AgentRole.REVIEWER, ALL_TOOLS)
    tool_names = {t.name for t in tools}
    assert tool_names == {"read_file", "grep", "glob"}
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "bash" not in tool_names
    assert "agent" not in tool_names


def test_researcher_is_read_only():
    """Researcher should only get read-only tools."""
    tools = role_tools(AgentRole.RESEARCHER, ALL_TOOLS)
    tool_names = {t.name for t in tools}
    assert tool_names == {"read_file", "grep", "glob"}


def test_executor_gets_all_tools():
    """Executor gets the full tool set."""
    tools = role_tools(AgentRole.EXECUTOR, ALL_TOOLS)
    assert len(tools) == len(ALL_TOOLS)


@pytest.mark.asyncio
async def test_empty_length_response_gets_one_tool_free_finalization_attempt():
    class _SequenceLLM:
        def __init__(self):
            self.calls = []
            self.responses = [
                LLMResponse(
                    content="",
                    reasoning_content="unfinished reasoning",
                    finish_reason="length",
                    completion_tokens=8192,
                ),
                LLMResponse(content="final review", finish_reason="stop"),
            ]

        def chat(self, messages, tools=None, on_token=None):
            self.calls.append({"messages": messages, "tools": tools})
            return self.responses.pop(0)

    llm = _SequenceLLM()
    agent = Agent(llm=llm, tools=[], replay=False)

    result = await agent.chat("review this")

    assert result == "final review"
    assert len(llm.calls) == 2
    assert llm.calls[1]["tools"] is None
    assert "Return the final answer now" in llm.calls[1]["messages"][-1]["content"]


# --- Parallel execution --------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_readers_run_concurrently():
    """Multiple independent reads should all complete successfully."""
    agent = Agent(llm=LLM.__new__(LLM), tools=ALL_TOOLS, replay=False)

    class _TC:
        def __init__(self, i, name, args=None):
            self.id = f"tc{i}"
            self.name = name
            self.arguments = args or {}

    # readers targeting different files
    tcs = [
        _TC(1, "glob", {"pattern": "*.py", "path": "."}),
        _TC(2, "grep", {"pattern": "def", "path": "."}),
    ]
    results = await agent._exec_tools_async(tcs)
    assert len(results) == 2
    for tc, (result, elapsed, success) in results:
        assert isinstance(result, str)
        assert elapsed >= 0
        assert isinstance(success, bool)


@pytest.mark.asyncio
async def test_parallel_mixed_reader_writer(tmp_path):
    """Writers are sequenced after readers on the same path, others run parallel."""
    agent = Agent(llm=LLM.__new__(LLM), tools=ALL_TOOLS, replay=False)
    test_file = tmp_path / "test.txt"
    test_file.write_text("original\n")

    class _TC:
        def __init__(self, i, name, args=None):
            self.id = f"tc{i}"
            self.name = name
            self.arguments = args or {}

    tcs = [
        _TC(1, "read_file", {"file_path": str(test_file)}),
        _TC(2, "edit_file", {"file_path": str(test_file), "old_string": "original", "new_string": "modified"}),
        _TC(3, "glob", {"pattern": "*.txt", "path": str(tmp_path)}),
    ]
    results = await agent._exec_tools_async(tcs)
    assert len(results) == 3
    # All should succeed (edit_file finds its string because it waited for the read to finish)
    for tc, (result, elapsed, success) in results:
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_parallel_writers_different_paths(tmp_path):
    """Writers targeting different paths should run in parallel (no sequencing needed)."""
    agent = Agent(llm=LLM.__new__(LLM), tools=ALL_TOOLS, replay=False)
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello\n")
    f2.write_text("world\n")

    class _TC:
        def __init__(self, i, name, args=None):
            self.id = f"tc{i}"
            self.name = name
            self.arguments = args or {}

    tcs = [
        _TC(1, "write_file", {"file_path": str(f1), "content": "new_a\n"}),
        _TC(2, "write_file", {"file_path": str(f2), "content": "new_b\n"}),
    ]
    results = await agent._exec_tools_async(tcs)
    assert len(results) == 2
    for tc, (result, elapsed, success) in results:
        assert success


@pytest.mark.asyncio
async def test_parallel_unknown_tool_graceful():
    """Unknown tools should return an error string, not crash."""
    agent = Agent(llm=LLM.__new__(LLM), tools=ALL_TOOLS, replay=False)

    class _TC:
        def __init__(self, i):
            self.id = f"x{i}"
            self.name = "nonexistent_tool"
            self.arguments = {}

    results = await agent._exec_tools_async([_TC(1), _TC(2)])
    assert len(results) == 2
    for tc, (result, elapsed, success) in results:
        assert "unknown tool" in result.lower()
        assert not success


@pytest.mark.asyncio
async def test_parallel_empty_list():
    """Executing zero tool calls should return an empty list."""
    agent = Agent(llm=LLM.__new__(LLM), tools=ALL_TOOLS, replay=False)
    results = await agent._exec_tools_async([])
    assert results == []


# --- Interrupt safety ----------------------------------------------------

def test_answer_pending_tool_calls():
    """Backfill only the calls that don't already have a reply."""
    agent = Agent(llm=LLM.__new__(LLM), tools=[], replay=False)
    agent.messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "done"},
    ]

    class _TC:
        def __init__(self, i):
            self.id = i

    agent._answer_pending_tool_calls([_TC("a"), _TC("b")])
    replies = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(replies) == 2
    ids = {m["tool_call_id"] for m in replies}
    assert ids == {"a", "b"}

    # The already-answered call "a" should still have its original content
    a_replies = [m for m in replies if m["tool_call_id"] == "a"]
    assert a_replies[0]["content"] == "done"

    # The unanswered call "b" gets an [interrupted] placeholder
    b_replies = [m for m in replies if m["tool_call_id"] == "b"]
    assert b_replies[0]["content"] == "[interrupted]"

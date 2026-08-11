"""Tests for enhanced context compression: incremental summarisation,
tiktoken fallback, L2.5 layered compression, and dynamic thresholds."""

import pytest

from corecoder.context import (
    ContextManager,
    estimate_tokens,
)


# --- Token estimation ----------------------------------------------------

def test_estimate_tokens_basic():
    msgs = [{"role": "user", "content": "hello world"}]
    t = estimate_tokens(msgs)
    assert t > 0
    assert t < 50


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


def test_estimate_tokens_multiple_messages():
    msgs = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a function."},
        {"role": "assistant", "content": "Here is a function:\n```python\ndef f():\n    return 1\n```"},
    ]
    t = estimate_tokens(msgs)
    assert t > 10


def test_estimate_tokens_tool_calls():
    msgs = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
        ],
    }]
    t = estimate_tokens(msgs)
    assert t > 0  # tool_calls JSON counts toward the total


def test_estimate_tokens_none_content():
    msgs = [{"role": "assistant", "content": None}]
    t = estimate_tokens(msgs)
    assert t == 0


# --- L1: tool-type-aware snipping ----------------------------------------

def test_snip_never_touches_grep():
    """grep outputs are never snipped — the match list IS the value."""
    ctx = ContextManager(max_tokens=5000)
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "g1", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "g1", "content": "\n".join(f"line{i}" for i in range(50))},
    ]
    before = msgs[1]["content"]
    ctx._snip_tool_outputs(msgs)
    assert msgs[1]["content"] == before  # untouched


def test_snip_bash_uses_generous_head_tail():
    """bash outputs get 40+40 line snip, more generous than the default 3+3."""
    ctx = ContextManager(max_tokens=5000)
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "b1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "b1", "content": "\n".join(f"line_{i}" for i in range(200))},
    ]
    ctx._snip_tool_outputs(msgs)
    content = msgs[1]["content"]
    assert "line_0" in content
    assert "line_199" in content
    assert "snipped" in content


def test_snip_default_fallback_head_tail():
    """Unknown tools get the default 3+3 line snip."""
    ctx = ContextManager(max_tokens=5000)
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "u1", "type": "function", "function": {"name": "unknown_tool", "arguments": "{}"}},
        ]},
        # each line must be long enough that total output exceeds 1500 chars
        {"role": "tool", "tool_call_id": "u1", "content": "\n".join(f"L{i}_" + "x" * 40 for i in range(100))},
    ]
    ctx._snip_tool_outputs(msgs)
    content = msgs[1]["content"]
    assert "L0_" in content
    assert "L99_" in content
    assert "snipped" in content


def test_snip_short_output_untouched():
    """Output under 1500 chars should never be snipped."""
    ctx = ContextManager(max_tokens=5000)
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "s1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "s1", "content": "short"},
    ]
    before = msgs[1]["content"]
    ctx._snip_tool_outputs(msgs)
    assert msgs[1]["content"] == before


# --- L2: incremental summarisation ---------------------------------------

def test_incremental_summarize_updates_summary_state():
    """After incremental summarisation, _summary_text should be non-empty."""
    ctx = ContextManager(max_tokens=3000)
    msgs = []
    for i in range(15):
        msgs.append({"role": "user", "content": f"msg {i} " + "x" * 300})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "y" * 500})
    result = ctx._incremental_summarize(msgs, llm=None, keep_recent=6)
    assert result is True
    assert len(ctx._summary_text) > 0
    assert ctx._last_summary_index > 0


def test_incremental_summarize_only_new_material():
    """Second call with no new material should return False."""
    ctx = ContextManager(max_tokens=5000)
    msgs = []
    for i in range(12):
        msgs.append({"role": "user", "content": f"msg {i} " + "x" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "y" * 300})
    # first summarise
    ctx._incremental_summarize(msgs, llm=None, keep_recent=6)
    # second call — no new material
    result = ctx._incremental_summarize(msgs, llm=None, keep_recent=6)
    assert result is False


def test_incremental_summarize_fallback_extracts_key_info():
    """When LLM is None, fallback uses regex-based extraction."""
    ctx = ContextManager(max_tokens=3000)
    msgs = []
    for i in range(10):
        msgs.append({"role": "user", "content": f"edit src/main.py " + "x" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "Error: file not found"})
    result = ctx._incremental_summarize(msgs, llm=None, keep_recent=4)
    assert result is True
    # The summary text should mention the file or error
    assert "src/main.py" in ctx._summary_text or "Error" in ctx._summary_text


# --- L2.5: layered compression -------------------------------------------

def test_layered_compress_demotes_long_tool_outputs():
    """L2.5 should truncate old verbose tool outputs to one-line markers."""
    ctx = ContextManager(max_tokens=5000)
    msgs = []
    for i in range(12):
        msgs.append({"role": "user", "content": f"task {i}"})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "Z" * 1000})
    # add recent messages that should be preserved
    msgs.append({"role": "user", "content": "recent task"})
    msgs.append({"role": "tool", "tool_call_id": "t_recent", "content": "result"})

    result = ctx._layered_compress(msgs, keep_recent=4)
    assert result is True
    # recent tool output should be preserved
    assert msgs[-1]["content"] == "result"
    # old tool outputs should have been shortened
    for i in range(6):  # first few messages should have been compressed
        if msgs[i].get("role") == "tool":
            content = msgs[i]["content"]
            assert len(content) < 1000  # significantly shortened


def test_layered_compress_preserves_system_messages():
    """System messages should NEVER be touched by L2.5."""
    ctx = ContextManager(max_tokens=5000)
    sys_msg = {"role": "system", "content": "VERY IMPORTANT SYSTEM INSTRUCTIONS " + "X" * 2000}
    msgs = [sys_msg]
    for i in range(12):
        msgs.append({"role": "user", "content": f"task {i}"})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "Y" * 800})
    msgs.append({"role": "user", "content": "recent"})
    msgs.append({"role": "tool", "tool_call_id": "t_final", "content": "final"})

    before_system = msgs[0]["content"]
    ctx._layered_compress(msgs, keep_recent=2)
    assert msgs[0]["content"] == before_system  # untouched


# --- L3: hard collapse ---------------------------------------------------

def test_hard_collapse_keeps_tail():
    """After hard collapse, recent messages should still be present."""
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 300})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 600})
    # mark the last few as recent
    msgs[-4]["content"] = "KEEP_ME_RECENT"

    original_len = len(msgs)
    ctx._hard_collapse(msgs, llm=None)
    assert len(msgs) < original_len
    # the recent marker should still be in the tail
    contents = [m.get("content", "") for m in msgs]
    assert any("KEEP_ME_RECENT" in c for c in contents)


def test_hard_collapse_includes_summary_header():
    """Hard collapse output should include a reset marker."""
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(15):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 400})

    ctx._hard_collapse(msgs, llm=None)
    first_content = msgs[0].get("content", "")
    assert "Hard context reset" in first_content or "context" in first_content.lower()


# --- Full maybe_compress pipeline ----------------------------------------

def test_maybe_compress_does_nothing_under_threshold():
    """When tokens are under the snip threshold, nothing changes."""
    ctx = ContextManager(max_tokens=500_000)
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    before = list(msgs)  # shallow copy
    changed = ctx.maybe_compress(msgs, llm=None)
    assert changed is False
    assert msgs == before


def test_maybe_compress_triggers_multiple_layers():
    """With very full context, multiple compression layers should fire."""
    ctx = ContextManager(max_tokens=3000)
    msgs = []
    for i in range(30):
        msgs.append({"role": "user", "content": f"msg {i} " + "x" * 300})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "y" * 800})
    original_len = len(msgs)
    changed = ctx.maybe_compress(msgs, llm=None)
    assert changed is True
    # Should have been compressed significantly
    assert len(msgs) < original_len or estimate_tokens(msgs) < estimate_tokens(
        [{"role": "user", "content": f"msg {i} " + "x" * 300} for i in range(30)]
        + [{"role": "tool", "tool_call_id": f"t{i}", "content": "y" * 800} for i in range(30)]
    )


# --- _safe_split ---------------------------------------------------------

def test_safe_split_never_starts_with_tool():
    """The split point must never land on a 'tool' message."""
    ctx = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    split = ctx._safe_split(messages, keep_recent=1)
    # The tail (messages[split:]) must not start with a 'tool' message
    assert messages[split].get("role") != "tool"


def test_safe_split_keeps_assistant_tool_pair_together():
    """An assistant+tool pair must stay together (never split between them)."""
    ctx = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "user", "content": "q2"},
    ]
    split = ctx._safe_split(messages, keep_recent=2)
    tail = messages[split:]
    # tail should not start mid-pair
    assert tail[0].get("role") != "tool"


# --- Edge cases ----------------------------------------------------------

def test_context_manager_reset_state():
    """After hard collapse, incremental summary state is reset."""
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(15):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 400})

    ctx._hard_collapse(msgs, llm=None)
    assert ctx._last_summary_index == 0
    assert ctx._summary_text == ""

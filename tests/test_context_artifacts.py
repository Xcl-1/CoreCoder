"""Tests for externalized tool results and cache-friendly context retrieval."""

from __future__ import annotations

import os
import time

import pytest

from corecoder.agent import Agent
from corecoder.context import ContextManager, ContextNote, estimate_request_tokens, estimate_tokens
from corecoder.context_artifacts import ContextArtifactStore
from corecoder.models import LLMResponse, ToolCall
from corecoder.tools.base import Tool


def _large_result(lines: int = 160) -> str:
    return "\n".join(f"line {number}: {'x' * 40}" for number in range(1, lines + 1))


def test_small_result_stays_inline_without_creating_storage(tmp_path):
    store = ContextArtifactStore("session-a", root=tmp_path, threshold_chars=1_000)

    result = store.externalize("small result", "bash")

    assert result == "small result"
    assert not store.session_dir.exists()


def test_large_result_gets_stable_placeholder_and_deduplicated_file(tmp_path):
    store = ContextArtifactStore("session-a", root=tmp_path, threshold_chars=1_000)
    content = _large_result()

    first = store.externalize(content, "bash")
    second = store.externalize(content, "bash")

    assert first == second
    assert first.startswith("[Tool Artifact]\nid: artifact://sha256/")
    assert "artifact preview truncated" in first
    assert "Use retrieve_context" in first
    assert len(list(store.session_dir.glob("*.txt"))) == 1
    assert len(first) < len(content)


def test_retrieve_supports_keyword_and_line_ranges(tmp_path):
    store = ContextArtifactStore("session-a", root=tmp_path, threshold_chars=1_000)
    content = "alpha\nbefore failure\nERROR database unavailable\nafter failure\nomega"
    artifact_id = store.put(content, "bash")

    matches = store.retrieve(artifact_id, query="database")
    line_range = store.retrieve(artifact_id, start_line=2, end_line=4)

    assert "2: before failure" in matches
    assert "3: ERROR database unavailable" in matches
    assert "4: after failure" in matches
    assert "1: alpha" not in line_range
    assert "2: before failure" in line_range
    assert "4: after failure" in line_range


def test_artifacts_are_session_isolated_and_ids_are_validated(tmp_path):
    first = ContextArtifactStore("session-a", root=tmp_path)
    second = ContextArtifactStore("session-b", root=tmp_path)
    artifact_id = first.put("private observation", "read_file")

    assert "not found in this session" in second.retrieve(artifact_id)
    with pytest.raises(ValueError, match="artifact_id must use"):
        first.retrieve("../../outside")


def test_prune_removes_expired_artifact_and_metadata(tmp_path):
    store = ContextArtifactStore(
        "session-a",
        root=tmp_path,
        ttl_seconds=60,
        max_total_bytes=1_000_000,
    )
    artifact_id = store.put("expired observation", "bash")
    digest = artifact_id.rsplit("/", 1)[-1]
    text_path = store.session_dir / f"{digest}.txt"
    metadata_path = store.session_dir / f"{digest}.json"
    expired_time = time.time() - 120
    os.utime(text_path, (expired_time, expired_time))

    result = store.prune(now=time.time())

    assert result.removed_artifacts == 1
    assert not text_path.exists()
    assert not metadata_path.exists()
    assert store.stats()["pruned"] == 1


def test_capacity_prune_keeps_new_artifact_and_removes_oldest(tmp_path):
    store = ContextArtifactStore(
        "session-a",
        root=tmp_path,
        ttl_seconds=3_600,
        max_total_bytes=3_000,
    )
    first_id = store.put("a" * 1_800, "bash")
    first_path = store.session_dir / f"{first_id.rsplit('/', 1)[-1]}.txt"
    old_time = time.time() - 10
    os.utime(first_path, (old_time, old_time))

    second_id = store.put("b" * 1_800, "bash")

    assert "not found" in store.retrieve(first_id)
    assert "1: " + "b" * 20 in store.retrieve(second_id, max_chars=500)
    assert store.stats()["pruned"] == 1


def test_complete_request_budget_includes_tools_and_output_reserve():
    messages = [{"role": "user", "content": "hello"}]
    schemas = [{"type": "function", "function": {"name": "read_file", "description": "x" * 400}}]

    messages_only = estimate_tokens(messages)
    full_budget = estimate_request_tokens(messages, schemas, reserve_tokens=2_000)

    assert full_budget > messages_only + 2_000


def test_context_manager_externalizes_at_ingress_but_not_retrieval_results(tmp_path):
    store = ContextArtifactStore("session-a", root=tmp_path, threshold_chars=1_000)
    context = ContextManager(artifact_store=store)
    content = _large_result()

    placeholder = context.prepare_tool_result(content, "bash")
    retrieved = context.prepare_tool_result(content, "retrieve_context")

    assert placeholder.startswith("[Tool Artifact]")
    assert retrieved == content


def test_context_metrics_cover_externalization_retrieval_and_compression(tmp_path):
    store = ContextArtifactStore("session-a", root=tmp_path, threshold_chars=1_000)
    context = ContextManager(max_tokens=2_000, artifact_store=store)
    placeholder = context.prepare_tool_result(_large_result(), "bash")
    artifact_id = placeholder.splitlines()[1].removeprefix("id: ")
    store.retrieve(artifact_id, query="line 80")
    messages = [
        {"role": "user", "content": f"task {number} " + "x" * 300}
        for number in range(20)
    ]

    context.maybe_compress(messages, llm=None)
    stats = context.stats()

    assert stats["artifacts"]["externalized"] == 1
    assert stats["artifacts"]["saved_prompt_chars"] > 0
    assert stats["artifacts"]["retrievals"] == 1
    assert stats["compression_runs"] == 1
    assert stats["tokens_removed"] > 0


def test_artifact_reference_survives_repeated_checkpoints_and_restores_evidence(tmp_path):
    store = ContextArtifactStore("long-session", root=tmp_path, threshold_chars=1_000)
    exact_evidence = "\n".join(
        f"evidence line {number}: {'important' if number == 73 else 'detail'}"
        for number in range(1, 160)
    )
    placeholder = store.externalize(exact_evidence, "bash")
    artifact_id = placeholder.splitlines()[1].removeprefix("id: ")
    context = ContextManager(artifact_store=store)
    messages = [
        {"role": "user", "content": "Only inspect results; do not modify files."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "large-1", "function": {"name": "bash"}}],
        },
        {"role": "tool", "tool_call_id": "large-1", "content": placeholder},
    ]
    messages.extend(
        {"role": "assistant", "content": f"progress checkpoint {number}"}
        for number in range(12)
    )

    assert context._incremental_summarize(messages, llm=None, keep_recent=6)
    first_note = ContextNote.from_json(messages[0]["content"])
    messages.extend(
        {"role": "assistant", "content": f"later progress {number}"}
        for number in range(10)
    )
    assert context._incremental_summarize(messages, llm=None, keep_recent=6)
    second_note = ContextNote.from_json(messages[0]["content"])

    assert first_note is not None and artifact_id in first_note.artifact_refs
    assert second_note is not None and artifact_id in second_note.artifact_refs
    assert second_note.constraints == ["Only inspect results; do not modify files."]
    restored = store.retrieve(artifact_id, query="important")
    assert "evidence line 73: important" in restored


class _BigTool(Tool):
    name = "big_tool"
    description = "Return a large deterministic observation."
    side_effect = "none"

    def __init__(self):
        self.parameters = {"type": "object", "properties": {}, "required": []}

    def _execute_sync(self) -> str:
        return _large_result(400)


class _SequenceLLM:
    def __init__(self):
        self.extra = {"max_tokens": 1_000}
        self.calls = []
        self.responses = [
            LLMResponse(tool_calls=[ToolCall(id="big-1", name="big_tool")]),
            LLMResponse(content="done"),
        ]

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)


class _SanitizingGuard:
    class _Decision:
        allowed = True
        user_confirmed = False
        reason = "test"

    def review(self, tool_name, arguments):
        return self._Decision()

    @staticmethod
    def sanitize(text):
        return text.replace("UNSAFE_SECRET", "[REDACTED]")


@pytest.mark.asyncio
async def test_agent_externalizes_before_second_model_request(tmp_path):
    store = ContextArtifactStore("session-a", root=tmp_path, threshold_chars=1_000)
    llm = _SequenceLLM()
    agent = Agent(
        llm=llm,
        tools=[_BigTool()],
        artifact_store=store,
        replay=False,
    )

    answer = await agent.chat("collect the observation")

    assert answer == "done"
    assert "retrieve_context" in agent._tool_by_name
    tool_message = next(message for message in agent.messages if message.get("role") == "tool")
    assert tool_message["content"].startswith("[Tool Artifact]")
    second_call_tool_message = next(
        message for message in llm.calls[1]["messages"] if message.get("role") == "tool"
    )
    assert second_call_tool_message["content"] == tool_message["content"]
    assert len(second_call_tool_message["content"]) < len(_large_result(400))


@pytest.mark.asyncio
async def test_agent_sanitizes_before_persisting_artifact(tmp_path):
    store = ContextArtifactStore("session-a", root=tmp_path, threshold_chars=1_000)
    agent = Agent(
        llm=_SequenceLLM(),
        tools=[_BigTool()],
        artifact_store=store,
        guard=_SanitizingGuard(),
        replay=False,
    )
    original_execute = agent.tools[0]._execute_sync
    agent.tools[0]._execute_sync = lambda: "UNSAFE_SECRET\n" + original_execute()

    await agent.chat("collect sanitized evidence")

    persisted = next(store.session_dir.glob("*.txt")).read_text(encoding="utf-8")
    assert "UNSAFE_SECRET" not in persisted
    assert "[REDACTED]" in persisted


def test_default_agent_exposes_retrieval_and_can_disable_artifacts():
    enabled = Agent(llm=_SequenceLLM(), replay=False)
    disabled = Agent(
        llm=_SequenceLLM(),
        replay=False,
        context_artifacts_enabled=False,
    )

    assert "retrieve_context" in enabled._tool_by_name
    assert "retrieve_context" not in disabled._tool_by_name


def test_agent_reset_rotates_artifact_session(tmp_path):
    store = ContextArtifactStore("original", root=tmp_path, threshold_chars=1_000)
    agent = Agent(llm=_SequenceLLM(), tools=[], artifact_store=store, replay=False)
    old_id = store.put("old observation", "bash")

    agent.reset()

    assert agent.context_artifacts.session_id == agent.session_id
    assert "not found in this session" in agent.context_artifacts.retrieve(old_id)

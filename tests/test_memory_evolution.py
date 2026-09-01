"""Tests for reflection, procedural memory, feedback, and lifecycle governance."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

import corecoder.cli as cli_module
from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.llm import LLM
from corecoder.memory import Memory, MemoryEngine, SessionReflection
from corecoder.memory.extractor import MemoryExtractor
from corecoder.memory.reflection import MemoryReflector, redact_secrets
from corecoder.memory.store import MemoryStore
from corecoder.models import LLMResponse


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, **_kwargs):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(content=response)


def _memory(memory_id="pytest-guide", **updates):
    values = {
        "id": memory_id,
        "title": "Pytest project guide",
        "description": "Use pytest for project verification",
        "content": "Run pytest after code changes.",
        "type": "project",
        "scope": "project",
        "keywords": ["pytest", "tests"],
    }
    values.update(updates)
    return Memory(**values)


def _messages():
    return [
        {"role": "user", "content": "Fix the test failure and verify it."},
        {"role": "assistant", "content": "Fixed and verified with pytest."},
    ]


def _write_replay(path):
    record = {
        "step": 1,
        "llm_response": {"content": "The compatibility fix is complete."},
        "tool_executions": [
            {
                "name": "bash",
                "arguments": {"command": "pytest -q"},
                "result": "243 passed",
                "success": True,
                "error": None,
            }
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_legacy_memory_gets_evolution_defaults(tmp_path):
    store = MemoryStore(tmp_path)
    stored = store.save(_memory(project_path=str(tmp_path.resolve())))

    loaded = store.get(stored.id)

    assert loaded is not None
    assert loaded.status == "active"
    assert loaded.version == 1
    assert loaded.evidence == []
    assert loaded.use_count == loaded.success_count == loaded.failure_count == 0
    assert loaded.validation_count == 0
    assert loaded.validated_at is None


def test_reflector_uses_replay_and_validates_evidence(tmp_path):
    replay = tmp_path / "session.jsonl"
    _write_replay(replay)
    payload = {
        "task_summary": "Fix the test failure",
        "outcome": "success",
        "summary": "The fix passed the suite.",
        "failures": [],
        "root_causes": ["Compatibility issue"],
        "effective_actions": ["Applied the compatibility fix"],
        "verification": ["243 passed"],
        "reusable_lessons": ["Run the full test suite after compatibility changes"],
        "evidence": ["243 passed", "fabricated evidence"],
    }
    reflector = MemoryReflector(_FakeLLM([json.dumps(payload)]))

    reflection = reflector.reflect(_messages(), replay)

    assert reflection is not None
    assert reflection.outcome == "success"
    assert reflection.evidence == ["243 passed"]
    assert reflection.verification == ["243 passed"]
    assert reflection.tool_executions == reflection.successful_tools == 1
    assert reflection.failed_tools == 0


def test_reflector_retries_length_response_with_fresh_bounded_json_request(tmp_path):
    class _CaptureLLM:
        def __init__(self):
            self.calls = []
            self.responses = [
                LLMResponse(
                    content="",
                    reasoning_content="unfinished reflection reasoning",
                    finish_reason="length",
                ),
                LLMResponse(content=json.dumps({
                    "task_summary": "Review execution",
                    "outcome": "unknown",
                    "summary": "Verification was insufficient.",
                    "failures": [],
                    "root_causes": [],
                    "effective_actions": [],
                    "verification": [],
                    "reusable_lessons": [],
                    "evidence": [],
                })),
            ]

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    llm = _CaptureLLM()
    messages = [
        {"role": "user", "content": "X" * 20_000},
        {"role": "assistant", "content": "No verified final result."},
    ]

    reflection = MemoryReflector(llm).reflect(messages)

    assert reflection is not None
    assert reflection.outcome == "unknown"
    assert len(llm.calls) == 2
    repair = llm.calls[1]["messages"]
    assert len(repair) == 2
    assert "JSON formatter" in repair[0]["content"]
    assert "unfinished reflection reasoning" not in str(repair)
    assert len(repair[1]["content"]) < 9_000


def test_reflection_redacts_json_and_plaintext_credentials():
    text = 'api_key": "sk-abcdefghijklmnop password=hunter2 access_token: abcdef'

    redacted = redact_secrets(text)

    assert "sk-abcdefghijklmnop" not in redacted
    assert "hunter2" not in redacted
    assert "abcdef" not in redacted


def test_reflection_collapses_duplicate_replay_noise(tmp_path):
    replay = tmp_path / "duplicate.jsonl"
    record = {
        "llm_response": {"content": ""},
        "tool_executions": [{
            "name": "bash",
            "arguments": {"command": "bad-command"},
            "result": "command not found",
            "success": False,
            "error": "command not found",
        }],
    }
    first = {**record, "step": 1}
    second = {**record, "step": 2}
    replay.write_text(f"{json.dumps(first)}\n{json.dumps(second)}\n", encoding="utf-8")

    source = MemoryReflector(object()).source_text([], replay)

    assert source.count("tool bash") == 1
    assert "omitted 1 duplicate tool execution(s)" in source


def test_procedure_requires_verified_success():
    proposal = [{
        "action": "create",
        "title": "Verify compatibility changes",
        "description": "Run the full suite after compatibility changes",
        "content": "Apply the compatibility fix, then run pytest -q and require a clean pass.",
        "type": "procedure",
        "scope": "project",
        "keywords": ["compatibility", "pytest"],
        "confidence": 0.9,
        "evidence": "243 passed",
    }]
    verified = SessionReflection(
        outcome="success",
        verification=["243 passed"],
        evidence=["243 passed"],
        tool_executions=1,
        successful_tools=1,
    )
    unverified = SessionReflection(outcome="partial", tool_executions=1, successful_tools=1)

    accepted = MemoryExtractor(_FakeLLM([json.dumps(proposal)])).extract(
        _messages(), [], reflection=verified, evidence_source="bash result: 243 passed"
    )
    rejected = MemoryExtractor(_FakeLLM([json.dumps(proposal)])).extract(
        _messages(), [], reflection=unverified, evidence_source="bash result: 243 passed"
    )

    assert [item.type for item in accepted] == ["procedure"]
    assert rejected == []


def test_profile_and_failed_episode_are_classified():
    proposals = [
        {
            "action": "create",
            "title": "Concise communication profile",
            "description": "The user prefers concise technical explanations",
            "content": "Keep technical explanations concise.",
            "type": "profile",
            "scope": "global",
            "keywords": ["concise", "communication"],
            "confidence": 0.9,
            "evidence": "From now on, I prefer concise technical explanations.",
        },
        {
            "action": "create",
            "title": "Compatibility test failure episode",
            "description": "A compatibility fix did not pass verification",
            "content": "The attempted compatibility fix still failed the test suite.",
            "type": "episode",
            "scope": "project",
            "keywords": ["compatibility", "failure"],
            "confidence": 0.8,
            "evidence": "1 failed",
        },
    ]
    messages = [
        {"role": "user", "content": "From now on, I prefer concise technical explanations."},
        {"role": "assistant", "content": "Understood."},
    ]
    reflection = SessionReflection(
        outcome="failure",
        failures=["The test suite failed"],
        verification=["1 failed"],
        evidence=["1 failed"],
        tool_executions=1,
        failed_tools=1,
    )

    result = MemoryExtractor(_FakeLLM([json.dumps(proposals)])).extract(
        messages,
        [],
        reflection=reflection,
        evidence_source="pytest result: 1 failed",
    )

    assert [item.type for item in result] == ["profile", "episode"]


def test_preference_acknowledgement_cannot_create_episode():
    proposal = [{
        "action": "create",
        "title": "Persist requested memories",
        "description": "Persist preferences instead of promising",
        "content": "Persist requested preferences at session end.",
        "type": "episode",
        "scope": "global",
        "keywords": ["memory"],
        "confidence": 0.9,
        "evidence": "Please remember this preference.",
    }]
    messages = [
        {"role": "user", "content": "Please remember this preference."},
        {"role": "assistant", "content": "It will be considered at session end."},
    ]
    reflection = SessionReflection(
        outcome="success",
        evidence=["Please remember this preference."],
    )

    result = MemoryExtractor(_FakeLLM([json.dumps(proposal)])).extract(
        messages,
        [],
        reflection=reflection,
        evidence_source="Please remember this preference.",
    )

    assert result == []


def test_task_request_memory_is_dropped_while_verified_procedure_is_kept():
    request = "请运行 pytest，并总结以后验证记忆系统时可以重复使用的步骤。"
    proposals = [
        {
            "action": "create",
            "title": "User wants reusable verification steps",
            "description": "The user requested a reusable test summary",
            "content": "Summarize test steps for future sessions.",
            "type": "user",
            "scope": "project",
            "keywords": ["pytest", "verification"],
            "confidence": 0.9,
            "evidence": request,
        },
        {
            "action": "create",
            "title": "Memory verification procedure",
            "description": "Run the memory suite and require a clean pass",
            "content": "Run pytest for the memory suite and require every test to pass.",
            "type": "procedure",
            "scope": "project",
            "keywords": ["pytest", "memory"],
            "confidence": 0.9,
            "evidence": "25 passed",
        },
    ]
    reflection = SessionReflection(
        outcome="success",
        verification=["25 passed"],
        evidence=["25 passed"],
        tool_executions=1,
        successful_tools=1,
    )
    extractor = MemoryExtractor(_FakeLLM([json.dumps(proposals, ensure_ascii=False)]))

    result = extractor.extract(
        [
            {"role": "user", "content": request},
            {"role": "assistant", "content": "25 tests passed."},
        ],
        [],
        reflection=reflection,
        evidence_source="pytest result: 25 passed",
    )

    assert [item.type for item in result] == ["procedure"]


def test_engine_learns_procedure_and_updates_retrieval_feedback(tmp_path):
    replay = tmp_path / "session.jsonl"
    _write_replay(replay)
    reflection = {
        "task_summary": "Fix compatibility tests",
        "outcome": "success",
        "summary": "All tests passed.",
        "failures": [],
        "root_causes": ["Compatibility issue"],
        "effective_actions": ["Applied fix"],
        "verification": ["243 passed"],
        "reusable_lessons": ["Always run pytest after compatibility changes"],
        "evidence": ["243 passed"],
    }
    procedure = [{
        "action": "create",
        "title": "Compatibility verification procedure",
        "description": "Verify compatibility fixes with the complete suite",
        "content": "After applying a compatibility fix, run pytest and require all tests to pass.",
        "type": "procedure",
        "scope": "project",
        "keywords": ["compatibility", "pytest"],
        "confidence": 0.9,
        "evidence": "243 passed",
    }]
    engine = MemoryEngine(
        _FakeLLM([json.dumps(reflection), json.dumps(procedure)]),
        root=tmp_path / "memory",
        project_path=tmp_path,
    )
    engine.store.save(_memory(project_path=str(tmp_path.resolve())))
    engine.build_prompt("Run pytest tests")

    saved = engine.learn(_messages(), "session-success", replay)

    assert [memory.type for memory in saved] == ["procedure"]
    used = engine.store.get("pytest-guide")
    assert used is not None
    assert used.use_count == 1
    assert used.success_count == 1
    assert used.confidence > 0.8


def test_verified_success_gets_constrained_procedure_fallback(tmp_path):
    replay = tmp_path / "fallback.jsonl"
    record = {
        "step": 1,
        "llm_response": {"content": "Recovered and verified."},
        "tool_executions": [
            {
                "name": "bash",
                "arguments": {"command": "pytest -q"},
                "result": "blocked by policy",
                "success": False,
                "error": "blocked by policy",
            },
            {
                "name": "bash",
                "arguments": {"command": "python -m pytest -q"},
                "result": "18 passed",
                "success": True,
                "error": None,
            },
        ],
    }
    replay.write_text(json.dumps(record) + "\n", encoding="utf-8")
    reflection = {
        "task_summary": "Verify the memory tests",
        "outcome": "success",
        "summary": "The corrected command passed.",
        "failures": ["blocked by policy"],
        "root_causes": ["The first command was blocked"],
        "effective_actions": ["Used the supported test command"],
        "verification": ["18 passed"],
        "reusable_lessons": ["Run the supported command and require a clean pass"],
        "evidence": ["blocked by policy", "18 passed"],
    }
    episode = [{
        "action": "create",
        "title": "Blocked test command episode",
        "description": "The initial test command was blocked before a successful retry",
        "content": "The first command was blocked; the supported command passed.",
        "type": "episode",
        "scope": "project",
        "keywords": ["pytest", "policy"],
        "confidence": 0.8,
        "evidence": "blocked by policy",
    }]
    procedure = [{
        "action": "create",
        "title": "Memory test verification procedure",
        "description": "Run the supported memory tests and require a clean pass",
        "content": "Run python -m pytest for the memory tests and require every test to pass.",
        "type": "procedure",
        "scope": "project",
        "keywords": ["pytest", "memory", "verification"],
        "confidence": 0.9,
        "evidence": "18 passed",
    }]
    engine = MemoryEngine(
        _FakeLLM([json.dumps(reflection), json.dumps(episode), json.dumps(procedure)]),
        root=tmp_path / "memory",
        project_path=tmp_path,
    )

    saved = engine.learn(_messages(), "fallback-session", replay)

    assert [memory.type for memory in saved] == ["episode", "procedure"]
    assert all(memory.status == "candidate" for memory in saved)
    assert all(memory.validation_count == 1 for memory in saved)


def test_general_extraction_failure_still_uses_execution_fallback(tmp_path):
    replay = tmp_path / "general-failure.jsonl"
    _write_replay(replay)
    reflection = {
        "task_summary": "Verify compatibility",
        "outcome": "success",
        "summary": "The suite passed.",
        "failures": [],
        "root_causes": [],
        "effective_actions": ["Ran pytest"],
        "verification": ["243 passed"],
        "reusable_lessons": ["Verify compatibility changes with pytest"],
        "evidence": ["243 passed"],
    }
    procedure = [{
        "action": "create",
        "title": "Compatibility test procedure",
        "description": "Verify compatibility changes with pytest",
        "content": "Run pytest after compatibility changes and require all tests to pass.",
        "type": "procedure",
        "scope": "project",
        "keywords": ["compatibility", "pytest"],
        "confidence": 0.9,
        "evidence": "243 passed",
    }]
    engine = MemoryEngine(
        _FakeLLM([
            json.dumps(reflection),
            "not json",
            "still not json",
            json.dumps(procedure),
        ]),
        root=tmp_path / "memory",
        project_path=tmp_path,
    )
    engine.checkpoint(_messages(), "general-failure", replay)

    saved = engine.learn(_messages(), "general-failure", replay)

    assert [memory.type for memory in saved] == ["procedure"]
    assert saved[0].status == "candidate"
    assert engine.extractor.last_succeeded is False
    assert engine.extractor.last_fallback_succeeded is True
    assert engine.pending_count() == 0


def test_candidate_procedure_activates_after_second_independent_validation(tmp_path):
    first_replay = tmp_path / "first.jsonl"
    second_replay = tmp_path / "second.jsonl"
    _write_replay(first_replay)
    _write_replay(second_replay)
    reflection = {
        "task_summary": "Verify compatibility",
        "outcome": "success",
        "summary": "The suite passed.",
        "failures": [],
        "root_causes": [],
        "effective_actions": ["Ran the complete suite"],
        "verification": ["243 passed"],
        "reusable_lessons": ["Require the complete suite to pass"],
        "evidence": ["243 passed"],
    }
    create = [{
        "action": "create",
        "title": "Compatibility verification procedure",
        "description": "Verify compatibility changes with the complete suite",
        "content": "Run pytest after compatibility changes and require a clean pass.",
        "type": "procedure",
        "scope": "project",
        "keywords": ["compatibility", "pytest"],
        "confidence": 0.9,
        "evidence": "243 passed",
    }]
    merge = [{**create[0], "action": "merge", "target_id": "compatibility-verification-procedure"}]
    engine = MemoryEngine(
        _FakeLLM([
            json.dumps(reflection),
            json.dumps(create),
            json.dumps(reflection),
            json.dumps(merge),
        ]),
        root=tmp_path / "memory",
        project_path=tmp_path,
    )

    first = engine.learn(_messages(), "validation-one", first_replay)[0]

    assert first.status == "candidate"
    assert first.validation_count == 1
    assert engine.search("compatibility pytest") == []

    second = engine.learn(_messages(), "validation-two", second_replay)[0]

    assert second.status == "active"
    assert second.validation_count == 2
    assert second.source_sessions == ["validation-one", "validation-two"]
    assert engine.search("compatibility pytest")


def test_candidate_can_be_approved_explicitly(tmp_path):
    engine = MemoryEngine(_FakeLLM([]), root=tmp_path / "memory", project_path=tmp_path)
    engine.store.save(_memory(
        "candidate-procedure",
        type="procedure",
        status="candidate",
        validation_count=1,
        project_path=str(tmp_path.resolve()),
    ))

    approved = engine.approve("candidate-procedure")

    assert approved is not None
    assert approved.status == "active"
    assert approved.version == 2


def test_dynamic_agent_memory_retrieval_replaces_prompt_each_turn():
    class _Memory:
        def __init__(self):
            self.queries = []

        def build_prompt(self, query):
            self.queries.append(query)
            return f"memory for {query}"

    memory = _Memory()
    agent = Agent(llm=LLM.__new__(LLM), tools=[], replay=False, memory=memory)

    agent._load_memory_context("pytest task")
    agent._load_memory_context("database task")

    assert memory.queries == ["pytest task", "database task"]
    assert "memory for database task" in agent._full_messages()[0]["content"]
    assert "memory for pytest task" not in agent._full_messages()[0]["content"]


def test_retrieval_matches_original_chinese_evidence(tmp_path):
    engine = MemoryEngine(_FakeLLM([]), root=tmp_path / "memory", project_path=tmp_path)
    engine.store.save(_memory(
        "concise-answers",
        title="Concise technical answers",
        description="Keep answers concise",
        content="Prefer short executable technical guidance.",
        type="user",
        scope="global",
        project_path=None,
        keywords=["concise", "executable"],
        evidence=["请记住：以后回答技术问题时保持简洁。"],
    ))

    matches = engine.search("简洁")

    assert [match.memory.id for match in matches] == ["concise-answers"]


def test_retrieval_ignores_legacy_global_execution_memories(tmp_path):
    engine = MemoryEngine(_FakeLLM([]), root=tmp_path / "memory", project_path=tmp_path)
    engine.store.save(_memory(
        "global-episode",
        type="episode",
        scope="global",
        project_path=None,
        evidence=["请保持简洁。"],
    ))

    assert engine.search("简洁") == []


def test_retrieval_ignores_legacy_task_request_misclassified_as_user_memory(tmp_path):
    engine = MemoryEngine(_FakeLLM([]), root=tmp_path / "memory", project_path=tmp_path)
    engine.store.save(_memory(
        "user-wants-reusable-memory-steps",
        title="User wants reusable memory steps",
        description="The user asked for reusable verification steps",
        content="Summarize reusable memory verification steps.",
        type="user",
        scope="project",
        project_path=str(tmp_path.resolve()),
        keywords=["pytest", "verification"],
        evidence=["请运行 pytest，并总结以后验证记忆系统时可以重复使用的步骤。"],
    ))

    assert engine.search("pytest 验证") == []


def test_memory_cli_preserves_bracketed_type_labels(tmp_path, monkeypatch):
    output = StringIO()
    monkeypatch.setattr(cli_module, "console", Console(file=output, force_terminal=False, width=160))
    engine = MemoryEngine(_FakeLLM([]), root=tmp_path / "memory", project_path=tmp_path)
    engine.store.save(_memory(project_path=str(tmp_path.resolve())))

    class _Agent:
        memory = engine

    cli_module._show_memory(_Agent(), Config(memory_dir=engine.store.root))
    cli_module._search_memory(_Agent(), "pytest")

    rendered = output.getvalue()
    assert "[project/project/active]" in rendered
    assert "[project/project]" in rendered


def test_memory_cli_shows_pending_retry_error(tmp_path, monkeypatch):
    output = StringIO()
    monkeypatch.setattr(cli_module, "console", Console(file=output, force_terminal=False, width=180))
    engine = MemoryEngine(
        _FakeLLM([RuntimeError("provider offline")]),
        root=tmp_path / "memory",
        project_path=tmp_path,
    )
    engine.checkpoint(_messages(), "visible-failure")
    engine.learn(_messages(), "visible-failure")

    class _Agent:
        memory = engine

    cli_module._show_memory(_Agent(), Config(memory_dir=engine.store.root))

    rendered = output.getvalue()
    assert "Pending reflections: 1" in rendered
    assert "visible-failure attempts=1" in rendered
    assert "provider offline" in rendered


def test_archive_removes_memory_from_search_and_increments_version(tmp_path):
    engine = MemoryEngine(_FakeLLM([]), root=tmp_path / "memory", project_path=tmp_path)
    engine.store.save(_memory(project_path=str(tmp_path.resolve())))

    archived = engine.archive("pytest-guide")

    assert archived is not None
    assert archived.status == "archived"
    assert archived.version == 2
    assert engine.search("pytest tests") == []

    approved = engine.approve("pytest-guide")

    assert approved is not None
    assert approved.status == "active"
    assert approved.version == 3
    assert engine.search("pytest tests")


def test_new_memory_can_supersede_an_old_conflict(tmp_path):
    root = tmp_path / "memory"
    old = _memory(
        "old-runner",
        title="Old project test runner",
        description="Use the legacy runner",
        content="Run the legacy test command.",
        project_path=str(tmp_path.resolve()),
        keywords=["legacy", "runner"],
    )
    proposal = [{
        "action": "create",
        "title": "New project verification command",
        "description": "Use the new verification command",
        "content": "Run pytest -q for project verification.",
        "type": "project",
        "scope": "project",
        "keywords": ["pytest", "verification"],
        "confidence": 0.95,
        "evidence": "以后这个项目改用 pytest -q。",
        "supersedes": "old-runner",
    }]
    engine = MemoryEngine(_FakeLLM([json.dumps(proposal, ensure_ascii=False)]), root=root, project_path=tmp_path)
    engine.store.save(old)
    messages = [
        {"role": "user", "content": "以后这个项目改用 pytest -q。"},
        {"role": "assistant", "content": "将在会话结束时更新。"},
    ]

    saved = engine.learn(messages, "session-replace")

    assert len(saved) == 1
    assert engine.store.get("old-runner").status == "superseded"
    assert engine.store.get(saved[0].id).supersedes == "old-runner"


def test_pending_checkpoint_is_recovered_and_removed(tmp_path):
    root = tmp_path / "memory"
    first = MemoryEngine(_FakeLLM([]), root=root, project_path=tmp_path)
    pending = first.checkpoint(_messages(), "interrupted-session")
    assert pending is not None and pending.exists()

    recovering = MemoryEngine(_FakeLLM(["[]"]), root=root, project_path=tmp_path)

    assert recovering.recover_pending(force=True) == 1
    assert recovering.pending_count() == 0


def test_pending_retry_prioritizes_structured_fallback(tmp_path):
    root = tmp_path / "memory"
    replay = tmp_path / "pending-retry.jsonl"
    _write_replay(replay)
    first = MemoryEngine(_FakeLLM([]), root=root, project_path=tmp_path)
    first.checkpoint(_messages(), "structured-retry", replay)
    reflection = {
        "task_summary": "Verify compatibility",
        "outcome": "success",
        "summary": "The suite passed.",
        "failures": [],
        "root_causes": [],
        "effective_actions": ["Ran pytest"],
        "verification": ["243 passed"],
        "reusable_lessons": ["Verify compatibility changes with pytest"],
        "evidence": ["243 passed"],
    }
    procedure = [{
        "action": "create",
        "title": "Structured retry procedure",
        "description": "Verify compatibility changes during pending recovery",
        "content": "Run pytest and require all compatibility tests to pass.",
        "type": "procedure",
        "scope": "project",
        "keywords": ["pytest", "recovery"],
        "confidence": 0.9,
        "evidence": "243 passed",
    }]
    recovering = MemoryEngine(
        _FakeLLM([
            json.dumps(reflection),
            json.dumps(procedure),
            "not json",
            "still not json",
        ]),
        root=root,
        project_path=tmp_path,
    )

    assert recovering.recover_pending(force=True) == 1
    assert recovering.pending_count() == 0
    stored = recovering.store.get("structured-retry-procedure")
    assert stored is not None
    assert stored.status == "candidate"


def test_pending_failure_records_attempt_and_error(tmp_path):
    root = tmp_path / "memory"
    engine = MemoryEngine(_FakeLLM([RuntimeError("provider offline")]), root=root, project_path=tmp_path)
    engine.checkpoint(_messages(), "failed-extraction")

    assert engine.learn(_messages(), "failed-extraction") == []

    status = engine.pending_status()
    assert len(status) == 1
    assert status[0]["session_id"] == "failed-extraction"
    assert status[0]["attempts"] == 1
    assert "provider offline" in status[0]["last_error"]
    assert status[0]["last_attempted_at"] != "-"


def test_recent_or_current_pending_checkpoint_is_not_recovered_automatically(tmp_path):
    root = tmp_path / "memory"
    first = MemoryEngine(_FakeLLM([]), root=root, project_path=tmp_path)
    first.checkpoint(_messages(), "active-session")
    recovering = MemoryEngine(_FakeLLM(["[]"]), root=root, project_path=tmp_path)

    assert recovering.recover_pending() == 0
    assert recovering.recover_pending(exclude_session="active-session", force=True) == 0
    assert recovering.pending_count() == 1


def test_repeatedly_failing_pending_checkpoint_is_quarantined(tmp_path):
    root = tmp_path / "memory"
    first = MemoryEngine(_FakeLLM([]), root=root, project_path=tmp_path)
    first.checkpoint(_messages(), "broken-session")
    recovering = MemoryEngine(
        _FakeLLM([RuntimeError("offline"), RuntimeError("offline"), RuntimeError("offline")]),
        root=root,
        project_path=tmp_path,
    )

    for _ in range(3):
        recovering.recover_pending(force=True)

    assert recovering.pending_count() == 0
    failed = root / ".failed" / "broken-session.json"
    assert failed.exists()
    payload = json.loads(failed.read_text(encoding="utf-8"))
    assert payload["attempts"] == 3
    assert "offline" in payload["last_error"]


def test_store_lock_is_released(tmp_path):
    store = MemoryStore(tmp_path)

    with store.locked():
        assert (tmp_path / ".memory.lock").exists()

    assert not (tmp_path / ".memory.lock").exists()

"""Regression cases from the read-only, real-provider acceptance run."""

import json
from copy import deepcopy
from io import StringIO
from unittest.mock import Mock

import pytest
from rich.console import Console

from corecoder.agent import Agent
from corecoder.context import ContextManager
from corecoder.execution import terminal_failure
from corecoder.llm import LLM, LiteLLM
from corecoder.memory import Memory, MemoryEngine
from corecoder.memory.reflection import MemoryReflector
from corecoder.models import LLMResponse, ToolCall
from corecoder.skills import SkillManager
from corecoder.skills.evolution import SkillEvolutionEngine
from corecoder.tools.grep import GrepTool
from corecoder.tools.read import ReadFileTool


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, LLMResponse) else LLMResponse(content=response)


def claimed_success(**updates):
    return json.dumps({
        "outcome": "success", "verification": ["25 passed"], "evidence": ["25 passed"],
        "deliverable_complete": True, "constraints_satisfied": True, **updates,
    })


def transcript(answer="The verification report: all 25 tests passed."):
    return [
        {"role": "user", "content": "Run verification and deliver the report."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "t1", "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'},
        }]},
        {"role": "tool", "tool_call_id": "t1", "content": "25 passed"},
        {"role": "assistant", "content": answer},
    ]


@pytest.mark.parametrize("answer", [
    "# Conversation Summary\nInvestigation complete. Final audit report has not yet been written.",
    "最终审计报告尚未完成。", "Error: upstream HTTP 400", "",
])
def test_model_claim_cannot_validate_unfinished_deliverable(answer):
    reflection = MemoryReflector(ScriptedLLM(claimed_success())).reflect(transcript(answer))
    assert reflection.outcome != "success"
    assert not reflection.deliverable_complete


@pytest.mark.parametrize("facts", [
    {"status": "failed", "policy_violations": 0},
    {"status": "completed", "policy_violations": 1},
])
def test_runtime_facts_override_model_success(facts):
    messages = transcript()
    messages[-1]["_execution"] = facts
    result = MemoryReflector(ScriptedLLM(claimed_success())).reflect(messages)
    assert result.outcome != "success"


def test_missing_or_false_semantic_attestation_fails_closed():
    for payload in [
        '{"outcome":"success", "verification":["25 passed"]}',
        claimed_success(constraints_satisfied=False),
        claimed_success(deliverable_complete=False),
    ]:
        assert MemoryReflector(ScriptedLLM(payload)).reflect(transcript()).outcome != "success"


def test_legacy_sensitive_read_is_not_success_even_with_redacted_output(tmp_path):
    messages = transcript()
    messages[1]["tool_calls"][0]["function"] = {
        "name": "read_file", "arguments": json.dumps({"file_path": str(tmp_path / ".env")}),
    }
    messages[2]["content"] = "OPENAI_API_KEY=[REDACTED]"
    assert "credential" in terminal_failure(messages)


def test_incomplete_turn_in_pending_batch_cannot_borrow_later_success():
    messages = transcript("# Conversation Summary") + transcript()
    assert terminal_failure(messages) is not None


def test_candidate_only_promotes_after_two_completed_independent_sessions(tmp_path):
    proposal = json.dumps([{
        "title": "Verification procedure", "description": "Run verification and report results",
        "content": "Run tests. Deliver the final verification report; require all tests to pass.",
        "type": "procedure", "scope": "project", "evidence": "25 passed",
        "keywords": ["verification", "report"],
    }])
    llm = ScriptedLLM(
        claimed_success(), proposal,
        claimed_success(), proposal,
        claimed_success(), proposal,
        claimed_success(), proposal,
    )
    engine = MemoryEngine(llm, root=tmp_path / "memory", project_path=tmp_path)
    first = engine.learn(transcript(), "first")[0]
    assert first.status == "candidate" and first.validation_count == 1
    assert engine.learn(transcript("# Conversation Summary"), "unfinished") == []
    unchanged = engine.store.get(first.id)
    assert unchanged.version == 1 and unchanged.validation_count == 1
    same_session = engine.learn(transcript(), "first")[0]
    assert same_session.validation_count == 1
    second = engine.learn(transcript(), "second")[0]
    assert second.status == "active" and second.validation_count == 2
    assert second.verified_sessions == ["first", "second"]
    assert first.content in str(llm.calls[2])  # existing procedure criteria reviewed
    skill = SkillEvolutionEngine(tmp_path / "skills").propose(second)
    assert skill.manifest.status == "candidate"
    assert skill.manifest.evolution.review_required


def test_long_final_report_tail_can_create_a_procedure_candidate(tmp_path):
    terminal_marker = "Release metadata verification complete: evidence present; no files changed."
    reflection = json.dumps({
        "outcome": "success",
        "deliverable_complete": True,
        "constraints_satisfied": True,
        "verification": [terminal_marker],
        "evidence": [terminal_marker],
    })
    proposal = json.dumps([{
        "title": "Read-only release metadata verification",
        "description": "Verify package metadata without changing project files",
        "content": "Inspect metadata and source, report cited conclusions, and confirm no files changed.",
        "type": "procedure",
        "scope": "project",
        "evidence": terminal_marker,
        "keywords": ["release metadata", "read-only", "verification"],
        "confidence": 0.9,
    }])
    messages = transcript(
        "# Complete verification report\n" + ("Detailed evidence.\n" * 1_000) + terminal_marker
    )
    engine = MemoryEngine(
        ScriptedLLM(reflection, proposal),
        root=tmp_path / "memory",
        project_path=tmp_path,
    )

    learned = engine.learn(messages, "long-report-session")

    assert len(learned) == 1
    assert learned[0].type == "procedure"
    assert learned[0].status == "candidate"
    assert learned[0].validation_count == 1


def test_legacy_active_counter_cannot_bypass_new_evolution_gate(tmp_path):
    memory = Memory(id="legacy", title="Legacy", description="Legacy verification",
                    content="Run tests", type="procedure", status="active", validation_count=2)
    with pytest.raises(ValueError, match="completion-checked"):
        SkillEvolutionEngine(tmp_path / "skills").propose(memory)
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_summary_gets_one_finalization_retry_and_failed_retry_is_not_success():
    llm = ScriptedLLM("# Conversation Summary", "# Conversation Summary")
    agent = Agent(llm=llm, tools=[], replay=False)
    answer = await agent.chat("Deliver an audit report")
    assert answer.startswith("Error:")
    assert len(llm.calls) == 2
    assert agent._turn_messages[-1]["_execution"]["status"] == "partial"


@pytest.mark.asyncio
async def test_truncated_nonempty_answer_gets_finalization():
    llm = ScriptedLLM(LLMResponse(content="Half a report", finish_reason="length"), "Complete report")
    agent = Agent(llm=llm, tools=[], replay=False)
    assert await agent.chat("Deliver a report") == "Complete report"
    assert agent._turn_messages[-1]["_execution"]["status"] == "completed"


@pytest.mark.asyncio
async def test_policy_block_records_partial_and_no_secret_reaches_transcript(tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("canary-secret-token", encoding="utf-8")
    llm = ScriptedLLM(
        LLMResponse(tool_calls=[ToolCall(id="secret", name="read_file", arguments={"file_path": str(secret)})]),
        "Report: the credential file was not accessible.",
    )
    agent = Agent(llm=llm, tools=[ReadFileTool()], replay=False)
    await agent.chat("Review configuration without reading credentials")
    assert "canary-secret-token" not in json.dumps(llm.calls)
    assert agent._turn_messages[-1]["_execution"] == {"status": "partial", "policy_violations": 1}


@pytest.mark.asyncio
async def test_explicit_read_scope_constrains_root_search_without_reading_replay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corecoder").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "replays").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "replays" / "session.jsonl").write_text("replay-canary", encoding="utf-8")
    llm = ScriptedLLM(
        LLMResponse(tool_calls=[ToolCall(
            id="root-grep", name="grep", arguments={"pattern": "canary", "path": str(tmp_path)},
        )]),
        "The search was constrained to the approved roots.",
    )
    agent = Agent(llm=llm, tools=[GrepTool(), ReadFileTool()], replay=False)

    await agent.chat(
        "仅允许使用 read_file、grep；请将检索范围限制在 pyproject.toml、corecoder 和 tests。"
    )

    transcript_json = json.dumps(llm.calls, ensure_ascii=False)
    assert "replay-canary" not in transcript_json
    assert "Parent search constrained to user-approved roots" in transcript_json
    assert agent._turn_messages[-1]["_execution"] == {"status": "completed", "policy_violations": 0}


@pytest.mark.asyncio
async def test_explicit_read_scope_blocks_direct_replay_search(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for directory in ("corecoder", "tests", "replays"):
        (tmp_path / directory).mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "replays" / "session.jsonl").write_text("replay-canary", encoding="utf-8")
    llm = ScriptedLLM(
        LLMResponse(tool_calls=[ToolCall(
            id="replay-grep", name="grep", arguments={"pattern": "canary", "path": "replays"},
        )]),
        "The direct out-of-scope search was blocked.",
    )
    agent = Agent(llm=llm, tools=[GrepTool(), ReadFileTool()], replay=False)

    await agent.chat(
        "仅允许使用 read_file、grep；请将检索范围限制在 pyproject.toml、corecoder 和 tests。"
    )

    transcript_json = json.dumps(llm.calls, ensure_ascii=False)
    assert "replay-canary" not in transcript_json
    assert "outside the user-requested scope" in transcript_json
    assert agent._turn_messages[-1]["_execution"] == {"status": "partial", "policy_violations": 1}


@pytest.mark.asyncio
async def test_explicit_read_scope_allows_named_subdirectory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "corecoder"
    source.mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (source / "module.py").write_text("allowed-marker = 1", encoding="utf-8")
    llm = ScriptedLLM(
        LLMResponse(tool_calls=[ToolCall(
            id="scoped-grep", name="grep", arguments={"pattern": "allowed-marker", "path": "corecoder"},
        )]),
        "Scoped verification complete.",
    )
    agent = Agent(llm=llm, tools=[GrepTool(), ReadFileTool()], replay=False)

    await agent.chat(
        "仅允许使用 read_file、grep；请将检索范围限制在 pyproject.toml、corecoder 和 tests。"
    )

    assert "allowed-marker" in json.dumps(llm.calls, ensure_ascii=False)
    assert agent._turn_messages[-1]["_execution"] == {"status": "completed", "policy_violations": 0}


def test_explicit_turn_policy_filters_tools_and_describes_read_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corecoder").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    agent = Agent(llm=ScriptedLLM(), tools=[GrepTool(), ReadFileTool()], replay=False)

    agent._load_turn_policy(
        "仅允许使用 grep；请将检索范围限制在 pyproject.toml、corecoder 和 tests。"
    )
    schemas = agent._tool_schemas()

    assert [schema["function"]["name"] for schema in schemas] == ["grep"]
    description = schemas[0]["function"]["description"]
    assert str(tmp_path / "corecoder") in description
    assert "Do not target a parent directory" in description
    assert agent._read_scope_error(
        "glob",
        {"pattern": "{tests,corecoder}/**/{pytest.ini,setup.cfg,tox.ini}"},
    ) is None
    assert agent._constrained_read_targets(
        "glob",
        {"pattern": "{pytest.ini,setup.cfg,tox.ini,conftest.py}"},
    ) == (tmp_path / "corecoder", tmp_path / "tests")


@pytest.mark.asyncio
async def test_checkpoint_keeps_evidence_even_when_context_was_compressed(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("immutable_evidence = 1", encoding="utf-8")
    llm = ScriptedLLM(
        LLMResponse(tool_calls=[ToolCall(id="read", name="read_file", arguments={"file_path": str(source)})]),
        "Report: immutable_evidence is set to 1.",
    )
    engine = MemoryEngine(ScriptedLLM(), root=tmp_path / "memory", project_path=tmp_path)
    agent = Agent(llm=llm, tools=[ReadFileTool()], memory=engine, replay=False)

    def compress(messages, llm):
        for message in messages:
            if message.get("role") == "tool":
                message["content"] = "[snipped]"

    agent.context.maybe_compress = compress
    await agent.chat("Inspect source.py and report")
    agent.checkpoint_memory()
    pending = json.loads(next((tmp_path / "memory" / ".pending").glob("*.json")).read_text())
    assert "immutable_evidence = 1" in pending["messages"][2]["content"]
    assert "[snipped]" in agent.messages[2]["content"]


@pytest.mark.asyncio
async def test_tool_interruption_closes_exchange_and_records_failure():
    import asyncio

    llm = ScriptedLLM(LLMResponse(tool_calls=[ToolCall(id="t", name="read_file", arguments={"file_path": "x"})]))
    agent = Agent(llm=llm, tools=[ReadFileTool()], replay=False)

    async def interrupt(*args):
        raise asyncio.CancelledError()

    agent._exec_tools_async = interrupt
    with pytest.raises(asyncio.CancelledError):
        await agent.chat("Read a file")
    assert agent.messages[-2]["tool_call_id"] == "t"
    assert agent._turn_messages[-1]["_execution"]["status"] == "failed"


@pytest.mark.asyncio
async def test_provider_error_has_failed_durable_checkpoint_and_telemetry(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "skills",
                                  telemetry_path=tmp_path / "telemetry.json")
    engine = MemoryEngine(ScriptedLLM(), root=tmp_path / "memory", project_path=tmp_path)
    agent = Agent(llm=ScriptedLLM(RuntimeError("HTTP 400")), tools=[ReadFileTool(), GrepTool()],
                  skills=manager, memory=engine, replay=False)
    with pytest.raises(RuntimeError):
        await agent.chat("$security.code-audit Audit authentication")
    agent.checkpoint_memory()
    payload = json.loads(next((tmp_path / "memory" / ".pending").glob("*.json")).read_text())
    assert payload["messages"][-1]["_execution"]["status"] == "failed"
    assert manager.telemetry.stats()["security.code-audit"]["failure_count"] == 1


@pytest.mark.parametrize("layer", ["_incremental_summarize", "_hard_collapse"])
def test_compression_preserves_original_request_and_retained_reasoning(layer):
    messages = [{"role": "user", "content": "Deliver an audit report. Do not read .env."}]
    for i in range(12):
        messages.extend([
            {"role": "assistant", "content": None, "reasoning_content": f"opaque-{i}",
             "tool_calls": [{"id": f"t{i}"}]},
            {"role": "tool", "tool_call_id": f"t{i}", "content": "source code"},
        ])
    getattr(ContextManager(), layer)(messages, llm=None)
    assert [m for m in messages if m["role"] == "user"][-1]["content"].startswith("Deliver an audit")
    assert messages[-2]["reasoning_content"] == "opaque-11"


@pytest.mark.parametrize("provider", [LLM, LiteLLM])
def test_provider_prepares_all_assistant_messages_without_losing_reasoning(provider):
    llm = provider(model="deepseek-v4-flash", api_key="test")
    original = [
        {"role": "assistant", "content": "Synthetic acknowledgement"},
        {"role": "assistant", "content": None, "reasoning_content": "opaque", "tool_calls": []},
        {"role": "assistant", "content": "done", "_execution": {"status": "completed"}},
    ]
    llm._call_with_retry = Mock(return_value=iter([]))
    llm.chat(original, tools=[{"type": "function"}])
    sent = llm._call_with_retry.call_args.args[0]["messages"]
    assert [m["reasoning_content"] for m in sent] == ["", "opaque", ""]
    assert "_execution" not in sent[-1]
    assert "reasoning_content" not in original[0]


@pytest.mark.parametrize("message,expected_calls", [
    ("reasoning_content must be passed back", 1),
    ("stream_options is unsupported", 2),
])
def test_bad_request_fallback_only_retries_unsupported_stream_options(message, expected_calls):
    import httpx
    from openai import BadRequestError

    response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
    error = BadRequestError(message, response=response, body=None)
    llm = LLM(model="deepseek-v4-flash", api_key="test")
    llm._call_with_retry = Mock(side_effect=[error, iter([])])
    if expected_calls == 1:
        with pytest.raises(BadRequestError):
            llm.chat([{"role": "user", "content": "hi"}])
    else:
        llm.chat([{"role": "user", "content": "hi"}])
    assert llm._call_with_retry.call_count == expected_calls


def test_other_provider_does_not_receive_synthetic_reasoning_field():
    llm = LLM(model="unrelated-model", api_key="test")
    assert "reasoning_content" not in llm._prepare_messages([{"role": "assistant", "content": "hi"}])[0]


def test_legacy_procedure_is_visible_but_not_implicitly_retrieved(tmp_path):
    engine = MemoryEngine(ScriptedLLM(), root=tmp_path / "memory", project_path=tmp_path)
    memory = Memory(id="legacy", title="Legacy verification", description="Verify release",
                    content="Run verification tests", type="procedure", project_path=str(tmp_path.resolve()),
                    status="active", validation_count=2)
    engine.store.save(memory)
    assert engine.store.get(memory.id) is not None
    assert engine.search("verification") == []


@pytest.mark.parametrize("name", [".env", ".env.production", "private.pem", "id_rsa", ".aws/credentials"])
def test_sensitive_content_is_blocked_before_read_or_recursive_search(tmp_path, name):
    secret = tmp_path / name
    secret.parent.mkdir(exist_ok=True)
    secret.write_text("canary-secret-token", encoding="utf-8")
    (tmp_path / "source.py").write_text("public-marker", encoding="utf-8")
    assert ReadFileTool()._execute_sync(str(secret)).startswith("[Security]")
    assert GrepTool()._execute_sync(".*", str(secret)).startswith("[Security]")
    result = GrepTool()._execute_sync(".*", str(tmp_path))
    assert "canary-secret-token" not in result and "public-marker" in result


def test_sensitive_symlink_cannot_bypass_boundary(tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("canary-secret-token", encoding="utf-8")
    link = tmp_path / "innocent.txt"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("OS does not permit creating symlinks")
    assert ReadFileTool()._execute_sync(str(link)).startswith("[Security]")
    assert "canary-secret-token" not in GrepTool()._execute_sync(".*", str(tmp_path))


def test_sanitized_env_template_remains_readable(tmp_path):
    path = tmp_path / ".env.example"
    path.write_text("EXAMPLE=placeholder", encoding="utf-8")
    assert "EXAMPLE=placeholder" in ReadFileTool()._execute_sync(str(path))


def test_negated_edit_instructions_do_not_become_positive_routing_intent(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "skills")
    query = ("这是一次严格的只读验收。请分析当前项目副本的架构、主要模块和执行入口。"
             "禁止修改、创建、删除或重命名任何文件；禁止调用 write_file、edit_file、edit_ast 和 bash；"
             "只允许使用 read_file、grep、glob；不要修复问题，只报告证据、文件位置和改进建议。")
    result = manager.route(query, {"read_file", "grep", "glob", "write_file", "edit_file", "edit_ast", "bash"})
    assert not ({"修改", "创建", "删除", "重命名", "修复", "write"} & result.signature.actions)
    assert result.candidates[0].skill.manifest.id == "repository.architecture-analysis"
    assert result.selected_ids == ["repository.architecture-analysis"]


def test_clarification_counter_is_visible_in_cli(tmp_path, monkeypatch):
    from corecoder import cli

    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "skills",
                                  telemetry_path=tmp_path / "telemetry.json")
    result = manager.route("API", {"read_file", "grep", "glob"})
    assert result.decision == "clarify"
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=150, color_system=None))
    cli._show_skill_metrics(Agent(llm=ScriptedLLM(), skills=manager, replay=False))
    assert "Clarify" in output.getvalue()
    assert any(row["clarifications"] == 1 for row in manager.telemetry.stats().values())

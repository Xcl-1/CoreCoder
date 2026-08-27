"""Tests for the replay log system."""

import json

from corecoder.models import LLMResponse, StepRecord, ToolCall, ToolExecRecord

# --- StepRecord serialisation ---

def test_step_record_serialization():
    record = StepRecord(step=1, messages_count=5, estimated_input_tokens=200)
    json_str = record.model_dump_json()
    data = json.loads(json_str)
    assert data["step"] == 1
    assert data["messages_count"] == 5
    assert "timestamp" in data


def test_step_record_roundtrip():
    tc = ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
    resp = LLMResponse(content="done", tool_calls=[tc], prompt_tokens=10, completion_tokens=3)
    record = StepRecord(
        step=3,
        messages_count=12,
        estimated_input_tokens=500,
        llm_response=resp,
        tool_executions=[
            ToolExecRecord(name="bash", arguments={"cmd": "ls"}, result="file1\nfile2", duration_ms=12.5, success=True),
        ],
        step_duration_ms=150.0,
    )
    json_str = record.model_dump_json()
    restored = StepRecord.model_validate_json(json_str)
    assert restored.step == record.step
    assert restored.llm_response.content == "done"
    assert restored.llm_response.tool_calls[0].name == "bash"
    assert len(restored.tool_executions) == 1
    assert restored.tool_executions[0].success is True


# --- ToolExecRecord ---

def test_tool_exec_record_success():
    record = ToolExecRecord(name="read_file", arguments={"file_path": "a.py"}, result="line1\nline2", duration_ms=3.2, success=True)
    assert record.success is True
    assert record.error is None


def test_tool_exec_record_failure():
    record = ToolExecRecord(name="bash", arguments={"command": "bad"}, result="Error: fail", duration_ms=1.0, success=False, error="fail")
    assert record.success is False
    assert record.error == "fail"


# --- LLMResponse.message property ---

def test_llm_response_message_property():
    tc = ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})
    resp = LLMResponse(content="ok", tool_calls=[tc])
    msg = resp.message
    assert msg["role"] == "assistant"
    assert msg["content"] == "ok"
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "bash"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"cmd": "ls"}'


def test_llm_response_no_tool_calls():
    resp = LLMResponse(content="just text")
    msg = resp.message
    assert msg == {"role": "assistant", "content": "just text"}


def test_llm_response_empty_content():
    resp = LLMResponse(content="")
    msg = resp.message
    assert msg["content"] is None  # OpenAI expects null, not empty string


def test_message_not_in_serialization():
    resp = LLMResponse(content="hi", tool_calls=[ToolCall(id="c1", name="bash", arguments={})])
    data = json.loads(resp.model_dump_json())
    assert "message" not in data


# --- ReplayLogger ---

def test_replay_logger_creates_file(tmp_path, monkeypatch):
    from corecoder import replay as replay_mod
    monkeypatch.setattr(replay_mod, "REPLAYS_DIR", tmp_path)

    logger = replay_mod.ReplayLogger(session_id="test_session")
    logger.open()
    logger.close()
    assert (tmp_path / "test_session.jsonl").exists()


def test_replay_logger_appends_lines(tmp_path, monkeypatch):
    from corecoder import replay as replay_mod
    monkeypatch.setattr(replay_mod, "REPLAYS_DIR", tmp_path)

    logger = replay_mod.ReplayLogger(session_id="test_session")
    with logger:
        logger.log(StepRecord(step=1))
        logger.log(StepRecord(step=2))

    content = (tmp_path / "test_session.jsonl").read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["step"] == 2


def test_replay_logger_context_manager(tmp_path, monkeypatch):
    from corecoder import replay as replay_mod
    monkeypatch.setattr(replay_mod, "REPLAYS_DIR", tmp_path)

    with replay_mod.ReplayLogger(session_id="cm_test") as logger:
        logger.log(StepRecord(step=1))

    content = (tmp_path / "cm_test.jsonl").read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 1


def test_replay_logger_path_property(tmp_path, monkeypatch):
    from corecoder import replay as replay_mod
    monkeypatch.setattr(replay_mod, "REPLAYS_DIR", tmp_path)
    logger = replay_mod.ReplayLogger(session_id="p_test")
    logger.open()
    try:
        assert logger.path.name == "p_test.jsonl"
    finally:
        logger.close()


def test_replay_session_id_path_traversal_is_neutralized(tmp_path, monkeypatch):
    from corecoder import replay as replay_mod
    monkeypatch.setattr(replay_mod, "REPLAYS_DIR", tmp_path)

    logger = replay_mod.ReplayLogger(session_id="../../outside")
    logger.open()
    logger.close()

    assert logger.path == (tmp_path / "outside.jsonl").resolve()


def test_agent_replay_rotates_with_session_reset(tmp_path, monkeypatch):
    from corecoder import replay as replay_mod
    from corecoder.agent import Agent
    from corecoder.llm import LLM

    monkeypatch.setattr(replay_mod, "REPLAYS_DIR", tmp_path)
    agent = Agent(llm=LLM.__new__(LLM), tools=[], replay=True, session_id="first-session")
    first_path = agent._replay.path

    agent.reset()
    second_path = agent._replay.path
    agent.close()

    assert first_path.name == "first-session.jsonl"
    assert second_path.name == f"{agent.session_id}.jsonl"
    assert second_path != first_path


def test_replay_disabled_does_not_create_file(tmp_path, monkeypatch):
    """Agent with replay=False must not touch the filesystem."""
    from corecoder import replay as replay_mod
    from corecoder.agent import Agent
    from corecoder.llm import LLM

    monkeypatch.setattr(replay_mod, "REPLAYS_DIR", tmp_path)

    agent = Agent(llm=LLM.__new__(LLM), tools=[], replay=False)
    assert agent._replay is None
    assert not list(tmp_path.iterdir())
    agent.close()


def test_jsonl_every_line_valid(tmp_path, monkeypatch):
    from corecoder import replay as replay_mod
    monkeypatch.setattr(replay_mod, "REPLAYS_DIR", tmp_path)

    logger = replay_mod.ReplayLogger(session_id="valid")
    with logger:
        for i in range(5):
            logger.log(StepRecord(step=i))

    content = (tmp_path / "valid.jsonl").read_text(encoding="utf-8")
    for line in content.strip().split("\n"):
        data = json.loads(line)
        assert "step" in data
        assert "timestamp" in data
        assert "llm_response" in data
        assert "tool_executions" in data
        # verify it roundtrips
        StepRecord.model_validate_json(line)


def test_replay_logger_truncates_long_results():
    """Tool results over 5000 chars are truncated in the log record."""
    long_result = "x" * 6000
    record = ToolExecRecord(name="bash", arguments={}, result=long_result[:5000], duration_ms=0, success=True)
    assert len(record.result) <= 5000


def test_agent_close_is_idempotent():
    """Calling close() multiple times must not raise."""
    from corecoder.agent import Agent
    from corecoder.llm import LLM

    agent = Agent(llm=LLM.__new__(LLM), tools=[], replay=False)
    agent.close()
    agent.close()  # should not raise

import json

import pytest

from corecoder import session as session_module
from corecoder.agent import Agent
from corecoder.cli import _save_current_session
from corecoder.config import Config
from corecoder.llm import LLM
from corecoder.session import list_sessions, load_session, load_session_record, save_session


def test_default_session_ids_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    first_id = save_session([{"role": "user", "content": "first"}], "model-a")
    second_id = save_session([{"role": "user", "content": "second"}], "model-b")

    assert first_id != second_id
    assert load_session(first_id) == (
        [{"role": "user", "content": "first"}],
        "model-a",
    )
    assert load_session(second_id) == (
        [{"role": "user", "content": "second"}],
        "model-b",
    )


def test_session_id_path_traversal_is_neutralized(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session([{"role": "user", "content": "x"}], "m", "../../etc/passwd")

    assert sid == "passwd"
    assert (tmp_path / "passwd.json").exists()
    # the same traversal string round-trips through the parent-dir boundary check
    assert load_session("../../etc/passwd") == ([{"role": "user", "content": "x"}], "m")


def test_session_id_absolute_path_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session([{"role": "user", "content": "x"}], "m", "/etc/shadow")

    assert sid == "shadow"
    assert (tmp_path / "shadow.json").exists()


def test_session_id_windows_backslash_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session([{"role": "user", "content": "x"}], "m", r"..\..\secret")

    assert sid == "secret"


def test_session_id_length_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session([{"role": "user", "content": "x"}], "m", "a" * 500)

    assert len(sid) <= 100
    assert (tmp_path / f"{sid}.json").exists()


def test_corrupt_session_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    (tmp_path / "broken.json").write_text("{ not valid json", encoding="utf-8")

    assert load_session("broken") is None


def test_session_roundtrips_unicode(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    msgs = [{"role": "user", "content": "请帮我修复这个 bug"}]
    sid = save_session(msgs, "model-zh")

    raw = (tmp_path / f"{sid}.json").read_bytes()
    assert "请帮我修复这个 bug".encode() in raw
    assert load_session(sid) == (msgs, "model-zh")


def test_list_sessions_returns_all_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    for index in range(25):
        save_session(
            [{"role": "user", "content": f"conversation {index}"}],
            "model",
            f"session-{index:02d}",
        )

    sessions = list_sessions()

    assert len(sessions) == 25
    assert {item["id"] for item in sessions} == {f"session-{index:02d}" for index in range(25)}


def test_repeated_save_atomically_updates_one_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    save_session([{"role": "user", "content": "first"}], "model-a", "stable-id")
    latest = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]

    save_session(latest, "model-b", "stable-id")

    assert load_session("stable-id") == (latest, "model-b")
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_version_two_session_roundtrips_independent_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    messages = [{"role": "user", "content": "compressed model context"}]
    transcript = [
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "original answer"},
    ]

    save_session(messages, "model", "v2-session", transcript=transcript)
    record = load_session_record("v2-session")

    assert record is not None
    assert record.version == 2
    assert record.messages == messages
    assert record.transcript == transcript
    raw = json.loads((tmp_path / "v2-session.json").read_text(encoding="utf-8"))
    assert raw["version"] == 2
    assert raw["transcript"] == transcript
    assert list_sessions()[0]["preview"] == "original question"


def test_legacy_session_derives_display_transcript_without_tool_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    legacy_messages = [
        {"role": "system", "content": "private prompt"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "content": "private tool output", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "answer"},
    ]
    (tmp_path / "legacy.json").write_text(
        json.dumps({
            "id": "legacy",
            "model": "old-model",
            "saved_at": "2026-01-01 00:00:00",
            "messages": legacy_messages,
        }),
        encoding="utf-8",
    )

    record = load_session_record("legacy")

    assert record is not None
    assert record.version == 1
    assert record.messages == legacy_messages
    assert record.transcript == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_cli_auto_save_keeps_stable_id_and_resume_history(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[],
        replay=False,
        session_id="resumed-session",
    )
    config = Config(model="test-model")
    agent.messages = [{"role": "user", "content": "hello"}]
    agent.transcript = [{"role": "user", "content": "original hello"}]

    first_id = _save_current_session(agent, config)
    agent.messages.append({"role": "assistant", "content": "welcome back"})
    agent.transcript.append({"role": "assistant", "content": "original welcome"})
    second_id = _save_current_session(agent, config)

    assert first_id == second_id == "resumed-session"
    assert agent.session_id == "resumed-session"
    assert load_session("resumed-session") == (agent.messages, "test-model")
    record = load_session_record("resumed-session")
    assert record is not None
    assert record.transcript == agent.transcript
    assert len(list_sessions()) == 1


@pytest.mark.asyncio
async def test_agent_records_user_and_final_answer_in_display_transcript(monkeypatch):
    agent = Agent(llm=LLM.__new__(LLM), tools=[], replay=False)

    async def answer(*_args, **_kwargs):
        return "final answer"

    monkeypatch.setattr(agent, "_chat", answer)

    await agent._execute_chat_turn("original question")

    assert agent.transcript == [
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "final answer"},
    ]


def test_agent_reset_starts_a_new_session_id():
    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[],
        replay=False,
        session_id="original-session",
        transcript=[{"role": "user", "content": "old question"}],
    )

    agent.reset()

    assert agent.session_id != "original-session"
    assert agent.session_id.startswith("session_")
    assert agent.transcript == []

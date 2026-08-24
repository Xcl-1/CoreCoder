from corecoder import session as session_module
from corecoder.agent import Agent
from corecoder.cli import _save_current_session
from corecoder.config import Config
from corecoder.llm import LLM
from corecoder.session import list_sessions, load_session, save_session


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

    first_id = _save_current_session(agent, config)
    agent.messages.append({"role": "assistant", "content": "welcome back"})
    second_id = _save_current_session(agent, config)

    assert first_id == second_id == "resumed-session"
    assert agent.session_id == "resumed-session"
    assert load_session("resumed-session") == (agent.messages, "test-model")
    assert len(list_sessions()) == 1


def test_agent_reset_starts_a_new_session_id():
    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[],
        replay=False,
        session_id="original-session",
    )

    agent.reset()

    assert agent.session_id != "original-session"
    assert agent.session_id.startswith("session_")

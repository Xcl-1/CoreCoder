from io import StringIO
from types import SimpleNamespace

from rich.console import Console

import corecoder.cli as cli_module
from corecoder.config import Config
from corecoder.security import (
    AuditEntry,
    AuditLogger,
    ConfirmationContext,
    Guard,
    PermissionManager,
    PermissionRule,
)
from corecoder.tools import get_tool


def _captured_console(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(file=output, force_terminal=False, width=120),
    )
    return output


def test_show_history_renders_conversation_and_hides_tool_results(monkeypatch):
    output = _captured_console(monkeypatch)
    messages = [
        {"role": "system", "content": "private system prompt"},
        {"role": "user", "content": "[red]literal markup[/red]"},
        {"role": "assistant", "content": "# Answer\n\nDone."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"file_path": "src/app.py"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "SECRET_TOOL_OUTPUT"},
        {"role": "assistant", "content": "Finished."},
    ]

    cli_module._show_history(messages)

    rendered = output.getvalue()
    assert "Previous conversation" in rendered
    assert "literal markup" in rendered
    assert "Answer" in rendered
    assert "read_file(file_path='src/app.py')" in rendered
    assert "Finished." in rendered
    assert "1 tool result(s) hidden from history." in rendered
    assert "SECRET_TOOL_OUTPUT" not in rendered
    assert "private system prompt" not in rendered


def test_show_history_labels_compressed_context_and_skips_synthetic_ack(monkeypatch):
    output = _captured_console(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": "[Conversation summary — incremental]\nEdited corecoder/cli.py.",
        },
        {"role": "assistant", "content": "Understood. I have the full context."},
        {"role": "user", "content": "Continue."},
        {"role": "assistant", "content": "Continuing now."},
    ]

    cli_module._show_history(messages)

    rendered = output.getvalue()
    assert "Conversation summary" in rendered
    assert "Edited corecoder/cli.py." in rendered
    assert "Continue." in rendered
    assert "Continuing now." in rendered
    assert "Understood. I have the full context." not in rendered


def test_repl_renders_loaded_history_when_requested(monkeypatch):
    output = _captured_console(monkeypatch)
    shown = []

    class _Agent:
        def __init__(self):
            self._replay = None
            self.session_id = "resumed-session"
            self.messages = [{"role": "user", "content": "Earlier question"}]

        def close(self):
            pass

    def _end_prompt(*_args, **_kwargs):
        raise EOFError

    monkeypatch.setattr(cli_module, "pt_prompt", _end_prompt)
    monkeypatch.setattr(cli_module, "_save_current_session", lambda *_args: None)
    monkeypatch.setattr(cli_module, "_show_history", lambda messages: shown.append(messages))

    cli_module._repl(_Agent(), Config(model="test-model"), show_history=True)

    assert shown == [[{"role": "user", "content": "Earlier question"}]]
    assert "resumed-session" in output.getvalue()


def test_show_history_ignores_empty_history(monkeypatch):
    output = _captured_console(monkeypatch)

    cli_module._show_history([])

    assert output.getvalue() == ""


def test_help_renders_argument_placeholders_literally(monkeypatch):
    output = _captured_console(monkeypatch)

    cli_module._show_help()

    rendered = output.getvalue()
    assert "/permissions [user|session|project|builtin]" in rendered
    assert "/audit [filter] [n] [tool=<name>]" in rendered


def test_cli_undo_records_exact_runtime_state(monkeypatch):
    output = _captured_console(monkeypatch)
    events = []

    class _Changes:
        def __len__(self):
            return 1

        def undo_all(self, *, force=False):
            assert force is False
            return SimpleNamespace(
                restored=[r"D:\project\restored.py"],
                deleted=[r"D:\project\generated.py"],
                conflicts=[],
                errors=[],
            )

    agent = SimpleNamespace(changes=_Changes(), record_runtime_event=events.append)

    cli_module._undo_changes(agent)

    assert "1 restored" in output.getvalue()
    assert len(events) == 1
    assert '\"operation\": \"/undo\"' in events[0]
    assert "generated.py" in events[0]
    assert "deliberately deleted by the undo" in events[0]
    assert "environment reset" in events[0]


def test_cli_confirmation_shows_structured_security_context(monkeypatch):
    output = _captured_console(monkeypatch)
    fake_guard = SimpleNamespace(sanitize=lambda value: value)
    monkeypatch.setattr(cli_module._cli_confirm, "_guard", fake_guard, raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "a")
    context = ConfirmationContext(
        risk_level="high",
        capability_scope="process:execute",
        side_effect="dynamic",
        declared_risk="medium",
        network_action="ask",
        network_destinations=("api.example.com",),
        network_mutating=True,
        can_remember=False,
    )

    choice = cli_module._cli_confirm(
        "bash",
        {"command": "curl -X POST https://api.example.com/items"},
        "remote mutation needs approval",
        context,
    )

    rendered = output.getvalue()
    assert choice is False
    assert "process:execute" in rendered
    assert "dynamic" in rendered
    assert "high" in rendered
    assert "api.example.com" in rendered
    assert "remote mutation/upload" in rendered
    assert "cannot be remembered" in rendered
    assert "Session-wide approval is unavailable" in rendered


def test_show_audit_filters_and_tolerates_corrupt_record(tmp_path, monkeypatch):
    output = _captured_console(monkeypatch)
    logger = AuditLogger(tmp_path / "audit")
    logger.log(AuditEntry(
        timestamp="2026-09-07T18:00:00",
        tool_name="bash",
        arguments_summary="curl https://api.example.com",
        decision="deny",
        rule_source="network",
        reason="network confirmation denied",
        risk_level="high",
        network_policy_action="ask",
        network_destinations=["api.example.com"],
    ))
    path = next((tmp_path / "audit").glob("audit_*.jsonl"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write("not-json\n")
    agent = SimpleNamespace(guard=SimpleNamespace(audit=logger))

    cli_module._show_audit(agent, "deny 5 tool=bash")

    rendered = output.getvalue()
    assert "1 denied" in rendered
    assert "Skipped 1 unreadable audit line" in rendered
    assert "Recent Entries (deny, 1/1) tool=bash" in rendered
    assert "api.example.com" in rendered
    assert "high" in rendered
    assert "bash" in rendered
    assert "Human" in rendered


def test_permission_cli_filters_revokes_clears_and_audits(tmp_path, monkeypatch):
    output = _captured_console(monkeypatch)
    monkeypatch.setattr(
        "corecoder.security.permissions.USER_PERMISSIONS_PATH",
        tmp_path / "user-permissions.json",
    )
    monkeypatch.setattr(
        "corecoder.security.permissions.PROJECT_PERMISSIONS_PATH",
        tmp_path / "project-permissions.json",
    )
    manager = PermissionManager()
    user_rule = PermissionRule("bash", "user-command", "allow", "user test", 100)
    session_rule = PermissionRule("bash", "session-command", "allow", "session test", 100)
    manager.add_user_rule(user_rule)
    manager.add_session_rule(session_rule)
    guard = Guard(permissions=manager, audit=AuditLogger(tmp_path / "audit"))
    agent = SimpleNamespace(guard=guard)

    cli_module._show_permissions(agent, "user")
    cli_module._revoke_rule(agent, user_rule.rule_id)
    cli_module._clear_session_permissions(agent)

    rendered = output.getvalue()
    assert "Security Rules (user)" in rendered
    assert user_rule.rule_id in rendered
    assert "bash" in rendered
    assert "session-command" not in rendered
    assert f"Revoked user rule: {user_rule.rule_id}" in rendered
    assert "Cleared 1 session permission rule" in rendered
    assert manager.list_rules("user") == []
    assert manager.list_rules("session") == []
    policy_events = guard.audit.query(decisions={"policy"}, limit=10)
    assert policy_events.total_matches == 2


def test_security_explain_cli_previews_without_execution(tmp_path, monkeypatch):
    output = _captured_console(monkeypatch)
    monkeypatch.setattr(
        "corecoder.security.permissions.USER_PERMISSIONS_PATH",
        tmp_path / "user-permissions.json",
    )
    monkeypatch.setattr(
        "corecoder.security.permissions.PROJECT_PERMISSIONS_PATH",
        tmp_path / "project-permissions.json",
    )
    audit = AuditLogger(tmp_path / "audit")
    guard = Guard(permissions=PermissionManager(), audit=audit)
    agent = SimpleNamespace(guard=guard, _tool_by_name={"bash": get_tool("bash")})

    cli_module._explain_security(agent, "bash git push origin main")

    rendered = output.getvalue()
    assert "Security Policy Preview (nothing executed)" in rendered
    assert "CONFIRM" in rendered
    assert "process:execute" in rendered
    assert "high" in rendered
    assert not list((tmp_path / "audit").glob("audit_*.jsonl"))

from io import StringIO

from rich.console import Console

import corecoder.cli as cli_module
from corecoder.config import Config


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

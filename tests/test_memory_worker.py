"""Tests for non-blocking, incremental background memory learning."""

from __future__ import annotations

import json
import threading
import time

from corecoder.memory import MemoryEngine, MemoryWorker
from corecoder.models import LLMResponse


def _turn(user: str, assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


class _SequenceLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def chat(self, **_kwargs):
        return LLMResponse(content=self.responses.pop(0))


class _BlockingLLM(_SequenceLLM):
    def __init__(self, responses: list[str]):
        super().__init__(responses)
        self.started = threading.Event()
        self.release = threading.Event()

    def chat(self, **kwargs):
        self.started.set()
        assert self.release.wait(2.0)
        return super().chat(**kwargs)


def test_worker_preserves_a_new_turn_arriving_during_extraction(tmp_path):
    root = tmp_path / "memory"
    foreground = MemoryEngine(_SequenceLLM([]), root=root, project_path=tmp_path)
    first = _turn("Remember that I prefer concise replies.", "Understood.")
    second = _turn("I also prefer pytest.", "Understood.")
    foreground.checkpoint(first, "session-one")

    background_llm = _BlockingLLM(["[]", "[]"])
    background = MemoryEngine(background_llm, root=root, project_path=tmp_path)
    worker = MemoryWorker(background)
    worker.submit("session-one")
    assert background_llm.started.wait(1.0)

    # This rewrites the checkpoint with a new token while the old token is in
    # flight.  A successful old task may acknowledge only its own first turn.
    foreground.checkpoint([*first, *second], "session-one")
    background_llm.release.set()
    assert worker.wait_idle(2.0)

    pending_path = root / ".pending" / "session-one.json"
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    assert len(payload["turns"]) == 1
    assert payload["turns"][0]["messages"] == second

    worker.submit("session-one")
    assert worker.wait_idle(2.0)
    assert not pending_path.exists()
    worker.close(wait=True, timeout=1.0)


def test_worker_close_does_not_wait_for_an_inflight_request():
    started = threading.Event()
    release = threading.Event()

    class _BlockingEngine:
        def recover_session(self, _source_session, *, force=True):
            assert force is True
            started.set()
            release.wait(2.0)

    worker = MemoryWorker(_BlockingEngine())
    worker.submit("session-one")
    assert started.wait(1.0)

    before = time.monotonic()
    worker.close(wait=False)
    elapsed = time.monotonic() - before

    assert elapsed < 0.1
    release.set()
    worker.close(wait=True, timeout=1.0)


def test_startup_recovery_runs_in_background_for_a_dead_owner(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    foreground = MemoryEngine(_SequenceLLM([]), root=root, project_path=tmp_path)
    foreground.checkpoint(_turn("Remember pytest.", "Understood."), "old-session")

    background = MemoryEngine(_SequenceLLM(["[]"]), root=root, project_path=tmp_path)
    monkeypatch.setattr(background, "_process_alive", lambda _process_id: False)
    worker = MemoryWorker(background)

    before = time.monotonic()
    worker.start(recover_existing=True)
    start_elapsed = time.monotonic() - before

    assert start_elapsed < 0.1
    assert worker.wait_idle(2.0)
    assert background.pending_count() == 0
    worker.close(wait=True, timeout=1.0)


def test_incremental_tool_turn_keeps_verified_procedure_learning(tmp_path):
    root = tmp_path / "memory"
    messages = [
        {"role": "user", "content": "Fix the failure and verify it."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "243 passed"},
        {"role": "assistant", "content": "The fix is verified."},
    ]
    foreground = MemoryEngine(_SequenceLLM([]), root=root, project_path=tmp_path)
    foreground.checkpoint(messages, "tool-session")
    reflection = {
        "task_summary": "Fix and verify the failure",
        "outcome": "success",
        "summary": "The suite passed.",
        "failures": [],
        "root_causes": [],
        "effective_actions": ["Ran pytest"],
        "verification": ["243 passed"],
        "reusable_lessons": ["Run the suite after the fix"],
        "evidence": ["243 passed"],
    }
    procedure = [{
        "action": "create",
        "title": "Verify fixes with pytest",
        "description": "Run the test suite after a fix",
        "content": "Run pytest -q and require the complete suite to pass.",
        "type": "procedure",
        "scope": "project",
        "keywords": ["pytest", "verification"],
        "confidence": 0.9,
        "evidence": "243 passed",
    }]
    background = MemoryEngine(
        _SequenceLLM([json.dumps(reflection), json.dumps(procedure), "[]"]),
        root=root,
        project_path=tmp_path,
    )

    assert background.recover_session("tool-session") is True
    stored = background.store.get("verify-fixes-with-pytest")
    assert stored is not None
    assert stored.status == "candidate"
    assert background.pending_count() == 0

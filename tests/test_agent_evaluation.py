"""Tests for the externally graded whole-Agent evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from corecoder.evaluation import (
    AgentEvaluator,
    EvaluationCheck,
    EvaluationSuite,
    EvaluationTask,
    load_suite,
    write_report,
)


class _FakeLLM:
    model = "fake-eval-model"

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.estimated_cost = 0.0
        self.closed = False

    def close(self):
        self.closed = True


class _NoTasks:
    def list_tasks(self, *, limit=100):
        return ()

    def events(self, *, limit=100):
        return ()


class _FakeAgent:
    def __init__(self, workspace: Path, *, bad=False):
        self.workspace = workspace
        self.bad = bad
        self.llm = _FakeLLM()
        self.tasks = _NoTasks()
        self._tool_calls_used = 0
        self.last_turn_workflow = None
        self.closed = False

    async def chat(self, _prompt, on_tool=None):
        self.llm.total_prompt_tokens += 10
        self.llm.total_completion_tokens += 4
        self.llm.estimated_cost += 0.01
        self._tool_calls_used = 1
        on_tool("write_file", {"file_path": "result.txt"})
        (self.workspace / "result.txt").write_text(
            "wrong" if self.bad else "verified", encoding="utf-8"
        )
        status = SimpleNamespace(value="completed")
        execution = SimpleNamespace(status=status, policy_violations=0)
        self.last_turn_workflow = SimpleNamespace(execution=execution)
        return "wrong" if self.bad else "answer 42"

    def close(self):
        self.closed = True


def _suite() -> EvaluationSuite:
    return EvaluationSuite(
        name="test suite",
        version="1",
        pass_threshold=0.8,
        tasks=(
            EvaluationTask(
                id="write-answer",
                category="coding",
                prompt="write the answer",
                files={"seed.txt": "seed"},
                required_tools=("write_file",),
                max_total_tokens=20,
                max_tool_calls=2,
                checks=(
                    EvaluationCheck(
                        name="answer",
                        kind="answer_contains",
                        value="42",
                    ),
                    EvaluationCheck(
                        name="file",
                        kind="file_equals",
                        path="result.txt",
                        value="verified",
                    ),
                    EvaluationCheck(
                        name="command",
                        kind="command_exit",
                        command=(
                            "python -c \"from pathlib import Path; "
                            "assert Path('result.txt').read_text() == 'verified'\""
                        ),
                    ),
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_agent_evaluator_grades_external_facts_and_usage(tmp_path):
    created = []

    def factory(workspace, _task):
        agent = _FakeAgent(workspace)
        created.append(agent)
        return agent

    evaluator = AgentEvaluator(factory, work_root=tmp_path / "runs", model="fake")
    report = await evaluator.run(_suite(), repeat=2)

    assert report.passed is True
    assert report.summary.overall_score == 1.0
    assert report.summary.reliability_rate == 1.0
    assert report.summary.total_prompt_tokens == 20
    assert report.summary.total_completion_tokens == 8
    assert report.summary.total_tool_calls == 2
    assert report.summary.total_estimated_cost_usd == 0.02
    assert all(item.passed for item in report.attempts)
    assert all(item.answer_sha256 for item in report.attempts)
    assert all(agent.closed and agent.llm.closed for agent in created)
    assert not list((tmp_path / "runs").iterdir())


@pytest.mark.asyncio
async def test_agent_evaluator_reports_failures_and_can_keep_workspace(tmp_path):
    evaluator = AgentEvaluator(
        lambda workspace, _task: _FakeAgent(workspace, bad=True),
        work_root=tmp_path / "runs",
        keep_workspaces=True,
    )
    report = await evaluator.run(_suite())

    attempt = report.attempts[0]
    assert report.passed is False
    assert attempt.passed is False
    assert attempt.correctness_score < 1
    assert Path(attempt.workspace, "result.txt").read_text(encoding="utf-8") == "wrong"


@pytest.mark.asyncio
async def test_evaluation_paths_cannot_escape_workspace(tmp_path):
    suite = EvaluationSuite(
        name="unsafe",
        version="1",
        tasks=(
            EvaluationTask(
                id="escape",
                category="safety",
                prompt="noop",
                files={"../outside.txt": "no"},
                checks=(
                    EvaluationCheck(name="absent", kind="file_not_exists", path="outside.txt"),
                ),
            ),
        ),
    )
    evaluator = AgentEvaluator(
        lambda workspace, _task: _FakeAgent(workspace),
        work_root=tmp_path / "runs",
    )
    report = await evaluator.run(suite)

    assert report.attempts[0].status == "error"
    assert "escapes its root" in report.attempts[0].error
    assert not (tmp_path / "outside.txt").exists()


def test_suite_load_and_atomic_report_write(tmp_path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(_suite().model_dump_json(), encoding="utf-8")
    loaded = load_suite(suite_path)
    assert loaded.name == "test suite"

    duplicate = json.loads(suite_path.read_text(encoding="utf-8"))
    duplicate["tasks"].append(duplicate["tasks"][0])
    suite_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="task ids must be unique"):
        load_suite(suite_path)


@pytest.mark.asyncio
async def test_report_round_trip(tmp_path):
    evaluator = AgentEvaluator(
        lambda workspace, _task: _FakeAgent(workspace),
        work_root=tmp_path / "runs",
    )
    report = await evaluator.run(_suite())
    destination = write_report(report, tmp_path / "reports" / "result.json")
    restored = json.loads(destination.read_text(encoding="utf-8"))
    assert restored["summary"]["overall_score"] == 1.0
    assert not list(destination.parent.glob("*.tmp"))


"""Isolated runner and deterministic graders for whole-Agent evaluations."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    CheckResult,
    DimensionSummary,
    EvaluationAttempt,
    EvaluationCheck,
    EvaluationReport,
    EvaluationSuite,
    EvaluationSummary,
    EvaluationTask,
)

AgentFactory = Callable[[Path, EvaluationTask], Any]
_MAX_EVIDENCE_CHARS = 4_000


def load_suite(path: str | Path) -> EvaluationSuite:
    """Load and validate a JSON evaluation suite."""

    suite_path = Path(path).expanduser().resolve()
    return EvaluationSuite.model_validate_json(suite_path.read_text(encoding="utf-8"))


def write_report(report: EvaluationReport, path: str | Path) -> Path:
    """Atomically persist a machine-readable report."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


class AgentEvaluator:
    """Run a suite against fresh Agent instances and grade external effects."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        work_root: str | Path,
        suite_root: str | Path | None = None,
        keep_workspaces: bool = False,
        model: str = "unknown",
    ):
        self.agent_factory = agent_factory
        self.work_root = Path(work_root).expanduser().resolve()
        self.suite_root = Path(suite_root or Path.cwd()).expanduser().resolve()
        self.keep_workspaces = keep_workspaces
        self.model = model

    async def run(
        self,
        suite: EvaluationSuite,
        *,
        repeat: int = 1,
        pass_threshold: float | None = None,
    ) -> EvaluationReport:
        if not 1 <= repeat <= 20:
            raise ValueError("repeat must be between 1 and 20")
        threshold = suite.pass_threshold if pass_threshold is None else pass_threshold
        if not 0 <= threshold <= 1:
            raise ValueError("pass_threshold must be between 0 and 1")

        self.work_root.mkdir(parents=True, exist_ok=True)
        started = _utc_now()
        attempts: list[EvaluationAttempt] = []
        for task in suite.tasks:
            for attempt_number in range(1, repeat + 1):
                attempts.append(await self._run_attempt(task, attempt_number))
        summary = _summarize(suite, attempts, repeat)
        return EvaluationReport(
            suite_name=suite.name,
            suite_version=suite.version,
            model=self.model,
            started_at=started,
            finished_at=_utc_now(),
            repeat=repeat,
            pass_threshold=threshold,
            passed=(
                summary.overall_score >= threshold
                and summary.pass_rate >= threshold
            ),
            summary=summary,
            attempts=tuple(attempts),
        )

    async def _run_attempt(
        self,
        task: EvaluationTask,
        attempt_number: int,
    ) -> EvaluationAttempt:
        workspace = Path(tempfile.mkdtemp(prefix=f"{task.id}-", dir=self.work_root)).resolve()
        agent = None
        answer = ""
        error = ""
        status = "error"
        tool_names: list[str] = []
        started = time.perf_counter()
        prompt_tokens_before = completion_tokens_before = 0
        cost_before: float | None = None
        policy_violations = 0
        prompt_tokens = completion_tokens = tool_calls = 0
        estimated_cost: float | None = None
        try:
            self._materialize(task, workspace)
            agent = self.agent_factory(workspace, task)
            llm = agent.llm
            prompt_tokens_before = int(getattr(llm, "total_prompt_tokens", 0) or 0)
            completion_tokens_before = int(getattr(llm, "total_completion_tokens", 0) or 0)
            cost_before = getattr(llm, "estimated_cost", None)

            def on_tool(name: str, _arguments: dict[str, Any]) -> None:
                tool_names.append(name)

            try:
                async def run_turns() -> str:
                    final_answer = ""
                    for prompt in (task.prompt, *task.follow_up_prompts):
                        final_answer = await agent.chat(prompt, on_tool=on_tool)
                    return final_answer

                answer = await asyncio.wait_for(
                    run_turns(),
                    timeout=task.timeout_seconds,
                )
                turn = getattr(agent, "last_turn_workflow", None)
                execution = getattr(turn, "execution", None)
                status = str(getattr(getattr(execution, "status", None), "value", "completed"))
                policy_violations = int(getattr(execution, "policy_violations", 0) or 0)
            except asyncio.TimeoutError:
                status = "timeout"
                error = f"agent exceeded {task.timeout_seconds:g}s timeout"
            except Exception as exc:  # noqa: BLE001 - one bad task must not abort the suite
                status = "error"
                error = f"{type(exc).__name__}: {exc}"[:2_000]

            task_snapshots = tuple(agent.tasks.list_tasks(limit=1_000)) if hasattr(agent, "tasks") else ()
            for snapshot in task_snapshots:
                result = agent.tasks.result(snapshot.task_id)
                if result is not None:
                    policy_violations += result.policy_violations
            child_tool_calls = sum(snapshot.usage.tool_calls for snapshot in task_snapshots)
            tool_calls = int(getattr(agent, "_tool_calls_used", len(tool_names)) or 0) + child_tool_calls
            if hasattr(agent, "tasks"):
                tool_names.extend(
                    event.tool_name
                    for event in agent.tasks.events(limit=1_000)
                    if event.tool_name
                )

            llm = agent.llm
            prompt_tokens = max(
                0,
                int(getattr(llm, "total_prompt_tokens", 0) or 0) - prompt_tokens_before,
            )
            completion_tokens = max(
                0,
                int(getattr(llm, "total_completion_tokens", 0) or 0)
                - completion_tokens_before,
            )
            cost_after = getattr(llm, "estimated_cost", None)
            estimated_cost = (
                max(0.0, float(cost_after) - float(cost_before))
                if cost_after is not None and cost_before is not None
                else None
            )
        except Exception as exc:  # noqa: BLE001 - materialization/factory failures are results
            status = "error"
            error = f"{type(exc).__name__}: {exc}"[:2_000]
        finally:
            duration_ms = (time.perf_counter() - started) * 1_000

        checks = [await _evaluate_check(check, answer, workspace) for check in task.checks]
        required_tools_ok = all(name in tool_names for name in task.required_tools)
        forbidden_tools_ok = all(name not in tool_names for name in task.forbidden_tools)
        for name in task.required_tools:
            checks.append(CheckResult(
                name=f"required tool: {name}",
                kind="tool_used",
                passed=name in tool_names,
                evidence=f"observed tools: {', '.join(dict.fromkeys(tool_names)) or '[none]'}",
            ))
        for name in task.forbidden_tools:
            checks.append(CheckResult(
                name=f"forbidden tool: {name}",
                kind="tool_not_used",
                passed=name not in tool_names,
                evidence=f"observed tools: {', '.join(dict.fromkeys(tool_names)) or '[none]'}",
            ))

        correctness = _weighted_check_score(checks[: len(task.checks)])
        completion = 1.0 if status == "completed" and required_tools_ok else 0.0
        safety = (
            1.0
            if policy_violations <= task.max_policy_violations and forbidden_tools_ok
            else 0.0
        )
        total_tokens = prompt_tokens + completion_tokens
        efficiency_checks = [
            task.max_total_tokens is None or total_tokens <= task.max_total_tokens,
            task.max_tool_calls is None or tool_calls <= task.max_tool_calls,
            task.max_duration_seconds is None or duration_ms <= task.max_duration_seconds * 1_000,
        ]
        efficiency = sum(efficiency_checks) / len(efficiency_checks)
        required_checks_ok = all(item.passed for item in checks if item.required)
        passed = (
            completion == 1.0
            and safety == 1.0
            and efficiency == 1.0
            and required_checks_ok
            and required_tools_ok
            and forbidden_tools_ok
        )
        multiplier = 0.7 + 0.1 * completion + 0.1 * safety + 0.1 * efficiency
        score = round(correctness * multiplier, 4)

        workspace_value = str(workspace) if self.keep_workspaces else ""
        result = EvaluationAttempt(
            task_id=task.id,
            category=task.category,
            attempt=attempt_number,
            passed=passed,
            status=status if status in {"completed", "partial", "failed", "timeout", "error"} else "error",
            score=score,
            correctness_score=correctness,
            completion_score=completion,
            safety_score=safety,
            efficiency_score=efficiency,
            duration_ms=round(duration_ms, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tool_calls,
            tool_names=tuple(dict.fromkeys(tool_names)),
            policy_violations=policy_violations,
            estimated_cost_usd=estimated_cost,
            answer_excerpt=answer[:2_000],
            answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            error=error,
            checks=tuple(checks),
            workspace=workspace_value,
        )
        if agent is not None:
            try:
                agent.close()
            finally:
                close_llm = getattr(agent.llm, "close", None)
                if callable(close_llm):
                    close_llm()
        if not self.keep_workspaces:
            _remove_workspace(workspace, self.work_root)
        return result

    def _materialize(self, task: EvaluationTask, workspace: Path) -> None:
        if task.fixture:
            source = _safe_path(self.suite_root, task.fixture)
            if not source.is_dir():
                raise ValueError(f"fixture is not a directory: {task.fixture}")
            if any(path.is_symlink() for path in source.rglob("*")):
                raise ValueError(f"fixture may not contain symlinks: {task.fixture}")
            shutil.copytree(source, workspace, dirs_exist_ok=True)
        for relative, content in task.files.items():
            destination = _safe_path(workspace, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")


async def _evaluate_check(
    check: EvaluationCheck,
    answer: str,
    workspace: Path,
) -> CheckResult:
    passed = False
    evidence = ""
    comparable_answer = answer if check.case_sensitive else answer.casefold()
    comparable_value = check.value if check.case_sensitive else check.value.casefold()
    try:
        if check.kind == "answer_contains":
            passed = comparable_value in comparable_answer
            evidence = f"answer contains expected value: {passed}"
        elif check.kind == "answer_equals":
            passed = comparable_answer.strip() == comparable_value.strip()
            evidence = f"answer exactly matches expected value: {passed}"
        elif check.kind == "answer_regex":
            flags = 0 if check.case_sensitive else re.IGNORECASE
            passed = re.search(check.value, answer, flags=flags) is not None
            evidence = f"answer matches /{check.value}/: {passed}"
        elif check.kind in {
            "file_exists",
            "file_not_exists",
            "file_contains",
            "file_not_contains",
            "file_equals",
        }:
            target = _safe_path(workspace, check.path)
            exists = target.is_file()
            if check.kind == "file_exists":
                passed, evidence = exists, f"file exists: {exists}"
            elif check.kind == "file_not_exists":
                passed, evidence = not target.exists(), f"path absent: {not target.exists()}"
            else:
                content = target.read_text(encoding="utf-8") if exists else ""
                left = content if check.case_sensitive else content.casefold()
                right = check.value if check.case_sensitive else check.value.casefold()
                if check.kind == "file_contains":
                    passed = exists and right in left
                    evidence = f"file contains expected value: {passed}"
                elif check.kind == "file_not_contains":
                    passed = exists and right not in left
                    evidence = f"file excludes forbidden value: {passed}"
                else:
                    passed = exists and left == right
                    evidence = f"file exactly matches expected value: {passed}"
        elif check.kind == "command_exit":
            proc = await asyncio.create_subprocess_shell(
                check.command,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=check.timeout_seconds
                )
                output = (stdout + b"\n" + stderr).decode("utf-8", errors="replace").strip()
                passed = proc.returncode == check.expected_exit_code
                evidence = (
                    f"exit={proc.returncode}, expected={check.expected_exit_code}\n{output}"
                )[-_MAX_EVIDENCE_CHARS:]
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                evidence = f"grader command timed out after {check.timeout_seconds:g}s"
    except Exception as exc:  # noqa: BLE001 - grader failures are recorded as evidence
        evidence = f"grader error: {type(exc).__name__}: {exc}"
    return CheckResult(
        name=check.name,
        kind=check.kind,
        passed=passed,
        required=check.required,
        weight=check.weight,
        evidence=evidence[:_MAX_EVIDENCE_CHARS],
    )


def _weighted_check_score(checks: list[CheckResult]) -> float:
    total = sum(item.weight for item in checks)
    return round(sum(item.weight for item in checks if item.passed) / total, 4) if total else 0.0


def _summarize(
    suite: EvaluationSuite,
    attempts: list[EvaluationAttempt],
    repeat: int,
) -> EvaluationSummary:
    task_weights = {task.id: task.weight for task in suite.tasks}
    weights = [task_weights[item.task_id] for item in attempts]
    total_weight = sum(weights)

    def weighted(field: str) -> float:
        if not total_weight:
            return 0.0
        return round(
            sum(getattr(item, field) * weight for item, weight in zip(attempts, weights, strict=True))
            / total_weight,
            4,
        )

    grouped: dict[str, list[EvaluationAttempt]] = defaultdict(list)
    by_task: dict[str, list[EvaluationAttempt]] = defaultdict(list)
    for item in attempts:
        grouped[item.category].append(item)
        by_task[item.task_id].append(item)
    dimensions = {
        category: DimensionSummary(
            attempts=len(items),
            passed=sum(item.passed for item in items),
            pass_rate=round(sum(item.passed for item in items) / len(items), 4),
            average_score=round(sum(item.score for item in items) / len(items), 4),
        )
        for category, items in sorted(grouped.items())
    }
    reliable = sum(
        len(items) == repeat and all(item.passed for item in items)
        for items in by_task.values()
    )
    costs = [item.estimated_cost_usd for item in attempts if item.estimated_cost_usd is not None]
    durations = [item.duration_ms for item in attempts]
    return EvaluationSummary(
        attempts=len(attempts),
        passed_attempts=sum(item.passed for item in attempts),
        pass_rate=round(sum(item.passed for item in attempts) / len(attempts), 4),
        overall_score=weighted("score"),
        correctness_score=weighted("correctness_score"),
        completion_score=weighted("completion_score"),
        safety_score=weighted("safety_score"),
        efficiency_score=weighted("efficiency_score"),
        reliability_rate=round(reliable / len(suite.tasks), 4),
        total_prompt_tokens=sum(item.prompt_tokens for item in attempts),
        total_completion_tokens=sum(item.completion_tokens for item in attempts),
        total_tool_calls=sum(item.tool_calls for item in attempts),
        total_policy_violations=sum(item.policy_violations for item in attempts),
        total_estimated_cost_usd=round(sum(costs), 8) if costs else None,
        average_duration_ms=round(sum(durations) / len(durations), 3),
        p95_duration_ms=round(_percentile(durations, 0.95), 3),
        dimensions=dimensions,
    )


def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"evaluation paths must be relative: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"evaluation path escapes its root: {relative}") from exc
    return resolved


def _remove_workspace(workspace: Path, work_root: Path) -> None:
    resolved = workspace.resolve()
    resolved.relative_to(work_root.resolve())
    if resolved == work_root.resolve():
        raise ValueError("refusing to remove evaluation work root")
    shutil.rmtree(resolved)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

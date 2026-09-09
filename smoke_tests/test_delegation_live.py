"""Opt-in real-provider acceptance for teams and worktree execution."""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.delegation import (
    AcceptanceCheck,
    AgentTeamTemplate,
    TaskRole,
    TaskSpec,
    TaskStatus,
    TeamMemberTemplate,
    WorkspaceMode,
)
from corecoder.llm import LLM, LiteLLM
from corecoder.security import AuditLogger, Guard
from corecoder.tools import ALL_TOOLS

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / ".test_runs" / "real-delegation"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
    )


def _fixture_repo(run_root: Path) -> Path:
    repo = Path(tempfile.mkdtemp(prefix="worktree-live-", dir=run_root))
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "acceptance@example.com")
    _git(repo, "config", "user.name", "CoreCoder Acceptance")
    (repo / ".gitignore").write_text(".corecoder/worktrees/\n", encoding="utf-8")
    (repo / "calculator.py").write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "synthetic fixture")
    return repo


def _verify_fixture(repo: Path) -> str:
    completed = subprocess.run(
        [
            "python",
            "-c",
            "from calculator import add; assert add(7, 5) == 12; print('PASS')",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


async def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="run-", dir=RUNS))
    config = Config.from_env()
    provider = LiteLLM if config.provider == "litellm" else LLM
    llm = provider(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_tokens=min(config.max_tokens, 2_048),
    )
    prompts: list[str] = []
    guard = Guard(
        audit=AuditLogger(run_root / "audit"),
        confirm_callback=lambda _tool, _args, reason, *_rest: prompts.append(reason) or False,
    )

    try:
        # Real staged team, deliberately read-only. Each stage receives only
        # the parent's bounded summary from the previous stage.
        team_agent = Agent(
            llm=llm,
            tools=ALL_TOOLS,
            replay=False,
            guard=guard,
            context_artifacts_enabled=False,
            task_concurrency=2,
            workspace_root=ROOT,
            agent_id="live-team-parent",
        )
        read_member = {
            "allowed_tools": ("read_file",),
            "read_paths": ("pyproject.toml",),
            # Includes the repeated system/tool schema tokens across tool rounds.
            "token_budget": 20_000,
            "max_tool_calls": 4,
            "max_rounds": 4,
            "timeout_seconds": 120,
        }
        team = AgentTeamTemplate(
            name="live-readonly-team",
            members=(
                TeamMemberTemplate(
                    name="researcher", role=TaskRole.RESEARCHER, stage=0, **read_member
                ),
                TeamMemberTemplate(
                    name="executor", role=TaskRole.EXECUTOR, stage=1, **read_member
                ),
                TeamMemberTemplate(
                    name="reviewer", role=TaskRole.REVIEWER, stage=2, **read_member
                ),
            ),
        )
        team_result = await team_agent.run_team(
            {
                "researcher": "Read pyproject.toml and identify the package name and version with evidence.",
                "executor": "Independently verify the package name and version. Do not modify anything.",
                "reviewer": "Cross-check the prior summaries against pyproject.toml. Do not modify anything.",
            },
            template=team,
            context="This is a read-only real-provider acceptance test.",
            acceptance_criteria=("Report the exact package name and version",),
        )
        assert team_result.completed, {
            name: {
                "status": result.status.value,
                "usage": result.usage.model_dump(),
                "error": result.error,
            }
            for name, result in team_result.results.items()
        }
        assert len(team_result.results) == 3
        assert all(not result.modifications for result in team_result.results.values())
        assert all(result.parent_id == "live-team-parent" for result in team_result.results.values())
        team_agent.close()

        # Real model + real tool calls + detached worktree + central merge.
        repo = _fixture_repo(run_root)
        worktree_agent = Agent(
            llm=llm,
            tools=ALL_TOOLS,
            replay=False,
            guard=guard,
            context_artifacts_enabled=False,
            workspace_root=repo,
            agent_id="live-worktree-parent",
        )
        spec = TaskSpec(
            objective=(
                "Fix calculator.py: add(left, right) incorrectly subtracts. "
                "Read the file and change only the return expression so it adds."
            ),
            role=TaskRole.EXECUTOR,
            execution_mode=WorkspaceMode.WORKTREE,
            allowed_tools=("read_file", "edit_file"),
            read_paths=(".",),
            write_paths=("calculator.py",),
            token_budget=20_000,
            max_tool_calls=6,
            max_rounds=6,
            timeout_seconds=180,
            acceptance_criteria=("add(7, 5) returns 12",),
        )
        worktree_result = await worktree_agent.delegate(spec)
        assert worktree_result.status == TaskStatus.COMPLETED, worktree_result.error
        assert worktree_result.merge_status == "applied", worktree_result.model_dump()
        assert worktree_result.workspace_path == ""
        assert worktree_result.modifications == [str(repo / "calculator.py")]

        verification = await asyncio.to_thread(_verify_fixture, repo)
        assert verification == "PASS"
        accepted = worktree_agent.accept_task(spec.task_id, [AcceptanceCheck(
            criterion="add(7, 5) returns 12",
            passed=True,
            evidence="Parent executed calculator.add(7, 5) and observed 12",
            verified_by_parent=True,
        )])
        assert accepted.accepted and not accepted.requires_parent_review

        undo = worktree_agent.changes.undo_all()
        assert not undo.conflicts and not undo.errors
        assert "return left - right" in (repo / "calculator.py").read_text(encoding="utf-8")
        worktree_agent.close()

        audit = guard.audit.query(limit=100)
        delegated_entries = [entry for entry in audit.entries if entry.get("task_id")]
        assert delegated_entries
        assert all(entry.get("agent_id") and entry.get("parent_id") for entry in delegated_entries)
        assert not prompts, f"delegated test unexpectedly requested confirmation: {prompts}"

        print(json.dumps({
            "model": config.model,
            "team": {
                "completed": team_result.completed,
                "members": {
                    name: result.status.value for name, result in team_result.results.items()
                },
            },
            "worktree": {
                "status": worktree_result.status.value,
                "merge": worktree_result.merge_status,
                "parent_verification": "passed",
                "accepted": accepted.accepted,
                "undo": "passed",
            },
            "audit_entries": len(delegated_entries),
            "permission_prompts": len(prompts),
        }, ensure_ascii=False, indent=2))
    finally:
        llm.close()


if __name__ == "__main__":
    asyncio.run(main())

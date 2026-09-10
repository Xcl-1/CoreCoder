"""Command-line entry point for repeatable whole-Agent evaluations."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.llm import LLM, LiteLLM
from corecoder.security import AuditLogger, ConfirmationContext, Guard, NetworkPolicy
from corecoder.skills import SkillManager

from .models import EvaluationTask
from .runner import AgentEvaluator, load_suite, write_report


def _ratio(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corecoder-eval",
        description="Run an isolated, externally graded CoreCoder capability suite.",
    )
    parser.add_argument(
        "suite",
        help="Path to a JSON evaluation suite, or 'builtin' for the bundled suite",
    )
    parser.add_argument("-o", "--output", default="evaluation-report.json")
    parser.add_argument("-m", "--model", help="Override $CORECODER_MODEL")
    parser.add_argument("--base-url", help="Override $OPENAI_BASE_URL")
    parser.add_argument("--api-key", help="Override the configured API key")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per task (1-20)")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="ID",
        help="Run only this task id; repeat the option to select multiple tasks",
    )
    parser.add_argument("--fail-under", type=_ratio, help="Override suite pass threshold")
    parser.add_argument(
        "--work-dir",
        default=".corecoder/evaluations",
        help="Evaluation state and temporary workspaces",
    )
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--no-skills", action="store_true", help="Disable Skill routing")
    parser.add_argument(
        "--allow-agent-shell",
        action="store_true",
        help=(
            "Auto-confirm medium-risk local shell calls in evaluation workspaces. "
            "Explicit network and high-risk calls remain denied. Use only with trusted suites."
        ),
    )
    return parser


def _local_confirmation(
    tool_name: str,
    _arguments: dict,
    _reason: str,
    context: ConfirmationContext | None = None,
) -> bool:
    """Permit only non-network, non-high-risk shell calls in disposable workspaces."""

    if tool_name != "bash" or context is None:
        return False
    if context.risk_level in {"high", "critical"}:
        return False
    if (
        context.network_destinations
        or context.network_mutating
        or context.carries_credentials
        or context.follows_redirects
    ):
        return False
    return context.network_action in {"", "allow", "ask"}


async def _run(args: argparse.Namespace) -> int:
    suite_path = _suite_path(args.suite)
    suite = load_suite(suite_path)
    if args.task:
        requested = set(args.task)
        known = {task.id for task in suite.tasks}
        unknown = requested - known
        if unknown:
            raise ValueError("unknown evaluation task id(s): " + ", ".join(sorted(unknown)))
        suite = suite.model_copy(update={
            "name": f"{suite.name} (selected tasks)",
            "tasks": tuple(task for task in suite.tasks if task.id in requested),
        })
    config = Config.from_env()
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key
    if not config.api_key:
        raise ValueError(
            "no API key found; set OPENAI_API_KEY, DEEPSEEK_API_KEY, "
            "or CORECODER_API_KEY"
        )

    work_root = Path(args.work_dir).expanduser().resolve()
    workspace_root = work_root / "workspaces"
    state_root = work_root / "state"
    provider = LiteLLM if config.provider == "litellm" else LLM

    def create_agent(workspace: Path, task: EvaluationTask) -> Agent:
        llm = provider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        skills = None
        if config.skills_enabled and not args.no_skills:
            skills = SkillManager.create(
                project_path=workspace,
                user_dir=state_root / "skills",
                top_k=config.skill_top_k,
                max_active=config.skill_max_active,
                max_prompt_chars=config.skill_prompt_chars,
                min_score=config.skill_min_score,
                auto_confidence=config.skill_auto_confidence,
                clarify_confidence=config.skill_clarify_confidence,
                ambiguity_margin=config.skill_ambiguity_margin,
            )
        guard = Guard(
            audit=AuditLogger(state_root / "audit" / workspace.name),
            confirm_callback=_local_confirmation if args.allow_agent_shell else None,
            network_policy=NetworkPolicy("confirm"),
        )
        return Agent(
            llm=llm,
            max_context_tokens=config.max_context_tokens,
            max_rounds=50,
            replay=False,
            guard=guard,
            memory=None,
            skills=skills,
            context_artifacts_enabled=config.context_artifacts_enabled,
            context_artifacts_dir=workspace / ".corecoder-context",
            context_artifact_threshold=config.context_artifact_threshold,
            context_artifact_ttl_days=config.context_artifact_ttl_days,
            context_artifact_max_mb=config.context_artifact_max_mb,
            task_state_dir=None,
            task_concurrency=config.task_concurrency,
            max_subagents_per_round=config.max_subagents_per_round,
            workspace_root=workspace,
            token_budget=task.max_total_tokens,
            max_tool_calls=task.max_tool_calls,
        )

    evaluator = AgentEvaluator(
        create_agent,
        work_root=workspace_root,
        suite_root=suite_path.parent,
        keep_workspaces=args.keep_workspaces,
        model=config.model,
    )
    report = await evaluator.run(
        suite,
        repeat=args.repeat,
        pass_threshold=args.fail_under,
    )
    report_path = write_report(report, args.output)
    summary = report.summary
    print(f"Suite: {report.suite_name} v{report.suite_version}")
    print(f"Model: {report.model}")
    print(
        f"Score: {summary.overall_score:.1%} | "
        f"Pass: {summary.passed_attempts}/{summary.attempts} | "
        f"Reliability: {summary.reliability_rate:.1%}"
    )
    print(
        f"Tokens: {summary.total_prompt_tokens + summary.total_completion_tokens} | "
        f"Tools: {summary.total_tool_calls} | "
        f"Policy violations: {summary.total_policy_violations}"
    )
    for name, dimension in summary.dimensions.items():
        print(
            f"  {name}: score={dimension.average_score:.1%}, "
            f"pass={dimension.passed}/{dimension.attempts}"
        )
    print(f"Report: {report_path}")
    return 0 if report.passed else 1


def _suite_path(value: str) -> Path:
    if value != "builtin":
        return Path(value).expanduser().resolve()
    packaged = Path(__file__).resolve().parent / "suites" / "agent_core_v1.json"
    if packaged.is_file():
        return packaged
    source_checkout = Path(__file__).resolve().parents[2] / "benchmarks" / "agent_core_v1.json"
    if source_checkout.is_file():
        return source_checkout
    raise ValueError("bundled evaluation suite is missing from this installation")


def main() -> None:
    args = _parser().parse_args()
    try:
        code = asyncio.run(_run(args))
    except (OSError, ValueError) as exc:
        print(f"corecoder-eval: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()

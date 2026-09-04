"""Tests for skill discovery, routing, prompt injection, and tool policy."""

from __future__ import annotations

import json

import pytest

from corecoder.agent import Agent, AgentRole
from corecoder.models import LLMResponse
from corecoder.skills import (
    RoutingCase,
    RoutingContext,
    SkillCandidate,
    SkillManager,
    SkillRegistry,
    SkillRouter,
    SkillSource,
    evaluate_router,
)
from corecoder.skills.loader import load_instructions, load_skill
from corecoder.tools import get_tool


def _write_skill(root, skill_id="test.alpha", **overrides):
    directory = root / skill_id.replace(".", "-")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "id": skill_id,
        "name": "Alpha workflow",
        "version": "1.0.0",
        "summary": "Handle alpha deployment failures safely.",
        "category": ["testing"],
        "tags": ["alpha", "deployment"],
        "aliases": ["alpha workflow"],
        "intents": ["fix alpha deployment", "修复 alpha 部署"],
        "applies_when": ["An alpha deployment fails."],
        "not_when": [],
        "examples": {
            "positive": ["the alpha deployment failed after release"],
            "negative": ["write alpha deployment documentation"],
        },
        "tools": {"required": [], "recommended": ["read_file"], "forbidden": []},
        "token_budget": 800,
        "status": "active",
    }
    manifest.update(overrides)
    (directory / "skill.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (directory / "SKILL.md").write_text("# Workflow\n\nFollow the alpha procedure.\n", encoding="utf-8")
    return directory


def _registry(root, scope="project", priority=30):
    return SkillRegistry([SkillSource(scope, root, priority)]).discover()


def test_loader_validates_manifest_and_defers_instructions(tmp_path):
    directory = _write_skill(tmp_path)
    skill = load_skill(directory, "project", 30)
    assert skill.manifest.id == "test.alpha"
    assert skill.instructions == ""
    assert "alpha procedure" in load_instructions(skill)


def test_registry_records_invalid_package_without_aborting(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "skill.json").write_text("{}", encoding="utf-8")
    (bad / "SKILL.md").write_text("broken", encoding="utf-8")
    registry = _registry(tmp_path)
    assert len(registry) == 0
    assert len(registry.errors) == 1


def test_registry_rejects_contradictory_tool_policy(tmp_path):
    _write_skill(
        tmp_path,
        tools={"required": ["read_file"], "recommended": [], "forbidden": ["read_file"]},
    )
    registry = _registry(tmp_path)
    assert len(registry) == 0
    assert any("both allowed and forbidden" in error for error in registry.errors)


def test_project_scope_overrides_user_scope(tmp_path):
    user = tmp_path / "user"
    project = tmp_path / "project"
    _write_skill(user, summary="user copy")
    _write_skill(project, summary="project copy")
    registry = SkillRegistry([
        SkillSource("user", user, 20),
        SkillSource("project", project, 30),
    ]).discover()
    assert registry.get("test.alpha").manifest.summary == "project copy"
    assert registry.get("test.alpha").scope == "project"
    assert len(registry.overrides) == 1


def test_router_recalls_ranks_and_renders_selected_skill(tmp_path):
    _write_skill(tmp_path)
    router = SkillRouter(_registry(tmp_path), top_k=5, max_active=2, max_prompt_chars=2000)
    result = router.route("Please fix the alpha deployment failure")
    assert result.selected_ids == ["test.alpha"]
    assert "Follow the alpha procedure" in result.prompt
    assert result.candidates[0].reasons


def test_router_supports_explicit_skill_and_reports_missing_id(tmp_path):
    _write_skill(tmp_path)
    router = SkillRouter(_registry(tmp_path))
    selected = router.route("Use $test.alpha for this unrelated request")
    missing = router.route("Use $test.missing for this request")
    assert selected.selected_ids == ["test.alpha"]
    assert any("was not found" in reason for reason in missing.rejected)


def test_router_does_not_treat_shell_variables_or_paths_as_explicit_skills(tmp_path):
    _write_skill(tmp_path)
    router = SkillRouter(_registry(tmp_path))
    result = router.route(
        "Inspect $PATH, $HOME, $env:USERNAME and C:\\Users\\$test.missing\\file.py"
    )
    assert not any("explicitly requested" in reason for reason in result.rejected)


def test_router_keeps_support_for_existing_simple_explicit_skill_ids(tmp_path):
    _write_skill(tmp_path, "alpha")
    result = SkillRouter(_registry(tmp_path), max_active=1).route(
        "Use $alpha for this unrelated request"
    )
    assert result.selected_ids == ["alpha"]


def test_router_preserves_explicit_candidates_and_reports_active_limit(tmp_path):
    for name in ("alpha", "beta", "gamma"):
        _write_skill(tmp_path, f"test.{name}", name=f"{name} workflow")
    router = SkillRouter(_registry(tmp_path), top_k=1, max_active=2)
    result = router.route("Use $test.alpha $test.beta and $test.gamma")
    assert {candidate.skill.manifest.id for candidate in result.candidates} == {
        "test.alpha",
        "test.beta",
        "test.gamma",
    }
    assert len(result.selected_ids) == 2
    assert any("maximum active skill count 2 reached" in reason for reason in result.rejected)


def test_router_honors_boundaries_and_required_tools(tmp_path):
    _write_skill(
        tmp_path,
        not_when=["documentation only"],
        tools={"required": ["bash"], "recommended": [], "forbidden": []},
    )
    router = SkillRouter(_registry(tmp_path))
    boundary = router.route("alpha deployment documentation only", available_tools={"bash"})
    missing_tool = router.route("fix alpha deployment", available_tools={"read_file"})
    assert boundary.selected == []
    assert missing_tool.selected == []
    assert any("not_when" in reason for reason in boundary.rejected)
    assert any("missing required tools" in reason for reason in missing_tool.rejected)


def test_router_treats_explicit_empty_tool_set_as_no_tools(tmp_path):
    _write_skill(
        tmp_path,
        tools={"required": ["read_file"], "recommended": [], "forbidden": []},
    )
    result = SkillRouter(_registry(tmp_path)).route(
        "fix alpha deployment",
        available_tools=set(),
    )
    assert result.selected == []
    assert any("missing required tools" in reason for reason in result.rejected)


def test_router_uses_positive_and_negative_examples(tmp_path):
    _write_skill(tmp_path)
    router = SkillRouter(_registry(tmp_path))
    positive = router.route("the alpha deployment failed after release")
    negative = router.route("write alpha deployment documentation")
    assert positive.selected_ids == ["test.alpha"]
    assert any("positive example" in reason for reason in positive.candidates[0].reasons)
    assert negative.selected == []
    assert any("negative example" in reason for reason in negative.rejected)


def test_router_resolves_exclusive_groups(tmp_path):
    _write_skill(tmp_path, "test.alpha", exclusive_group="deploy")
    _write_skill(
        tmp_path,
        "test.beta",
        name="Beta workflow",
        summary="Handle alpha beta deployment failures.",
        tags=["alpha", "beta", "deployment"],
        intents=["fix alpha deployment"],
        exclusive_group="deploy",
    )
    result = SkillRouter(_registry(tmp_path), max_active=3).route("fix alpha deployment")
    assert len(result.selected) == 1
    assert any("exclusive group" in reason for reason in result.rejected)


def test_router_resolves_cross_skill_tool_policy_conflicts(tmp_path):
    _write_skill(
        tmp_path,
        "test.alpha",
        priority=10,
        tools={"required": ["bash"], "recommended": [], "forbidden": []},
    )
    _write_skill(
        tmp_path,
        "test.beta",
        name="Beta workflow",
        summary="Handle alpha deployment failures safely.",
        tags=["alpha", "deployment"],
        intents=["fix alpha deployment"],
        tools={"required": [], "recommended": [], "forbidden": ["bash"]},
    )
    result = SkillRouter(_registry(tmp_path), max_active=3).route(
        "fix alpha deployment",
        available_tools={"bash"},
    )
    assert len(result.selected_ids) == 1
    assert result.selected_ids[0] in {"test.alpha", "test.beta"}
    assert any("tool policy conflicts" in reason for reason in result.rejected)


def test_router_resolves_declared_skill_conflicts_in_both_directions(tmp_path):
    _write_skill(tmp_path, "test.alpha", priority=10, conflicts_with=["test.beta"])
    _write_skill(tmp_path, "test.beta", priority=0)
    first = SkillRouter(_registry(tmp_path), max_active=3).route(
        "Use $test.alpha and $test.beta"
    )

    _write_skill(tmp_path, "test.alpha", priority=10, conflicts_with=[])
    _write_skill(tmp_path, "test.beta", priority=0, conflicts_with=["test.alpha"])
    second = SkillRouter(_registry(tmp_path), max_active=3).route(
        "Use $test.alpha and $test.beta"
    )

    assert len(first.selected_ids) == 1
    assert len(second.selected_ids) == 1
    assert any("conflict" in reason for reason in first.rejected)
    assert any("conflict" in reason for reason in second.rejected)


def test_skill_search_limit_is_independent_of_route_top_k(tmp_path):
    for index in range(12):
        _write_skill(tmp_path, f"test.skill-{index}")
    router = SkillRouter(_registry(tmp_path), top_k=2)
    assert len(router.search("alpha deployment", limit=11)) == 11
    assert router.search("alpha deployment", limit=0) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"max_active": 0}, "max_active"),
        ({"max_prompt_chars": 0}, "max_prompt_chars"),
        ({"min_score": float("nan")}, "min_score"),
        ({"min_score": -0.1}, "min_score"),
    ],
)
def test_router_rejects_invalid_configuration(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        SkillRouter(_registry(tmp_path), **kwargs)


def test_router_reports_prompt_budget_rejection_consistently(tmp_path):
    _write_skill(tmp_path)
    result = SkillRouter(_registry(tmp_path), max_prompt_chars=200).route(
        "fix alpha deployment"
    )
    assert result.selected == []
    assert result.prompt == ""
    assert any("prompt budget exceeded" in reason for reason in result.rejected)


def test_schema_v2_signature_improves_ranking_and_is_exposed(tmp_path):
    _write_skill(
        tmp_path,
        schema_version=2,
        signature={
            "domains": ["deployment"],
            "actions": ["repair"],
            "objects": ["release pipeline"],
            "artifacts": [],
            "outputs": ["working deployment"],
            "constraints": [],
        },
    )
    result = SkillRouter(_registry(tmp_path)).route(
        "repair the release pipeline deployment"
    )
    assert result.selected_ids == ["test.alpha"]
    assert result.decision == "auto"
    assert result.signature.actions == {"repair"}
    assert {"domains", "actions", "objects"} <= set(
        result.candidates[0].matched_dimensions
    )


def test_router_abstains_for_weak_generic_request(tmp_path):
    _write_skill(tmp_path)
    result = SkillRouter(_registry(tmp_path)).route("Tell me a joke about the weather")
    assert result.selected == []
    assert result.decision == "abstain"
    assert result.prompt == ""


def test_per_turn_skill_exclusion_overrides_explicit_and_automatic_routing(tmp_path):
    _write_skill(tmp_path)
    result = SkillRouter(_registry(tmp_path)).route(
        "Do not use $test.alpha to fix the alpha deployment"
    )
    assert result.selected == []
    assert any("disabled for this turn" in reason for reason in result.rejected)


def test_v2_mutually_exclusive_near_tie_requests_one_clarification(tmp_path):
    _write_skill(tmp_path, "test.alpha", schema_version=2, exclusive_group="deploy")
    _write_skill(tmp_path, "test.beta", schema_version=2, exclusive_group="deploy")
    result = SkillRouter(
        _registry(tmp_path),
        ambiguity_margin=1.0,
    ).route("fix alpha deployment")
    assert result.selected == []
    assert result.decision == "clarify"
    assert result.needs_clarification
    assert "Alpha workflow" in result.clarification
    assert "Do not begin the task" in result.prompt


def test_explicit_skill_suppresses_unrelated_automatic_ambiguity(tmp_path):
    _write_skill(tmp_path, "test.explicit", schema_version=2)
    _write_skill(tmp_path, "test.alpha", schema_version=2, exclusive_group="deploy")
    _write_skill(tmp_path, "test.beta", schema_version=2, exclusive_group="deploy")
    result = SkillRouter(
        _registry(tmp_path),
        max_active=3,
        ambiguity_margin=1.0,
    ).route("Use $test.explicit to fix alpha deployment")
    assert result.selected_ids == ["test.explicit"]
    assert result.decision == "explicit"
    assert not result.needs_clarification


def test_hard_negative_blocks_a_close_lexical_match(tmp_path):
    _write_skill(
        tmp_path,
        schema_version=2,
        examples={
            "positive": ["fix alpha deployment"],
            "negative": [],
            "hard_negative": ["document alpha deployment"],
            "contrastive": [],
        },
    )
    result = SkillRouter(_registry(tmp_path)).route("document alpha deployment")
    assert result.selected == []
    assert any("hard negative" in reason for reason in result.rejected)


def test_contrastive_example_redirects_to_alternative_skill(tmp_path):
    _write_skill(
        tmp_path,
        "test.source",
        schema_version=2,
        examples={
            "positive": ["alpha deployment"],
            "negative": [],
            "hard_negative": [],
            "contrastive": [
                {
                    "query": "write alpha deployment docs",
                    "expected_skill": "test.docs",
                    "reason": "Documentation is a separate outcome.",
                }
            ],
        },
    )
    _write_skill(
        tmp_path,
        "test.docs",
        schema_version=2,
        name="Specialized writer",
        summary="Produce a specialized written artifact.",
        tags=["writer"],
        intents=["produce written artifact"],
        examples={"positive": [], "negative": []},
    )
    result = SkillRouter(_registry(tmp_path)).route("write alpha deployment docs")
    assert result.selected_ids == ["test.docs"]
    assert any("contrastive route" in reason for reason in result.candidates[0].reasons)


def test_workflow_activates_declared_atomic_dependency(tmp_path):
    _write_skill(
        tmp_path,
        "test.atomic",
        name="Atomic helper",
        layer="atomic",
        summary="Provide a low-level prerequisite.",
        tags=["prerequisite"],
        intents=["atomic prerequisite"],
        examples={"positive": [], "negative": []},
    )
    _write_skill(
        tmp_path,
        "test.workflow",
        schema_version=2,
        priority=10,
        relations={
            "dependencies": ["test.atomic"],
            "composes_with": [],
            "supersedes": [],
        },
    )
    result = SkillRouter(_registry(tmp_path), max_active=2).route("fix alpha deployment")
    assert result.selected_ids == ["test.workflow", "test.atomic"]
    assert result.candidates[0].skill.manifest.layer == "workflow"
    assert "required by test.workflow" in result.selected[1].reasons


def test_declared_composition_can_add_a_relevant_lower_scoring_support_skill(tmp_path):
    _write_skill(
        tmp_path,
        "test.primary",
        priority=10,
        relations={
            "dependencies": [],
            "composes_with": ["test.support"],
            "supersedes": [],
        },
    )
    _write_skill(
        tmp_path,
        "test.support",
        name="Deployment support",
        summary="Support an alpha deployment failure.",
        tags=["alpha", "deployment"],
        intents=["fix alpha deployment"],
        examples={"positive": ["alpha deployment failed"], "negative": []},
    )
    result = SkillRouter(_registry(tmp_path), max_active=2).route("fix alpha deployment")
    assert set(result.selected_ids) == {"test.primary", "test.support"}


def test_shadow_skill_is_scored_but_never_activated(tmp_path):
    _write_skill(tmp_path, schema_version=2, status="shadow")
    result = SkillRouter(_registry(tmp_path)).route("fix alpha deployment")
    assert result.selected == []
    assert result.candidates[0].shadow is True
    assert result.decision == "abstain"


def test_canary_rollout_zero_requires_explicit_invocation(tmp_path):
    _write_skill(
        tmp_path,
        schema_version=2,
        status="canary",
        routing={"allow_implicit": True, "risk": "low", "rollout_percent": 0},
    )
    router = SkillRouter(_registry(tmp_path))
    assert router.route("fix alpha deployment").selected == []
    assert router.route("Use $test.alpha").selected_ids == ["test.alpha"]


def test_catalog_reports_missing_capability_graph_target(tmp_path):
    _write_skill(
        tmp_path,
        schema_version=2,
        relations={
            "dependencies": ["test.missing"],
            "composes_with": [],
            "supersedes": [],
        },
    )
    router = SkillRouter(_registry(tmp_path))
    assert any(issue.code == "missing-relation" for issue in router.catalog.issues)


def test_offline_router_evaluation_measures_precision_abstention_and_cost(tmp_path):
    _write_skill(tmp_path)
    metrics = evaluate_router(
        SkillRouter(_registry(tmp_path)),
        [
            RoutingCase(query="fix alpha deployment", expected_skill="test.alpha"),
            RoutingCase(query="tell me a joke", expected_skill=None),
        ],
    )
    assert metrics.precision_at_1 == 1.0
    assert metrics.invocation_precision == 1.0
    assert metrics.false_activation_rate == 0.0
    assert metrics.average_loaded_tokens > 0


class _CaptureLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append({"messages": messages, "tools": tools})
        return LLMResponse(content="done")


@pytest.mark.asyncio
async def test_agent_injects_skill_and_enforces_forbidden_tools(tmp_path):
    _write_skill(
        tmp_path,
        tools={"required": ["read_file"], "recommended": [], "forbidden": ["bash"]},
    )
    registry = _registry(tmp_path)
    manager = SkillManager(registry, SkillRouter(registry))
    llm = _CaptureLLM()
    agent = Agent(
        llm=llm,
        tools=[get_tool("read_file"), get_tool("bash")],
        skills=manager,
        replay=False,
    )

    assert await agent.chat("fix alpha deployment") == "done"
    call = llm.calls[0]
    assert "Skill: test.alpha" in call["messages"][0]["content"]
    assert {item["function"]["name"] for item in call["tools"]} == {"read_file"}

    class _TC:
        def __init__(self):
            self.name = "bash"
            self.arguments = {"command": "echo should-not-run"}

    result, _, success = await agent._exec_tool(_TC())
    assert "active skill policy forbids" in result
    assert not success


def test_builtin_skills_are_discoverable(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    ids = {skill.manifest.id for skill in manager.registry.all()}
    assert {
        "coding.code-review",
        "python.test-debug",
        "coding.safe-refactor",
        "coding.feature-implementation",
        "coding.bug-fix",
        "testing.test-generation",
        "docs.technical-documentation",
        "security.code-audit",
        "repository.architecture-analysis",
        "coding.api-design",
        "coding.api-migration",
        "maintenance.dependency-upgrade",
        "quality.performance-optimization",
        "debugging.concurrency",
        "debugging.configuration",
        "quality.error-handling-hardening",
        "maintenance.dead-code-cleanup",
        "quality.backward-compatibility",
        "coding.cli-design",
        "testing.flaky-test-debug",
        "testing.integration-test-generation",
        "testing.coverage-analysis",
        "devops.ci-cd-debug",
        "devops.containerization",
        "devops.deployment-troubleshooting",
        "release.preparation",
        "release.package-publishing",
        "data.database-migration",
        "data.validation",
        "security.secrets-audit",
        "security.dependency-audit",
        "reliability.logging-observability",
        "docs.migration-guide",
        "docs.changelog-release-notes",
        "reliability.incident-root-cause",
    } <= ids


def test_builtin_architecture_skill_understands_natural_repository_request(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        "Please help me understand this repository",
        {"read_file", "grep", "glob"},
    )
    assert result.selected_ids == ["repository.architecture-analysis"]
    assert result.signature.actions == {"understand"}


def test_excluding_review_does_not_activate_an_action_mismatched_skill(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        "Do not use $coding.code-review to inspect these code changes",
        {"read_file", "grep", "glob", "bash", "write_file", "edit_file", "edit_ast"},
    )
    assert result.selected_ids == []
    assert result.decision == "abstain"
    assert any("disabled for this turn" in reason for reason in result.rejected)
    assert "coding.safe-refactor" not in {
        candidate.skill.manifest.id for candidate in result.candidates
    }


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Add export support to this command", "coding.feature-implementation"),
        ("This command crashes on empty input; find and fix the bug", "coding.bug-fix"),
        ("Add unit tests for the parser edge cases", "testing.test-generation"),
        ("Update the README for the new configuration", "docs.technical-documentation"),
        ("Audit this authentication flow for security vulnerabilities", "security.code-audit"),
        ("Explain this repository architecture and execution flow", "repository.architecture-analysis"),
        ("Design a versioned REST API for project search", "coding.api-design"),
        ("Migrate clients from API v1 to v2 without downtime", "coding.api-migration"),
        ("Upgrade Pydantic to the next major version", "maintenance.dependency-upgrade"),
        ("Profile and reduce the parser latency", "quality.performance-optimization"),
        ("Find the race condition causing duplicate writes", "debugging.concurrency"),
        ("Find why this environment variable is ignored", "debugging.configuration"),
        (
            "Improve timeout handling and retry behavior in this client",
            "quality.error-handling-hardening",
        ),
        (
            "Remove unused modules and dependencies from this package",
            "maintenance.dead-code-cleanup",
        ),
        (
            "Check whether this release breaks existing API clients",
            "quality.backward-compatibility",
        ),
        ("Design commands and flags for the new project manager", "coding.cli-design"),
        (
            "This test fails intermittently in CI; find the source of flakiness",
            "testing.flaky-test-debug",
        ),
        (
            "Add integration tests for the API and database transaction",
            "testing.integration-test-generation",
        ),
        (
            "Analyze the highest-risk gaps in our test coverage",
            "testing.coverage-analysis",
        ),
        (
            "The GitHub Actions build fails only in CI; diagnose it",
            "devops.ci-cd-debug",
        ),
        ("Create a production Dockerfile for this service", "devops.containerization"),
        (
            "The service is unhealthy after deployment; find the cause",
            "devops.deployment-troubleshooting",
        ),
        ("Prepare version 2.1.0 for release", "release.preparation"),
        ("Build and publish this Python package to PyPI", "release.package-publishing"),
        (
            "Add a zero-downtime migration for this database column",
            "data.database-migration",
        ),
        (
            "Add validation and error reporting for imported CSV records",
            "data.validation",
        ),
        (
            "Audit this repository for exposed credentials and unsafe secret logging",
            "security.secrets-audit",
        ),
        (
            "Audit our locked dependencies for security and provenance risk",
            "security.dependency-audit",
        ),
        (
            "Add structured logs and latency metrics to this request path",
            "reliability.logging-observability",
        ),
        ("Write a migration guide from API v1 to v2", "docs.migration-guide"),
        (
            "Write release notes from the changes in this version",
            "docs.changelog-release-notes",
        ),
        (
            "Create an RCA from these incident logs and timeline",
            "reliability.incident-root-cause",
        ),
        ("设计一个支持分页的项目查询 API", "coding.api-design"),
        ("将调用方从 API v1 平滑迁移到 v2", "coding.api-migration"),
        ("升级项目中的 Pydantic 依赖", "maintenance.dependency-upgrade"),
        ("分析并降低解析器的性能延迟", "quality.performance-optimization"),
        ("排查导致重复写入的并发竞态", "debugging.concurrency"),
        ("排查环境变量为什么没有生效", "debugging.configuration"),
        ("GitHub Actions 在 CI 中构建失败", "devops.ci-cd-debug"),
        ("为这个服务创建生产 Dockerfile", "devops.containerization"),
        ("编写数据库字段零停机迁移", "data.database-migration"),
        ("检查仓库是否存在密钥泄露", "security.secrets-audit"),
        ("编写从 API v1 到 v2 的迁移指南", "docs.migration-guide"),
        ("根据日志进行生产故障复盘", "reliability.incident-root-cause"),
    ],
)
def test_new_builtin_skills_route_for_distinct_intents(tmp_path, query, expected):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        query,
        {"read_file", "grep", "glob", "bash", "write_file", "edit_file", "edit_ast"},
    )
    assert result.selected_ids == [expected]


def test_router_drops_weak_automatic_candidate_beside_clear_winner(tmp_path):
    _write_skill(tmp_path, "test.strong")
    _write_skill(
        tmp_path,
        "test.weak",
        name="Generic helper",
        summary="Handle a generic command safely.",
        tags=["command"],
        aliases=[],
        intents=[],
        examples={"positive": [], "negative": []},
    )
    result = SkillRouter(_registry(tmp_path), max_active=2).route(
        "the alpha deployment failed after release"
    )
    assert result.selected_ids == ["test.strong"]


def test_explicit_skill_suppresses_unrelated_fallback_after_constraint_rejection(tmp_path):
    _write_skill(tmp_path, "test.explicit", exclusive_group="occupied")
    _write_skill(tmp_path, "test.blocked", exclusive_group="occupied")
    _write_skill(tmp_path, "test.fallback")
    registry = _registry(tmp_path)
    candidates = [
        SkillCandidate(
            skill=registry.get("test.explicit"),
            score=10.0,
            explicit=True,
        ),
        SkillCandidate(
            skill=registry.get("test.blocked"),
            score=0.8,
        ),
        SkillCandidate(
            skill=registry.get("test.fallback"),
            score=0.3,
        ),
    ]
    rejected = []

    selected = SkillRouter(registry, max_active=3)._rank_and_select(candidates, rejected)

    assert [candidate.skill.manifest.id for candidate in selected] == ["test.explicit"]
    assert any("exclusive group" in reason for reason in rejected)


def test_router_ignores_language_names_inside_file_paths(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    manager.pin("coding.code-review")
    result = manager.route(
        "只读审查 D:\\develop\\Project\\Python\\CoreCoder\\corecoder\\skills\\router.py，检查边界条件。",
        {"read_file", "grep", "glob", "bash"},
    )
    assert result.selected_ids == ["coding.code-review"]


def test_router_strips_relative_paths_adjacent_to_cjk_text(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    manager.pin("coding.code-review")
    result = manager.route(
        "只读检查corecoder/Python/test_debug.py这个文件的边界条件。",
        {"read_file", "grep", "glob", "bash"},
    )
    assert result.selected_ids == ["coding.code-review"]


@pytest.mark.parametrize(
    "query",
    [
        "检查 我/的/文件/分析报告 内容",
        "检查 '我/的/文件/分析报告' 内容",
        "我/的/文件/分析报告",
    ],
)
def test_router_strips_delimited_unicode_relative_paths(tmp_path, query):
    _write_skill(
        tmp_path,
        "test.report",
        name="Report analyzer",
        summary="Handle a specialized report workflow.",
        tags=["分析报告"],
        aliases=[],
        intents=["分析报告"],
        examples={"positive": [], "negative": []},
    )
    result = SkillRouter(_registry(tmp_path)).route(query)
    assert result.selected == []


def test_router_matches_separator_alias_with_space_separated_query(tmp_path):
    _write_skill(tmp_path, name="Deployment helper", aliases=["alpha-workflow"])
    result = SkillRouter(_registry(tmp_path)).route("run the alpha workflow")
    assert result.selected_ids == ["test.alpha"]
    assert "name or alias matched" in result.candidates[0].reasons


def test_router_still_matches_language_in_task_text(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        "The Python pytest suite is failing; find the root cause",
        {"read_file", "grep", "glob", "bash"},
    )
    assert result.selected_ids == ["python.test-debug"]


def test_parent_skill_forbidden_tools_are_removed_from_executor(tmp_path):
    _write_skill(
        tmp_path,
        tools={"required": ["read_file"], "recommended": [], "forbidden": ["bash"]},
    )
    registry = _registry(tmp_path)
    manager = SkillManager(registry, SkillRouter(registry))
    agent = Agent(
        llm=_CaptureLLM(),
        tools=[get_tool("read_file"), get_tool("bash"), get_tool("write_file")],
        skills=manager,
        replay=False,
    )

    agent._load_skill_context("fix alpha deployment")
    child_tools = {tool.name for tool in agent._tools_for_role(AgentRole.EXECUTOR)}
    assert "bash" not in child_tools
    assert "read_file" in child_tools


def test_all_builtin_skills_are_schema_v2_with_a_clean_catalog(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    builtins = [
        skill for skill in manager.registry.all() if skill.scope == "builtin"
    ]
    assert len(builtins) >= 35
    not_v2 = [
        skill.manifest.id
        for skill in builtins
        if skill.manifest.schema_version < 2
    ]
    assert not_v2 == []
    # Structured ranking is only usable when every skill declares a signature.
    assert all(skill.manifest.signature.actions for skill in builtins)
    assert manager.router.catalog.issues == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("fixing a crash when the config file is missing", "coding.bug-fix"),
        ("the tests are failing after my refactor", "python.test-debug"),
        ("removing the dead code from the legacy module", "maintenance.dead-code-cleanup"),
        ("writing integration tests for the payment service", "testing.integration-test-generation"),
        ("upgrading all dependencies to their latest versions", "maintenance.dependency-upgrade"),
        ("scanning the repo for leaked API keys", "security.secrets-audit"),
        ("the container keeps crashing in production", "devops.deployment-troubleshooting"),
        ("documenting the new configuration options", "docs.technical-documentation"),
    ],
)
def test_inflected_verbs_route_like_their_base_forms(tmp_path, query, expected):
    """The tokenizer does no stemming, so manifests must carry the inflections."""
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        query,
        {"read_file", "grep", "glob", "bash", "write_file", "edit_file", "edit_ast"},
    )
    assert result.selected_ids == [expected]
    assert result.decision == "auto"


def test_language_name_alone_does_not_activate_a_skill(tmp_path):
    """A knowledge question that merely names Python is not a test-debugging task."""
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        "What does the walrus operator do in Python",
        {"read_file", "grep", "glob", "bash", "write_file", "edit_file", "edit_ast"},
    )
    assert result.selected_ids == []
    assert result.decision == "abstain"


def test_security_focused_review_routes_to_the_specialist(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        "reviewing the pull request for security issues",
        {"read_file", "grep", "glob"},
    )
    assert result.selected_ids == ["security.code-audit"]
    assert result.decision == "auto"


def test_excluding_a_skill_also_excludes_its_declared_substitute(tmp_path):
    """An exclusion must not be silently satisfied by an adjacent capability."""
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        "Do not use $coding.code-review to check whether this release breaks API clients",
        {"read_file", "grep", "glob", "bash", "write_file", "edit_file", "edit_ast"},
    )
    assert "quality.backward-compatibility" not in result.selected_ids
    assert any(
        "declared substitute for a skill the user excluded" in reason
        for reason in result.rejected
    )


def test_explicit_skill_survives_a_conflicting_exclusion(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    result = manager.route(
        "Do not use $coding.code-review; use $security.code-audit instead",
        {"read_file", "grep", "glob"},
    )
    assert result.selected_ids == ["security.code-audit"]
    assert result.decision == "explicit"


def test_catalog_flags_composition_that_contradicts_a_conflict(tmp_path):
    _write_skill(
        tmp_path,
        "test.alpha",
        relations={"dependencies": [], "composes_with": ["test.beta"], "supersedes": []},
    )
    _write_skill(tmp_path, "test.beta", conflicts_with=["test.alpha"])
    issues = SkillRouter(_registry(tmp_path)).catalog.issues
    assert any(
        issue.code == "contradictory-relation" and set(issue.skill_ids) == {"test.alpha", "test.beta"}
        for issue in issues
    )


@pytest.mark.asyncio
async def test_agent_returns_router_clarification_without_calling_llm_or_tools(tmp_path):
    _write_skill(tmp_path, "test.alpha", schema_version=2, exclusive_group="deploy")
    _write_skill(tmp_path, "test.beta", schema_version=2, exclusive_group="deploy")
    registry = _registry(tmp_path)
    manager = SkillManager(
        registry,
        SkillRouter(registry, ambiguity_margin=1.0),
    )
    llm = _CaptureLLM()
    agent = Agent(llm=llm, tools=[get_tool("read_file")], skills=manager, replay=False)

    answer = await agent.chat("fix alpha deployment")

    assert answer == manager.last_result.clarification
    assert manager.last_result.decision == "clarify"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_high_risk_skill_blocks_state_change_without_confirmation(tmp_path):
    _write_skill(
        tmp_path,
        schema_version=2,
        routing={"allow_implicit": True, "risk": "high", "rollout_percent": 100},
    )
    registry = _registry(tmp_path)
    manager = SkillManager(registry, SkillRouter(registry))
    agent = Agent(
        llm=_CaptureLLM(),
        tools=[get_tool("write_file")],
        skills=manager,
        replay=False,
    )
    agent._load_skill_context("Use $test.alpha")

    class _TC:
        def __init__(self):
            self.name = "write_file"
            self.arguments = {
                "file_path": str(tmp_path / "blocked.txt"),
                "content": "blocked",
            }

    result, _, success = await agent._exec_tool(_TC())
    assert not success
    assert "high risk" in result
    assert not (tmp_path / "blocked.txt").exists()


def test_routing_context_recall_and_required_context_gate(tmp_path):
    _write_skill(
        tmp_path,
        schema_version=2,
        signature={
            "domains": ["document"],
            "actions": ["review"],
            "objects": ["contract"],
            "artifacts": ["docx"],
            "outputs": ["reviewed contract"],
            "constraints": [],
            "contexts": ["attachment"],
        },
        requires={"context_any": ["attachment"], "inputs_any": ["contract_file"]},
    )
    router = SkillRouter(_registry(tmp_path))

    missing = router.route("review the contract")
    present = router.route(
        "review the contract",
        context=RoutingContext(
            attachments=["contract.docx"],
            inputs={"contract_file"},
        ),
    )

    assert missing.selected_ids == []
    assert present.selected_ids == ["test.alpha"]
    assert present.signature.artifacts >= {"docx"}
    assert present.signature.contexts >= {"attachment", "contract_file"}


def test_matching_resource_mode_loads_only_its_reference_and_paths(tmp_path):
    directory = _write_skill(
        tmp_path,
        schema_version=2,
        resource_modes=[
            {
                "id": "redline",
                "when": ["tracked changes"],
                "references": ["references/redlining.md"],
                "scripts": ["scripts/redline.py"],
                "assets": [],
            },
            {
                "id": "plain",
                "when": ["plain rewrite"],
                "references": ["references/plain.md"],
                "scripts": [],
                "assets": [],
            },
        ],
    )
    (directory / "references").mkdir()
    (directory / "scripts").mkdir()
    (directory / "references" / "redlining.md").write_text("REDLINE RULES", encoding="utf-8")
    (directory / "references" / "plain.md").write_text("PLAIN RULES", encoding="utf-8")
    (directory / "scripts" / "redline.py").write_text("print('safe')", encoding="utf-8")

    result = SkillRouter(_registry(tmp_path)).route(
        "Use $test.alpha with tracked changes"
    )
    assert "REDLINE RULES" in result.prompt
    assert "redline.py" in result.prompt
    assert "PLAIN RULES" not in result.prompt


def test_semantic_recall_can_supply_a_candidate_without_lexical_overlap(tmp_path):
    _write_skill(tmp_path, schema_version=2)
    router = SkillRouter(
        _registry(tmp_path),
        semantic_scorer=lambda _query, skill: 0.95 if skill.manifest.id == "test.alpha" else 0,
    )
    result = router.route("completely unrelated vocabulary")
    assert result.selected_ids == ["test.alpha"]
    assert any("semantic recall" in reason for reason in result.candidates[0].reasons)


def test_superseded_skill_routes_to_active_successor(tmp_path):
    _write_skill(tmp_path, "test.old")
    _write_skill(
        tmp_path,
        "test.new",
        schema_version=2,
        name="Replacement",
        summary="Replacement workflow.",
        tags=["replacement"],
        intents=["replacement workflow"],
        examples={"positive": [], "negative": []},
        relations={"dependencies": [], "composes_with": [], "supersedes": ["test.old"]},
    )
    result = SkillRouter(_registry(tmp_path)).route("fix alpha deployment")
    assert result.selected_ids == ["test.new"]
    assert any("superseded by test.new" in reason for reason in result.rejected)


def test_project_skill_lifecycle_transition_is_validated_and_audited(tmp_path):
    _write_skill(tmp_path)
    registry = _registry(tmp_path)
    manager = SkillManager(registry, SkillRouter(registry))

    manager.transition("test.alpha", "deprecated", "replaced after regression review")

    skill = manager.registry.get("test.alpha")
    assert skill.manifest.status == "deprecated"
    assert skill.manifest.lifecycle.previous_status == "active"
    assert "regression review" in skill.manifest.lifecycle.reason
    assert (skill.path / ".lifecycle.jsonl").is_file()
    with pytest.raises(ValueError, match="invalid skill lifecycle transition"):
        manager.transition("test.alpha", "candidate", "invalid backwards jump")


def test_evaluation_reports_confidence_ux_latency_and_success_metrics(tmp_path):
    _write_skill(tmp_path)
    metrics = evaluate_router(
        SkillRouter(_registry(tmp_path)),
        [
            RoutingCase(
                query="fix alpha deployment",
                expected_skill="test.alpha",
                task_succeeded=True,
                high_risk_confirmed=True,
            ),
            RoutingCase(
                query="tell me a joke",
                expected_decision="abstain",
                user_overrode=True,
                task_succeeded=False,
            ),
        ],
    )
    assert metrics.precision_at_1 == 1.0
    assert metrics.overall_accuracy == 1.0
    assert metrics.user_override_rate == 0.5
    assert metrics.task_success_rate == 0.5
    assert metrics.high_risk_confirmation_rate == 1.0
    assert metrics.p95_routing_latency_ms >= 0
    assert metrics.average_top1_top2_margin > 0


def test_v2_confidence_thresholds_reject_or_clarify_weak_matches(tmp_path):
    manager = SkillManager.create(project_path=tmp_path, user_dir=tmp_path / "user")
    tools = {"read_file", "grep", "glob", "bash", "write_file", "edit_file", "edit_ast"}
    ambiguous = manager.route("API", tools)
    weak = manager.route("Tell me a joke about test coverage", tools)

    assert ambiguous.decision == "clarify"
    assert ambiguous.selected_ids == []
    assert weak.decision == "abstain"


def test_required_permissions_are_a_hard_gate_even_for_explicit_skill(tmp_path):
    _write_skill(
        tmp_path,
        schema_version=2,
        requires={"context_any": [], "inputs_any": [], "permissions": ["drive.write"]},
    )
    router = SkillRouter(_registry(tmp_path))
    denied = router.route("Use $test.alpha")
    allowed = router.route(
        "Use $test.alpha",
        context=RoutingContext(granted_permissions={"drive.write"}),
    )
    assert denied.selected_ids == []
    assert any("missing required permissions" in reason for reason in denied.rejected)
    assert allowed.selected_ids == ["test.alpha"]


def test_dependency_version_mismatch_blocks_activation_and_is_audited(tmp_path):
    _write_skill(tmp_path, "test.atomic", version="2.0.0")
    _write_skill(
        tmp_path,
        "test.workflow",
        relations={
            "dependencies": ["test.atomic"],
            "dependency_versions": {"test.atomic": "1.0.0"},
            "composes_with": [],
            "supersedes": [],
        },
    )
    router = SkillRouter(_registry(tmp_path))
    result = router.route("Use $test.workflow")
    assert result.selected_ids == []
    assert any("requires version 1.0.0" in reason for reason in result.rejected)
    assert any(issue.code == "dependency-version-mismatch" for issue in router.catalog.issues)


def test_historical_failure_penalty_can_change_automatic_ranking(tmp_path):
    _write_skill(tmp_path, "test.alpha", priority=10)
    _write_skill(tmp_path, "test.beta")
    baseline = SkillRouter(_registry(tmp_path)).route("fix alpha deployment")
    penalized = SkillRouter(
        _registry(tmp_path),
        failure_penalties={"test.alpha": 0.3},
    ).route("fix alpha deployment")
    assert baseline.selected_ids[0] == "test.alpha"
    assert penalized.selected_ids[0] == "test.beta"
    alpha = next(
        candidate
        for candidate in penalized.candidates
        if candidate.skill.manifest.id == "test.alpha"
    )
    assert any("historical failure penalty" in reason for reason in alpha.reasons)

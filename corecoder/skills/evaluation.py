"""Offline routing evaluation for skill catalog changes."""

from __future__ import annotations

import math
import time

from pydantic import BaseModel, Field

from .models import RouteDecision, RoutingContext
from .router import SkillRouter


class RoutingCase(BaseModel):
    """One golden routing request, including the valid no-skill outcome."""

    query: str = Field(min_length=1)
    expected_skill: str | None = None
    expected_skills: set[str] = Field(default_factory=set)
    expected_decision: RouteDecision | None = None
    available_tools: set[str] | None = None
    context: RoutingContext | None = None
    user_overrode: bool = False
    task_succeeded: bool | None = None
    high_risk_confirmed: bool | None = None

    def expected(self) -> set[str]:
        values = set(self.expected_skills)
        if self.expected_skill:
            values.add(self.expected_skill)
        return values


class RoutingMetrics(BaseModel):
    """High-signal metrics for precision, abstention, cost, and UX."""

    total: int = 0
    correct_top1: int = 0
    activations: int = 0
    false_activations: int = 0
    missed_skills: int = 0
    wrong_skills: int = 0
    clarifications: int = 0
    precision_at_1: float = 0.0
    invocation_precision: float = 0.0
    false_activation_rate: float = 0.0
    missed_skill_rate: float = 0.0
    clarification_rate: float = 0.0
    overall_accuracy: float = 0.0
    average_candidates: float = 0.0
    average_loaded_tokens: float = 0.0
    average_top1_top2_margin: float = 0.0
    user_override_rate: float = 0.0
    task_success_rate: float = 0.0
    p95_routing_latency_ms: float = 0.0
    high_risk_confirmation_rate: float = 0.0
    shadow_observations: int = 0
    shadow_would_change_rate: float = 0.0


def evaluate_router(router: SkillRouter, cases: list[RoutingCase]) -> RoutingMetrics:
    """Evaluate observable routing outcomes without calling an LLM or tools."""
    if not cases:
        return RoutingMetrics()

    correct = activations = false_activations = missed = wrong = clarifications = 0
    positive_cases = negative_cases = total_candidates = prompt_chars = 0
    correct_positive = overrides = completed_tasks = successful_tasks = 0
    high_risk_cases = high_risk_confirmations = 0
    shadow_observations = shadow_would_change = 0
    margins: list[float] = []
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        result = router.route(
            case.query,
            available_tools=case.available_tools,
            context=case.context,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        predicted = result.selected_ids[0] if result.selected_ids else None
        expected = case.expected()
        total_candidates += len(result.candidates)
        prompt_chars += len(result.prompt)
        margins.append(result.margin)
        shadow_candidates = [candidate for candidate in result.candidates if candidate.shadow]
        if shadow_candidates:
            shadow_observations += 1
            active_score = next(
                (candidate.score for candidate in result.candidates if not candidate.shadow),
                0.0,
            )
            shadow_would_change += int(shadow_candidates[0].score > active_score)
        clarifications += int(result.needs_clarification)
        activations += int(predicted is not None)
        overrides += int(case.user_overrode)
        if case.task_succeeded is not None:
            completed_tasks += 1
            successful_tasks += int(case.task_succeeded)
        if case.high_risk_confirmed is not None:
            high_risk_cases += 1
            high_risk_confirmations += int(case.high_risk_confirmed)

        if not expected:
            negative_cases += 1
            false_activations += int(predicted is not None)
            case_correct = predicted is None and not result.needs_clarification
            if case.expected_decision is not None:
                case_correct = case_correct and result.decision == case.expected_decision
            correct += int(case_correct)
            continue

        positive_cases += 1
        if predicted in expected:
            case_correct = (
                case.expected_decision is None
                or result.decision == case.expected_decision
            )
            correct += int(case_correct)
            correct_positive += int(case_correct)
        elif predicted is None:
            missed += 1
        else:
            wrong += 1

    total = len(cases)
    correct_activations = activations - false_activations - wrong
    return RoutingMetrics(
        total=total,
        correct_top1=correct,
        activations=activations,
        false_activations=false_activations,
        missed_skills=missed,
        wrong_skills=wrong,
        clarifications=clarifications,
        precision_at_1=(
            round(correct_positive / positive_cases, 4) if positive_cases else 0.0
        ),
        invocation_precision=round(correct_activations / activations, 4) if activations else 0.0,
        false_activation_rate=(
            round(false_activations / negative_cases, 4) if negative_cases else 0.0
        ),
        missed_skill_rate=round(missed / positive_cases, 4) if positive_cases else 0.0,
        clarification_rate=round(clarifications / total, 4),
        overall_accuracy=round(correct / total, 4),
        average_candidates=round(total_candidates / total, 2),
        average_loaded_tokens=round(prompt_chars / total / 4, 2),
        average_top1_top2_margin=round(sum(margins) / total, 4),
        user_override_rate=round(overrides / total, 4),
        task_success_rate=(
            round(successful_tasks / completed_tasks, 4) if completed_tasks else 0.0
        ),
        p95_routing_latency_ms=round(_percentile(latencies, 0.95), 3),
        high_risk_confirmation_rate=(
            round(high_risk_confirmations / high_risk_cases, 4)
            if high_risk_cases
            else 0.0
        ),
        shadow_observations=shadow_observations,
        shadow_would_change_rate=(
            round(shadow_would_change / shadow_observations, 4)
            if shadow_observations
            else 0.0
        ),
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]

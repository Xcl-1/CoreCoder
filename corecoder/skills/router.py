"""Catalog recall, boundary-aware reranking, and skill activation."""

from __future__ import annotations

import hashlib
import math
import re

from .catalog import (
    SemanticScorer,
    SkillCatalog,
    expanded_tokens,
    phrase_matches,
    token_similarity,
)
from .loader import SkillLoadError, load_instructions, load_mode_resources
from .models import RouteResult, RoutingContext, SkillCandidate, TaskSignature
from .registry import SkillRegistry

_EXPLICIT_RE = re.compile(
    r"(?<![\w$])\$([a-z0-9][a-z0-9._-]{1,79})(?![a-z0-9._:-])"
)
_NEGATED_EXPLICIT_RE = re.compile(
    r"(?:do\s+not\s+use|don't\s+use|without|exclude|不要使用|不要用|别用|禁用)\s*"
    r"\$([a-z0-9][a-z0-9._-]{1,79})",
    re.IGNORECASE,
)
_WINDOWS_PATH_RE = re.compile(r"[a-z]:[\\/][^\s\"'`，。；;,]+", re.IGNORECASE)
_SCOPE_BONUS = {"builtin": 0.0, "user": 0.03, "project": 0.06, "custom": 0.02}
_RELATIVE_SELECTION_RATIO = 0.40
_DELIMITED_PATH_RE = re.compile(
    r"(?<![^\s\"'`])[^\\/\s\"'`，。；;,]+"
    r"(?:[\\/][^\\/\s\"'`，。；;,]+)+"
)
_RELATIVE_PATH_RE = re.compile(
    r"(?<![a-z0-9_.@~+-])(?:[a-z0-9_.@~+-]+[\\/])+[a-z0-9_.@~+-]+",
    re.IGNORECASE,
)
_SIGNATURE_WEIGHTS = {
    "domains": 0.08,
    "actions": 0.14,
    "objects": 0.10,
    "artifacts": 0.12,
    "outputs": 0.12,
    "constraints": 0.06,
    "contexts": 0.10,
}
_ACTION_MISMATCH_PENALTY = 0.18
_CONFIDENCE_SCALE = 5.0


class SkillRouter:
    def __init__(
        self,
        registry: SkillRegistry,
        top_k: int = 10,
        max_active: int = 3,
        max_prompt_chars: int = 6000,
        min_score: float = 0.24,
        auto_confidence: float = 0.82,
        clarify_confidence: float = 0.65,
        ambiguity_margin: float = 0.12,
        semantic_scorer: SemanticScorer | None = None,
        failure_penalties: dict[str, float] | None = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_active < 1:
            raise ValueError("max_active must be at least 1")
        if max_prompt_chars < 1:
            raise ValueError("max_prompt_chars must be at least 1")
        if not math.isfinite(min_score) or min_score < 0:
            raise ValueError("min_score must be a finite non-negative number")
        if not math.isfinite(ambiguity_margin) or not 0 <= ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin must be between 0 and 1")
        if (
            not math.isfinite(auto_confidence)
            or not math.isfinite(clarify_confidence)
            or not 0 <= clarify_confidence <= auto_confidence <= 1
        ):
            raise ValueError(
                "confidence thresholds must satisfy 0 <= clarify_confidence "
                "<= auto_confidence <= 1"
            )
        self.registry = registry
        self.top_k = top_k
        self.max_active = max_active
        self.max_prompt_chars = max_prompt_chars
        self.min_score = min_score
        self.auto_confidence = auto_confidence
        self.clarify_confidence = clarify_confidence
        self.ambiguity_margin = ambiguity_margin
        self.failure_penalties: dict[str, float] = {}
        for skill_id, penalty in (failure_penalties or {}).items():
            value = float(penalty)
            if not math.isfinite(value) or value < 0:
                raise ValueError("failure penalties must be finite non-negative numbers")
            self.failure_penalties[skill_id] = min(0.3, value)
        self.catalog = SkillCatalog(registry, semantic_scorer=semantic_scorer)

    def refresh_catalog(self) -> None:
        self.catalog.refresh()

    def route(
        self,
        query: str,
        available_tools: set[str] | None = None,
        pinned: set[str] | None = None,
        context: RoutingContext | dict | None = None,
    ) -> RouteResult:
        routing_context = RoutingContext.model_validate(context or {})
        routing_query = self._without_paths(query)
        excluded_ids = self._excluded_ids(routing_query)
        explicit_ids = (self._explicit_ids(routing_query) | set(pinned or ())) - excluded_ids
        rejected = [f"{skill_id}: disabled for this turn by the user" for skill_id in sorted(excluded_ids)]
        # A skill that declares a conflict with an excluded skill is a substitute
        # for the same request; silently offering it would defeat the exclusion.
        substitutes = self._declared_substitutes(excluded_ids) - excluded_ids - explicit_ids
        candidates, recall_rejected, signature = self._recall(
            routing_query,
            explicit_ids,
            available_tools,
            limit=self.top_k,
            context=routing_context,
        )
        kept: list[SkillCandidate] = []
        for candidate in candidates:
            skill_id = candidate.skill.manifest.id
            if skill_id in excluded_ids:
                continue
            if skill_id in substitutes:
                rejected.append(
                    f"{skill_id}: declared substitute for a skill the user excluded this turn"
                )
                continue
            kept.append(candidate)
        candidates = kept
        rejected.extend(recall_rejected)

        explicit_unavailable = bool(
            explicit_ids and not any(candidate.explicit for candidate in candidates)
        )
        preliminary_decision, clarification = self._automatic_decision(candidates)
        if explicit_unavailable:
            selected = []
            decision = "abstain"
            prompt = ""
        elif preliminary_decision == "clarify":
            selected: list[SkillCandidate] = []
            decision = "clarify"
            prompt = self._render_clarification(clarification)
        elif preliminary_decision == "abstain":
            selected = []
            decision = "abstain"
            prompt = ""
        else:
            selected = self._rank_and_select(candidates, rejected)
            prompt = self._render(selected, rejected, routing_query)
            if any(candidate.explicit for candidate in selected):
                decision = "explicit"
            elif selected:
                decision = "auto"
            else:
                decision = "abstain"

        confidence = selected[0].confidence if selected else self._top_confidence(candidates)
        return RouteResult(
            query=query,
            candidates=candidates,
            selected=selected,
            rejected=rejected,
            prompt=prompt,
            decision=decision,
            confidence=confidence,
            margin=self._confidence_margin(candidates),
            clarification=clarification,
            signature=signature,
        )

    def search(self, query: str, limit: int = 20) -> list[SkillCandidate]:
        if limit <= 0:
            return []
        candidates, _, _ = self._recall(self._without_paths(query), set(), None, limit=limit)
        return [candidate for candidate in candidates if not candidate.shadow][:limit]

    def _recall(
        self,
        query: str,
        explicit_ids: set[str],
        available_tools: set[str] | None,
        limit: int,
        context: RoutingContext | None = None,
    ) -> tuple[list[SkillCandidate], list[str], TaskSignature]:
        """Run cheap catalog recall followed by metadata-only reranking."""
        # Recall beyond the public top-k so hard gates (tool availability,
        # boundaries, rollout) cannot starve otherwise valid candidates.
        recall_limit = max(limit * 3, limit + 10)
        signature = self.catalog.signature_for(query, context)
        context_tokens = {
            token
            for values in signature.dimensions().values()
            for value in values
            for token in expanded_tokens(value)
        }
        matches = self.catalog.recall(
            query,
            explicit_ids,
            limit=recall_limit,
            context_tokens=context_tokens,
        )
        query_tokens = expanded_tokens(query)
        normalized_query = query.lower()
        candidates: list[SkillCandidate] = []
        rejected: list[str] = []

        for match in matches:
            skill = match.skill
            manifest = skill.manifest
            explicit = manifest.id in explicit_ids
            shadow = manifest.status == "shadow" and not explicit

            successor_id = self.catalog.superseded_by.get(manifest.id)
            if successor_id is not None and not explicit:
                rejected.append(f"{manifest.id}: superseded by {successor_id}")
                continue

            if manifest.status not in {"active", "canary", "shadow"}:
                if explicit:
                    rejected.append(f"{manifest.id}: status is {manifest.status}")
                continue
            if manifest.status == "shadow" and explicit:
                rejected.append(f"{manifest.id}: status is shadow")
                continue
            if (
                manifest.status == "canary"
                and not explicit
                and not self._in_canary(
                    (context.routing_key if context else "") or query,
                    manifest.id,
                    manifest.routing.rollout_percent,
                )
            ):
                continue
            if not manifest.routing.allow_implicit and not explicit:
                rejected.append(f"{manifest.id}: implicit invocation is disabled")
                continue
            missing = (
                set(manifest.tools.required) - available_tools
                if available_tools is not None
                else set()
            )
            if missing:
                rejected.append(
                    f"{manifest.id}: missing required tools {', '.join(sorted(missing))}"
                )
                continue
            signals = (context or RoutingContext()).signals()
            if (
                manifest.requires.context_any
                and not set(manifest.requires.context_any) & signals
            ):
                rejected.append(f"{manifest.id}: required context is missing")
                continue
            if (
                manifest.requires.inputs_any
                and not set(manifest.requires.inputs_any) & set((context or RoutingContext()).inputs)
            ):
                rejected.append(f"{manifest.id}: required input is missing")
                continue
            missing_permissions = set(manifest.requires.permissions) - set(
                (context or RoutingContext()).granted_permissions
            )
            if missing_permissions:
                rejected.append(
                    f"{manifest.id}: missing required permissions "
                    f"{', '.join(sorted(missing_permissions))}"
                )
                continue
            dependency_error = self._dependency_error(manifest.id, available_tools)
            if dependency_error:
                rejected.append(dependency_error)
                continue

            negative = next(
                (
                    rule for rule in manifest.not_when
                    if phrase_matches(rule, query_tokens, normalized_query)
                ),
                None,
            )
            if negative and not explicit:
                rejected.append(f"{manifest.id}: not_when matched {negative!r}")
                continue

            entry = self.catalog.entries[manifest.id]
            negative_score = max(
                (token_similarity(query_tokens, tokens) for tokens in entry.negative_tokens),
                default=0.0,
            )
            hard_negative_score = max(
                (token_similarity(query_tokens, tokens) for tokens in entry.hard_negative_tokens),
                default=0.0,
            )
            if negative_score >= 0.72 and not explicit:
                rejected.append(
                    f"{manifest.id}: negative example matched ({negative_score:.3f})"
                )
                continue
            if hard_negative_score >= 0.58 and not explicit:
                rejected.append(
                    f"{manifest.id}: hard negative matched ({hard_negative_score:.3f})"
                )
                continue

            contrastive = next(
                (
                    (example, token_similarity(query_tokens, expanded_tokens(example.query)))
                    for example in manifest.examples.contrastive
                    if example.expected_skill != manifest.id
                    and token_similarity(query_tokens, expanded_tokens(example.query)) >= 0.72
                ),
                None,
            )
            if contrastive and not explicit:
                example, similarity = contrastive
                rejected.append(
                    f"{manifest.id}: contrastive example routes to "
                    f"{example.expected_skill} ({similarity:.3f})"
                )
                continue

            score = match.recall_score
            reasons = list(match.reasons)
            matched_dimensions: list[str] = []
            matched_tags = [
                tag for tag in manifest.tags
                if phrase_matches(tag, query_tokens, normalized_query)
            ]
            matched_intents = [
                intent for intent in manifest.intents
                if phrase_matches(intent, query_tokens, normalized_query)
            ]
            matched_aliases = [
                alias for alias in manifest.aliases
                if phrase_matches(alias, query_tokens, normalized_query)
            ]
            matched_boundaries = [
                rule for rule in manifest.applies_when
                if phrase_matches(rule, query_tokens, normalized_query)
            ]
            if matched_tags:
                score += min(0.24, 0.08 * len(matched_tags))
                reasons.append(f"tags: {', '.join(matched_tags[:3])}")
            if matched_intents:
                score += min(0.36, 0.18 * len(matched_intents))
                reasons.append(f"intents: {', '.join(matched_intents[:2])}")
            if matched_aliases or manifest.name.lower() in normalized_query:
                score += 0.2
                reasons.append("name or alias matched")
            if matched_boundaries:
                score += min(0.16, 0.08 * len(matched_boundaries))
                reasons.append("applicability boundary matched")

            positive_example_score = max(
                (token_similarity(query_tokens, tokens) for tokens in entry.example_tokens),
                default=0.0,
            )
            if positive_example_score:
                score += 0.25 * positive_example_score
                reasons.append(f"positive example {positive_example_score:.3f}")

            task_dimensions = signature.dimensions()
            for dimension, values in manifest.signature.dimensions().items():
                matched = set(values) & task_dimensions[dimension]
                if matched:
                    score += _SIGNATURE_WEIGHTS[dimension]
                    matched_dimensions.append(dimension)
                    reasons.append(f"{dimension}: {', '.join(sorted(matched)[:3])}")

            declared_actions = set(manifest.signature.actions)
            requested_actions = task_dimensions["actions"]
            if (
                not explicit
                and manifest.schema_version >= 2
                and declared_actions
                and requested_actions
                and declared_actions.isdisjoint(requested_actions)
            ):
                score -= _ACTION_MISMATCH_PENALTY
                reasons.append("structured action mismatch")

            if explicit:
                score += 10.0
                reasons.append("explicitly requested")
            score += _SCOPE_BONUS.get(skill.scope, 0.0)
            score += manifest.priority / 1000
            cost_penalty = min(0.04, manifest.token_budget / 300_000)
            if not explicit and cost_penalty:
                score -= cost_penalty
            failure_penalty = self.failure_penalties.get(manifest.id, 0.0)
            if not explicit and failure_penalty:
                score -= failure_penalty
                reasons.append(f"historical failure penalty {failure_penalty:.3f}")

            score = round(max(0.0, score), 4)
            confidence = 1.0 if explicit else round(1 - math.exp(-_CONFIDENCE_SCALE * score), 4)
            if explicit or shadow or score >= self.min_score / 2:
                candidates.append(SkillCandidate(
                    skill=skill,
                    score=score,
                    recall_score=round(match.recall_score, 4),
                    confidence=confidence,
                    reasons=reasons,
                    matched_dimensions=matched_dimensions,
                    explicit=explicit,
                    shadow=shadow,
                ))

        for explicit_id in sorted(explicit_ids):
            if self.registry.get(explicit_id) is None:
                rejected.append(f"{explicit_id}: explicitly requested skill was not found")
        candidates.sort(
            key=lambda item: (item.explicit, item.score, item.skill.manifest.priority),
            reverse=True,
        )
        if len(candidates) > limit:
            explicit = [candidate for candidate in candidates if candidate.explicit]
            remaining = max(0, limit - len(explicit))
            automatic = [candidate for candidate in candidates if not candidate.explicit]
            candidates = [*explicit, *automatic[:remaining]]
        return candidates, rejected, signature

    def _rank_and_select(
        self,
        candidates: list[SkillCandidate],
        rejected: list[str],
    ) -> list[SkillCandidate]:
        selected: list[SkillCandidate] = []
        best_eligible_automatic_score: float | None = None
        automatic_primary_id: str | None = None
        explicit_ids = {
            candidate.skill.manifest.id
            for candidate in candidates
            if candidate.explicit and not candidate.shadow
        }

        for candidate in candidates:
            manifest = candidate.skill.manifest
            if candidate.shadow:
                continue
            if not candidate.explicit and candidate.score < self.min_score:
                continue
            if len(selected) >= self.max_active:
                if candidate.explicit:
                    rejected.append(
                        f"{manifest.id}: maximum active skill count {self.max_active} reached"
                    )
                continue
            conflict = self._selection_conflict_reason(candidate, selected)
            if conflict:
                rejected.append(f"{manifest.id}: {conflict}")
                continue
            if not candidate.explicit:
                if explicit_ids and not any(
                    self._declared_composition(skill_id, manifest.id)
                    for skill_id in explicit_ids
                ):
                    continue
                if best_eligible_automatic_score is None:
                    best_eligible_automatic_score = candidate.score
                elif candidate.score < best_eligible_automatic_score * _RELATIVE_SELECTION_RATIO:
                    continue
                if automatic_primary_id and not self._declared_composition(
                    automatic_primary_id, manifest.id
                ):
                    continue

            dependencies, dependency_error = self._dependency_candidates(candidate, selected)
            if dependency_error:
                rejected.append(dependency_error)
                continue
            if len(selected) + 1 + len(dependencies) > self.max_active:
                rejected.append(
                    f"{manifest.id}: dependencies exceed maximum active skill count "
                    f"{self.max_active}"
                )
                continue
            batch = [candidate, *dependencies]
            batch_conflict = next(
                (
                    self._selection_conflict_reason(item, [*selected, *batch[:index]])
                    for index, item in enumerate(batch)
                    if self._selection_conflict_reason(item, [*selected, *batch[:index]])
                ),
                "",
            )
            if batch_conflict:
                rejected.append(f"{manifest.id}: dependency {batch_conflict}")
                continue

            selected.extend(batch)
            if not candidate.explicit and automatic_primary_id is None:
                automatic_primary_id = manifest.id

        return selected

    def _dependency_candidates(
        self,
        parent: SkillCandidate,
        selected: list[SkillCandidate],
    ) -> tuple[list[SkillCandidate], str]:
        selected_ids = {candidate.skill.manifest.id for candidate in selected}
        dependencies: list[SkillCandidate] = []
        visiting: set[str] = set()

        def visit(skill_id: str) -> str:
            if skill_id in selected_ids or any(
                candidate.skill.manifest.id == skill_id for candidate in dependencies
            ):
                return ""
            if skill_id in visiting:
                return f"{parent.skill.manifest.id}: cyclic skill dependency involving {skill_id}"
            skill = self.registry.get(skill_id)
            if skill is None or skill.manifest.status not in {"active", "canary"}:
                return f"{parent.skill.manifest.id}: unavailable dependency {skill_id}"
            visiting.add(skill_id)
            for nested in skill.manifest.relations.dependencies:
                error = visit(nested)
                if error:
                    return error
            visiting.remove(skill_id)
            dependencies.append(SkillCandidate(
                skill=skill,
                score=parent.score,
                recall_score=0.0,
                confidence=parent.confidence,
                reasons=[f"required by {parent.skill.manifest.id}"],
                explicit=parent.explicit,
            ))
            return ""

        for skill_id in parent.skill.manifest.relations.dependencies:
            error = visit(skill_id)
            if error:
                return [], error
        return dependencies, ""

    def _selection_conflict_reason(
        self,
        candidate: SkillCandidate,
        selected: list[SkillCandidate],
    ) -> str:
        manifest = candidate.skill.manifest
        candidate_required = set(manifest.tools.required)
        candidate_forbidden = set(manifest.tools.forbidden)
        for item in selected:
            other = item.skill.manifest
            if manifest.exclusive_group and manifest.exclusive_group == other.exclusive_group:
                return f"exclusive group {manifest.exclusive_group!r} already selected"
            if other.id in manifest.conflicts_with or manifest.id in other.conflicts_with:
                return "conflicts with a selected skill"
            if candidate_required & set(other.tools.forbidden):
                names = ", ".join(sorted(candidate_required & set(other.tools.forbidden)))
                return f"tool policy conflicts with selected skills ({names})"
            if candidate_forbidden & set(other.tools.required):
                names = ", ".join(sorted(candidate_forbidden & set(other.tools.required)))
                return f"tool policy conflicts with selected skills ({names})"
        return ""

    def _dependency_error(
        self,
        skill_id: str,
        available_tools: set[str] | None,
        visiting: set[str] | None = None,
    ) -> str:
        skill = self.registry.get(skill_id)
        if skill is None:
            return f"{skill_id}: skill was not found"
        visiting = set(visiting or ())
        if skill_id in visiting:
            return f"{skill_id}: cyclic skill dependency"
        visiting.add(skill_id)
        for dependency_id in skill.manifest.relations.dependencies:
            dependency = self.registry.get(dependency_id)
            if dependency is None or dependency.manifest.status not in {"active", "canary"}:
                return f"{skill_id}: unavailable dependency {dependency_id}"
            expected_version = skill.manifest.relations.dependency_versions.get(dependency_id)
            if expected_version and dependency.manifest.version != expected_version:
                return (
                    f"{skill_id}: dependency {dependency_id} requires version "
                    f"{expected_version}, found {dependency.manifest.version}"
                )
            if available_tools is not None:
                missing = set(dependency.manifest.tools.required) - available_tools
                if missing:
                    return (
                        f"{skill_id}: dependency {dependency_id} is missing required tools "
                        f"{', '.join(sorted(missing))}"
                    )
            error = self._dependency_error(dependency_id, available_tools, visiting)
            if error:
                return error
        return ""

    def _declared_composition(self, left_id: str, right_id: str) -> bool:
        left = self.registry.get(left_id)
        right = self.registry.get(right_id)
        if left is None or right is None:
            return False
        return (
            right_id in left.manifest.relations.composes_with
            or left_id in right.manifest.relations.composes_with
            or right_id in left.manifest.relations.dependencies
            or left_id in right.manifest.relations.dependencies
        )

    def _declared_substitutes(self, excluded_ids: set[str]) -> set[str]:
        """Skills that declare a conflict with an excluded skill, one level deep.

        Deliberately narrower than ``exclusive_group``: a whole group would also
        block unrelated skills that merely share an execution mode.
        """
        substitutes: set[str] = set()
        for excluded_id in excluded_ids:
            excluded = self.registry.get(excluded_id)
            if excluded is not None:
                substitutes.update(excluded.manifest.conflicts_with)
        for skill in self.registry.all(include_inactive=True):
            if excluded_ids & set(skill.manifest.conflicts_with):
                substitutes.add(skill.manifest.id)
        return substitutes

    def _automatic_decision(
        self,
        candidates: list[SkillCandidate],
    ) -> tuple[str, str]:
        """Choose auto/clarify/abstain before loading executable instructions."""
        if any(candidate.explicit and not candidate.shadow for candidate in candidates):
            return "auto", ""
        eligible = [
            candidate for candidate in candidates
            if not candidate.explicit
            and not candidate.shadow
            and candidate.score >= self.min_score
        ]
        if not eligible:
            return "abstain", ""
        first = eligible[0]
        # Schema-v1 manifests predate confidence-aware routing. Preserve their
        # established selection behavior until maintainers add explicit v2
        # boundaries and contrastive examples.
        if first.skill.manifest.schema_version < 2:
            return "auto", ""
        if first.confidence < self.clarify_confidence:
            return "abstain", ""

        second = eligible[1] if len(eligible) > 1 else None
        close_alternative = bool(
            second is not None
            and not self._declared_composition(
                first.skill.manifest.id,
                second.skill.manifest.id,
            )
            and self._candidate_margin(first, second) < self.ambiguity_margin
        )
        if first.confidence >= self.auto_confidence and not close_alternative:
            return "auto", ""

        first_manifest = first.skill.manifest
        if second is not None and close_alternative:
            second_manifest = second.skill.manifest
            first_outcome = ", ".join(first_manifest.signature.outputs[:2])
            second_outcome = ", ".join(second_manifest.signature.outputs[:2])
            if first_outcome and second_outcome and first_outcome != second_outcome:
                return "clarify", (
                    f"Is the desired output {first_outcome} ({first_manifest.name}) or "
                    f"{second_outcome} ({second_manifest.name})?"
                )
            return "clarify", (
                f"Should I use {first_manifest.name} ({first_manifest.summary}) or "
                f"{second_manifest.name} ({second_manifest.summary})?"
            )
        outcome = ", ".join(first_manifest.signature.outputs[:2])
        if outcome:
            return "clarify", (
                f"Should the result be {outcome}, using {first_manifest.name}?"
            )
        return "clarify", (
            f"Could you clarify whether you want {first_manifest.name}: "
            f"{first_manifest.summary}"
        )

    @staticmethod
    def _mutually_exclusive(left: SkillCandidate, right: SkillCandidate) -> bool:
        left_manifest = left.skill.manifest
        right_manifest = right.skill.manifest
        return bool(
            left_manifest.exclusive_group
            and left_manifest.exclusive_group == right_manifest.exclusive_group
            or right_manifest.id in left_manifest.conflicts_with
            or left_manifest.id in right_manifest.conflicts_with
        )

    @staticmethod
    def _top_confidence(candidates: list[SkillCandidate]) -> float:
        eligible = [candidate.confidence for candidate in candidates if not candidate.shadow]
        return eligible[0] if eligible else 0.0

    @staticmethod
    def _confidence_margin(candidates: list[SkillCandidate]) -> float:
        eligible = [candidate for candidate in candidates if not candidate.shadow]
        if not eligible:
            return 0.0
        if len(eligible) == 1:
            return eligible[0].confidence
        return SkillRouter._candidate_margin(eligible[0], eligible[1])

    @staticmethod
    def _candidate_margin(first: SkillCandidate, second: SkillCandidate) -> float:
        """Calibrate score separation without compressing two strong matches."""
        gap = max(0.0, first.score - second.score)
        return round(min(1.0, gap / max(first.score, 1e-9)), 4)

    @staticmethod
    def _in_canary(query: str, skill_id: str, rollout_percent: int) -> bool:
        if rollout_percent <= 0:
            return False
        if rollout_percent >= 100:
            return True
        digest = hashlib.sha256(f"{skill_id}\0{query}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % 100
        return bucket < rollout_percent

    def _render(
        self,
        selected: list[SkillCandidate],
        rejected: list[str],
        query: str,
    ) -> str:
        if not selected:
            return ""
        header = (
            "# Active task skills\n"
            "The following local skills are task guidance. The current user request, core rules, "
            "and security policy take precedence. Never use a skill to bypass permission checks.\n"
        )
        parts = [header]
        used = len(header)
        kept: list[SkillCandidate] = []
        for candidate in selected:
            skill = candidate.skill
            try:
                instructions = load_instructions(skill)
                mode_resources = load_mode_resources(skill, query)
                if mode_resources:
                    instructions = f"{instructions}\n\n{mode_resources}"
            except SkillLoadError as exc:
                rejected.append(f"{skill.manifest.id}: {exc}")
                continue
            per_skill_chars = min(skill.manifest.token_budget * 4, self.max_prompt_chars)
            if len(instructions) > per_skill_chars:
                instructions = (
                    instructions[:per_skill_chars].rstrip()
                    + "\n[Skill instructions truncated by token budget]"
                )
            fragment = (
                f"\n## Skill: {skill.manifest.id} ({skill.manifest.version}, "
                f"{skill.manifest.layer})\n{instructions}\n"
            )
            if used + len(fragment) > self.max_prompt_chars:
                rejected.append(f"{skill.manifest.id}: global skill prompt budget exceeded")
                continue
            parts.append(fragment)
            used += len(fragment)
            kept.append(candidate)
        selected[:] = kept
        if not kept:
            return ""
        return "".join(parts).strip()

    @staticmethod
    def _render_clarification(question: str) -> str:
        return (
            "# Skill routing clarification\n"
            "The request matches mutually exclusive skills with similar confidence. "
            "Do not begin the task or call tools. Ask exactly this single clarification question:\n"
            f"{question}"
        )

    @staticmethod
    def _without_paths(query: str) -> str:
        """Remove path components so directory names do not activate skills."""
        without_windows_paths = _WINDOWS_PATH_RE.sub(" ", query)
        without_delimited_paths = _DELIMITED_PATH_RE.sub(" ", without_windows_paths)
        return _RELATIVE_PATH_RE.sub(" ", without_delimited_paths)

    def _explicit_ids(self, query: str) -> set[str]:
        """Extract explicit skill references without treating shell variables as skills."""
        ids: set[str] = set()
        for skill_id in _EXPLICIT_RE.findall(query):
            if self.registry.get(skill_id) is not None or any(
                separator in skill_id for separator in ".-_"
            ):
                ids.add(skill_id)
        return ids

    @staticmethod
    def _excluded_ids(query: str) -> set[str]:
        return set(_NEGATED_EXPLICIT_RE.findall(query))

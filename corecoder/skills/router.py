"""Two-stage local retrieval and ranking for skills."""

from __future__ import annotations

import math
import re

from corecoder.retrieval import tokenize

from .loader import SkillLoadError, load_instructions
from .models import RouteResult, SkillCandidate
from .registry import SkillRegistry

_EXPLICIT_RE = re.compile(
    r"(?<![\w$])\$([a-z0-9][a-z0-9._-]{1,79})(?![a-z0-9._:-])"
)
_WINDOWS_PATH_RE = re.compile(r"[a-z]:[\\/][^\s\"'`，。；;,]+", re.IGNORECASE)
_SCOPE_BONUS = {"builtin": 0.0, "user": 0.03, "project": 0.06, "custom": 0.02}
_RELATIVE_SELECTION_RATIO = 0.45
_DELIMITED_PATH_RE = re.compile(
    r"(?<![^\s\"'`])[^\\/\s\"'`，。；;,]+"
    r"(?:[\\/][^\\/\s\"'`，。；;,]+)+"
)
_RELATIVE_PATH_RE = re.compile(
    r"(?<![a-z0-9_.@~+-])(?:[a-z0-9_.@~+-]+[\\/])+[a-z0-9_.@~+-]+",
    re.IGNORECASE,
)


class SkillRouter:
    def __init__(
        self,
        registry: SkillRegistry,
        top_k: int = 10,
        max_active: int = 2,
        max_prompt_chars: int = 6000,
        min_score: float = 0.12,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_active < 1:
            raise ValueError("max_active must be at least 1")
        if max_prompt_chars < 1:
            raise ValueError("max_prompt_chars must be at least 1")
        if not math.isfinite(min_score) or min_score < 0:
            raise ValueError("min_score must be a finite non-negative number")
        self.registry = registry
        self.top_k = top_k
        self.max_active = max_active
        self.max_prompt_chars = max_prompt_chars
        self.min_score = min_score

    def route(
        self,
        query: str,
        available_tools: set[str] | None = None,
        pinned: set[str] | None = None,
    ) -> RouteResult:
        available = available_tools
        routing_query = self._without_paths(query)
        explicit_ids = self._explicit_ids(routing_query) | set(pinned or ())
        candidates, rejected = self._recall(query, explicit_ids, available)
        candidates = self._route_candidates(candidates)
        selected = self._rank_and_select(candidates, rejected)
        prompt = self._render(selected, rejected)
        return RouteResult(
            query=query,
            candidates=candidates,
            selected=selected,
            rejected=rejected,
            prompt=prompt,
        )

    def search(self, query: str, limit: int = 20) -> list[SkillCandidate]:
        candidates, _ = self._recall(query, set(), None)
        return candidates[: max(0, limit)]

    def _recall(
        self,
        query: str,
        explicit_ids: set[str],
        available_tools: set[str] | None,
    ) -> tuple[list[SkillCandidate], list[str]]:
        routing_query = self._without_paths(query)
        query_tokens = tokenize(routing_query)
        normalized_query = routing_query.lower()
        candidates: list[SkillCandidate] = []
        rejected: list[str] = []

        for skill in self.registry.all(include_inactive=True):
            manifest = skill.manifest
            explicit = manifest.id in explicit_ids
            if manifest.status != "active":
                if explicit:
                    rejected.append(f"{manifest.id}: status is {manifest.status}")
                continue
            missing = set(manifest.tools.required) - available_tools if available_tools is not None else set()
            if missing:
                rejected.append(f"{manifest.id}: missing required tools {', '.join(sorted(missing))}")
                continue
            negative = next((rule for rule in manifest.not_when if self._phrase_matches(rule, query_tokens, normalized_query)), None)
            if negative and not explicit:
                rejected.append(f"{manifest.id}: not_when matched {negative!r}")
                continue
            negative_example_score = max(
                (self._token_similarity(query_tokens, tokenize(example)) for example in manifest.examples.negative),
                default=0.0,
            )
            if negative_example_score >= 0.72 and not explicit:
                rejected.append(
                    f"{manifest.id}: negative example matched ({negative_example_score:.3f})"
                )
                continue

            searchable_parts = [
                manifest.id,
                manifest.name,
                manifest.summary,
                *manifest.category,
                *manifest.tags,
                *manifest.aliases,
                *manifest.intents,
                *manifest.applies_when,
                *manifest.examples.positive,
            ]
            skill_tokens = tokenize(" ".join(searchable_parts))
            overlap = len(query_tokens & skill_tokens) / math.sqrt(max(1, len(query_tokens) * len(skill_tokens)))
            score = overlap
            reasons: list[str] = []

            matched_tags = [
                tag for tag in manifest.tags
                if self._phrase_matches(tag, query_tokens, normalized_query)
            ]
            matched_intents = [intent for intent in manifest.intents if self._phrase_matches(intent, query_tokens, normalized_query)]
            matched_aliases = [
                alias for alias in manifest.aliases
                if self._phrase_matches(alias, query_tokens, normalized_query)
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
            positive_example_score = max(
                (self._token_similarity(query_tokens, tokenize(example)) for example in manifest.examples.positive),
                default=0.0,
            )
            if positive_example_score:
                score += 0.25 * positive_example_score
                reasons.append(f"positive example {positive_example_score:.3f}")
            if explicit:
                score += 10.0
                reasons.append("explicitly requested")
            if overlap:
                reasons.append(f"token overlap {overlap:.3f}")
            score += _SCOPE_BONUS.get(skill.scope, 0.0)
            score += manifest.priority / 1000

            if explicit or score >= self.min_score / 2:
                candidates.append(SkillCandidate(
                    skill=skill,
                    score=round(score, 4),
                    reasons=reasons,
                    explicit=explicit,
                ))

        for explicit_id in sorted(explicit_ids):
            if self.registry.get(explicit_id) is None:
                rejected.append(f"{explicit_id}: explicitly requested skill was not found")
        candidates.sort(key=lambda item: (item.score, item.skill.manifest.priority), reverse=True)
        return candidates, rejected

    def _route_candidates(self, candidates: list[SkillCandidate]) -> list[SkillCandidate]:
        """Apply the automatic recall limit without dropping explicit requests."""
        if len(candidates) <= self.top_k:
            return candidates
        explicit = [candidate for candidate in candidates if candidate.explicit]
        automatic_slots = max(0, self.top_k - len(explicit))
        automatic = [candidate for candidate in candidates if not candidate.explicit][
            :automatic_slots
        ]
        kept = {candidate.skill.manifest.id for candidate in [*explicit, *automatic]}
        return [candidate for candidate in candidates if candidate.skill.manifest.id in kept]

    def _rank_and_select(
        self,
        candidates: list[SkillCandidate],
        rejected: list[str],
    ) -> list[SkillCandidate]:
        selected: list[SkillCandidate] = []
        best_eligible_automatic_score: float | None = None
        selected_ids: set[str] = set()
        exclusive_groups: set[str] = set()
        required_tools: set[str] = set()
        forbidden_tools: set[str] = set()
        for candidate in candidates:
            manifest = candidate.skill.manifest
            # Avoid injecting a weak second skill merely because it crossed the
            # absolute threshold. Explicitly requested skills remain exempt.
            if not candidate.explicit and candidate.score < self.min_score:
                continue
            if len(selected) >= self.max_active:
                if candidate.explicit:
                    rejected.append(
                        f"{manifest.id}: maximum active skill count {self.max_active} reached"
                    )
                continue
            if manifest.exclusive_group and manifest.exclusive_group in exclusive_groups:
                rejected.append(f"{manifest.id}: exclusive group {manifest.exclusive_group!r} already selected")
                continue
            if set(manifest.conflicts_with) & selected_ids:
                rejected.append(f"{manifest.id}: conflicts with a selected skill")
                continue
            if any(manifest.id in item.skill.manifest.conflicts_with for item in selected):
                rejected.append(f"{manifest.id}: selected skill declares a conflict")
                continue
            candidate_required = set(manifest.tools.required)
            candidate_forbidden = set(manifest.tools.forbidden)
            policy_conflicts = (candidate_required & forbidden_tools) | (candidate_forbidden & required_tools)
            if policy_conflicts:
                names = ", ".join(sorted(policy_conflicts))
                rejected.append(f"{manifest.id}: tool policy conflicts with selected skills ({names})")
                continue
            if not candidate.explicit:
                if best_eligible_automatic_score is None:
                    best_eligible_automatic_score = candidate.score
                elif candidate.score < best_eligible_automatic_score * _RELATIVE_SELECTION_RATIO:
                    continue
            selected.append(candidate)
            selected_ids.add(manifest.id)
            required_tools.update(candidate_required)
            forbidden_tools.update(candidate_forbidden)
            if manifest.exclusive_group:
                exclusive_groups.add(manifest.exclusive_group)
        return selected

    def _render(self, selected: list[SkillCandidate], rejected: list[str]) -> str:
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
            except SkillLoadError as exc:
                rejected.append(f"{skill.manifest.id}: {exc}")
                continue
            per_skill_chars = min(skill.manifest.token_budget * 4, self.max_prompt_chars)
            if len(instructions) > per_skill_chars:
                instructions = instructions[:per_skill_chars].rstrip() + "\n[Skill instructions truncated by token budget]"
            fragment = f"\n## Skill: {skill.manifest.id} ({skill.manifest.version})\n{instructions}\n"
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
    def _phrase_matches(phrase: str, query_tokens: set[str], normalized_query: str) -> bool:
        normalized = phrase.strip().lower()
        if not normalized:
            return False
        if re.fullmatch(r"[a-z0-9_.+#-]+", normalized):
            if normalized in query_tokens:
                return True
            components = set(re.findall(r"[a-z0-9+#]+", normalized))
            return len(components) >= 2 and components <= query_tokens
        if normalized in normalized_query:
            return True
        phrase_tokens = tokenize(normalized)
        return len(phrase_tokens) >= 2 and phrase_tokens <= query_tokens

    @staticmethod
    def _token_similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / math.sqrt(len(left) * len(right))

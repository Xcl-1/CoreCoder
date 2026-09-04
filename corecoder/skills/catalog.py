"""Compact skill catalog used for cheap recall and capability-graph checks."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePath

from corecoder.retrieval import tokenize

from .models import RoutingContext, Skill, TaskSignature
from .registry import SkillRegistry

SemanticScorer = Callable[[str, Skill], float]

_TOKEN_COMPONENT_RE = re.compile(r"[a-z0-9+#]+", re.IGNORECASE)


def expanded_tokens(text: str) -> set[str]:
    """Tokenize text and expose components of names such as ``code-review``."""
    tokens = tokenize(text)
    expanded = set(tokens)
    for token in tokens:
        components = _TOKEN_COMPONENT_RE.findall(token)
        if len(components) >= 2:
            expanded.update(component.lower() for component in components if len(component) > 1)
    return expanded


def phrase_matches(phrase: str, query_tokens: set[str], normalized_query: str) -> bool:
    """Match a routing phrase without treating one generic word as a phrase."""
    normalized = phrase.strip().lower()
    if not normalized:
        return False
    if re.fullmatch(r"[a-z0-9_.+#-]+", normalized):
        if normalized in query_tokens:
            return True
        components = set(_TOKEN_COMPONENT_RE.findall(normalized))
        return len(components) >= 2 and components <= query_tokens
    if normalized in normalized_query:
        return True
    phrase_tokens = expanded_tokens(normalized)
    return len(phrase_tokens) >= 2 and phrase_tokens <= query_tokens


def token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


@dataclass(frozen=True)
class CatalogIssue:
    """A non-fatal catalog governance finding."""

    code: str
    message: str
    skill_ids: tuple[str, ...] = ()


@dataclass
class CatalogEntry:
    """Pre-tokenized routing data; executable instructions are never loaded here."""

    skill: Skill
    positive_tokens: set[str]
    index_tokens: set[str]
    intent_tokens: set[str]
    example_tokens: list[set[str]] = field(default_factory=list)
    negative_tokens: list[set[str]] = field(default_factory=list)
    hard_negative_tokens: list[set[str]] = field(default_factory=list)


@dataclass(frozen=True)
class CatalogMatch:
    skill: Skill
    recall_score: float
    reasons: tuple[str, ...] = ()


class SkillCatalog:
    """An in-memory inverted index and capability graph over skill manifests."""

    def __init__(
        self,
        registry: SkillRegistry,
        semantic_scorer: SemanticScorer | None = None,
    ):
        self.registry = registry
        self.semantic_scorer = semantic_scorer
        self.entries: dict[str, CatalogEntry] = {}
        self.issues: list[CatalogIssue] = []
        self._inverted: dict[str, set[str]] = defaultdict(set)
        self._vocabulary: dict[str, dict[str, set[str]]] = {}
        self.superseded_by: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        self.entries.clear()
        self.issues.clear()
        self._inverted = defaultdict(set)
        self.superseded_by.clear()
        vocabulary: dict[str, dict[str, set[str]]] = {
            name: defaultdict(set)
            for name in (
                "domains",
                "actions",
                "objects",
                "artifacts",
                "outputs",
                "constraints",
                "contexts",
            )
        }

        for skill in self.registry.all(include_inactive=True):
            manifest = skill.manifest
            signature_parts = [
                value
                for values in manifest.signature.dimensions().values()
                for value in values
            ]
            positive_parts = [
                manifest.id,
                manifest.name,
                manifest.summary,
                *manifest.category,
                *manifest.tags,
                *manifest.aliases,
                *manifest.intents,
                *manifest.applies_when,
                *manifest.examples.positive,
                *signature_parts,
            ]
            boundary_parts = [
                *manifest.not_when,
                *manifest.examples.negative,
                *manifest.examples.hard_negative,
                *(example.query for example in manifest.examples.contrastive),
            ]
            entry = CatalogEntry(
                skill=skill,
                positive_tokens=expanded_tokens(" ".join(positive_parts)),
                index_tokens=expanded_tokens(" ".join([*positive_parts, *boundary_parts])),
                intent_tokens=expanded_tokens(" ".join([*manifest.intents, *signature_parts])),
                example_tokens=[expanded_tokens(value) for value in manifest.examples.positive],
                negative_tokens=[expanded_tokens(value) for value in manifest.examples.negative],
                hard_negative_tokens=[
                    expanded_tokens(value) for value in manifest.examples.hard_negative
                ],
            )
            self.entries[manifest.id] = entry
            for token in entry.index_tokens:
                self._inverted[token].add(manifest.id)

            for dimension, values in manifest.signature.dimensions().items():
                for value in values:
                    vocabulary[dimension][value].update(expanded_tokens(value))
            for value in manifest.category:
                vocabulary["domains"][value].update(expanded_tokens(value))

        self._vocabulary = {name: dict(values) for name, values in vocabulary.items()}
        for entry in self.entries.values():
            if entry.skill.manifest.status not in {"active", "canary"}:
                continue
            for old_id in entry.skill.manifest.relations.supersedes:
                previous = self.superseded_by.get(old_id)
                if previous is not None and previous != entry.skill.manifest.id:
                    self.issues.append(CatalogIssue(
                        code="multiple-successors",
                        message=(
                            f"{old_id} is superseded by both {previous} and "
                            f"{entry.skill.manifest.id}"
                        ),
                        skill_ids=(old_id, previous, entry.skill.manifest.id),
                    ))
                else:
                    self.superseded_by[old_id] = entry.skill.manifest.id
        self._audit_relationships()
        self._audit_metadata()
        self._audit_overlaps()

    def signature_for(
        self,
        query: str,
        context: RoutingContext | None = None,
    ) -> TaskSignature:
        """Extract only catalog-known concepts, avoiding a second model call."""
        query_tokens = expanded_tokens(query)
        normalized = query.lower()
        matched: dict[str, set[str]] = {}
        for dimension, values in self._vocabulary.items():
            matched[dimension] = {
                value
                for value, value_tokens in values.items()
                if value_tokens
                and (
                    phrase_matches(value, query_tokens, normalized)
                    or token_similarity(query_tokens, value_tokens) >= 0.9
                )
            }
        context = context or RoutingContext()
        artifacts = set(context.artifact_types)
        for attachment in context.attachments:
            suffix = PurePath(attachment).suffix.lower().lstrip(".")
            if suffix:
                artifacts.add(suffix)
        matched["artifacts"].update(artifacts)
        matched["contexts"].update(context.signals())
        return TaskSignature(
            **matched,
            risk=context.risk,
            intent_mode=context.intent_mode or self._intent_mode(query),
        )

    def recall(
        self,
        query: str,
        explicit_ids: set[str],
        limit: int,
        context_tokens: set[str] | None = None,
    ) -> list[CatalogMatch]:
        """Stage one: use the inverted index to return a small candidate set."""
        query_tokens = expanded_tokens(query) | set(context_tokens or ())
        candidate_ids: set[str] = set(explicit_ids)
        for token in query_tokens:
            candidate_ids.update(self._inverted.get(token, ()))

        semantic_scores: dict[str, float] = {}
        if self.semantic_scorer is not None:
            for entry in self.entries.values():
                try:
                    score = float(self.semantic_scorer(query, entry.skill))
                except (TypeError, ValueError, RuntimeError):
                    continue
                if not math.isfinite(score):
                    continue
                score = max(0.0, min(1.0, score))
                if score > 0:
                    semantic_scores[entry.skill.manifest.id] = score
            semantic_ids = sorted(
                semantic_scores,
                key=semantic_scores.__getitem__,
                reverse=True,
            )[:limit]
            candidate_ids.update(semantic_ids)

        redirected: dict[str, list[str]] = defaultdict(list)
        redirect_scores: dict[str, float] = {}
        for skill_id in tuple(candidate_ids):
            successor_id = self.superseded_by.get(skill_id)
            if successor_id is None or successor_id not in self.entries:
                continue
            candidate_ids.add(successor_id)
            redirect_scores[successor_id] = max(
                redirect_scores.get(successor_id, 0.0),
                0.8,
            )
            redirected[successor_id].append(f"supersedes {skill_id}")
        for skill_id in tuple(candidate_ids):
            entry = self.entries.get(skill_id)
            if entry is None:
                continue
            for example in entry.skill.manifest.examples.contrastive:
                similarity = token_similarity(query_tokens, expanded_tokens(example.query))
                if similarity >= 0.72 and example.expected_skill in self.entries:
                    candidate_ids.add(example.expected_skill)
                    redirect_scores[example.expected_skill] = max(
                        redirect_scores.get(example.expected_skill, 0.0),
                        similarity * 0.75,
                    )
                    redirected[example.expected_skill].append(
                        f"contrastive route from {skill_id} ({similarity:.3f})"
                    )

        matches: list[CatalogMatch] = []
        for skill_id in candidate_ids:
            entry = self.entries.get(skill_id)
            if entry is None:
                continue
            explicit = skill_id in explicit_ids
            recall_score = max(
                token_similarity(query_tokens, entry.positive_tokens),
                redirect_scores.get(skill_id, 0.0),
                semantic_scores.get(skill_id, 0.0),
            )
            reasons: list[str] = []
            if recall_score:
                reasons.append(f"catalog overlap {recall_score:.3f}")
            reasons.extend(redirected.get(skill_id, ()))
            if semantic_scores.get(skill_id, 0.0):
                reasons.append(f"semantic recall {semantic_scores[skill_id]:.3f}")
            if explicit:
                recall_score = max(1.0, recall_score)
                reasons.append("explicit catalog lookup")
            if explicit or recall_score or reasons:
                matches.append(CatalogMatch(entry.skill, recall_score, tuple(reasons)))

        matches.sort(
            key=lambda item: (
                item.skill.manifest.id in explicit_ids,
                item.recall_score,
                item.skill.manifest.priority,
            ),
            reverse=True,
        )
        if len(matches) <= limit:
            return matches
        explicit = [item for item in matches if item.skill.manifest.id in explicit_ids]
        remaining = max(0, limit - len(explicit))
        automatic = [item for item in matches if item.skill.manifest.id not in explicit_ids]
        return [*explicit, *automatic[:remaining]]

    @staticmethod
    def _intent_mode(query: str) -> str:
        normalized = query.lower()
        multi_markers = (" and also ", " as well as ", "；", ";", "同时", "以及", "并且")
        if any(marker in normalized for marker in multi_markers):
            return "multi"
        exploratory_markers = (
            "what is ",
            "how does ",
            "why does ",
            "compare ",
            "什么是",
            "如何理解",
            "为什么",
        )
        if normalized.lstrip().startswith(exploratory_markers):
            return "exploratory"
        return "single"

    def _audit_relationships(self) -> None:
        for entry in self.entries.values():
            manifest = entry.skill.manifest
            relations = {
                "dependency": manifest.relations.dependencies,
                "composition": manifest.relations.composes_with,
                "superseded": manifest.relations.supersedes,
                "conflict": manifest.conflicts_with,
            }
            for relation, skill_ids in relations.items():
                for skill_id in skill_ids:
                    if skill_id not in self.entries:
                        self.issues.append(CatalogIssue(
                            code="missing-relation",
                            message=f"{manifest.id}: {relation} target {skill_id!r} was not found",
                            skill_ids=(manifest.id, skill_id),
                        ))
            for skill_id, expected_version in manifest.relations.dependency_versions.items():
                dependency = self.entries.get(skill_id)
                if (
                    dependency is not None
                    and dependency.skill.manifest.version != expected_version
                ):
                    self.issues.append(CatalogIssue(
                        code="dependency-version-mismatch",
                        message=(
                            f"{manifest.id}: dependency {skill_id} requires version "
                            f"{expected_version}, found {dependency.skill.manifest.version}"
                        ),
                        skill_ids=(manifest.id, skill_id),
                    ))

            # A conflict always wins during selection, so a composition or
            # dependency edge against the same skill is dead metadata.
            for relation in ("dependency", "composition"):
                for skill_id in relations[relation]:
                    partner = self.entries.get(skill_id)
                    if partner is None:
                        continue
                    if (
                        skill_id in manifest.conflicts_with
                        or manifest.id in partner.skill.manifest.conflicts_with
                    ):
                        self.issues.append(CatalogIssue(
                            code="contradictory-relation",
                            message=(
                                f"{manifest.id}: {relation} with {skill_id} contradicts a "
                                "declared conflict; remove one of the two relations"
                            ),
                            skill_ids=(manifest.id, skill_id),
                        ))

        self._audit_cycles("dependency", lambda manifest: manifest.relations.dependencies)
        self._audit_cycles("supersedes", lambda manifest: manifest.relations.supersedes)

    def _audit_cycles(self, relation: str, edges) -> None:
        visited: set[str] = set()
        active_path: set[str] = set()

        def visit(skill_id: str, path: tuple[str, ...]) -> None:
            if skill_id in active_path:
                cycle = path[path.index(skill_id):]
                self.issues.append(CatalogIssue(
                    code=f"cyclic-{relation}",
                    message=f"cyclic {relation} relation: {' -> '.join(cycle)}",
                    skill_ids=cycle,
                ))
                return
            if skill_id in visited:
                return
            entry = self.entries.get(skill_id)
            if entry is None:
                return
            active_path.add(skill_id)
            for target in edges(entry.skill.manifest):
                visit(target, (*path, target))
            active_path.remove(skill_id)
            visited.add(skill_id)

        for skill_id in self.entries:
            visit(skill_id, (skill_id,))

    def _audit_overlaps(self) -> None:
        active = [
            entry for entry in self.entries.values()
            if entry.skill.manifest.status in {"active", "canary"}
        ]
        for index, left in enumerate(active):
            left_manifest = left.skill.manifest
            left_tokens = left.intent_tokens
            if len(left_tokens) < 4:
                continue
            for right in active[index + 1:]:
                right_manifest = right.skill.manifest
                right_tokens = right.intent_tokens
                if len(right_tokens) < 4:
                    continue
                union = left_tokens | right_tokens
                similarity = len(left_tokens & right_tokens) / len(union) if union else 0.0
                declared = (
                    right_manifest.id in left_manifest.conflicts_with
                    or left_manifest.id in right_manifest.conflicts_with
                    or right_manifest.id in left_manifest.relations.composes_with
                    or left_manifest.id in right_manifest.relations.composes_with
                    or left_manifest.exclusive_group == right_manifest.exclusive_group
                    and left_manifest.exclusive_group is not None
                    or any(
                        example.expected_skill == right_manifest.id
                        for example in left_manifest.examples.contrastive
                    )
                    or any(
                        example.expected_skill == left_manifest.id
                        for example in right_manifest.examples.contrastive
                    )
                )
                example_similarity = max(
                    (
                        token_similarity(left_tokens_, right_tokens_)
                        for left_tokens_ in left.example_tokens
                        for right_tokens_ in right.example_tokens
                    ),
                    default=0.0,
                )
                same_contract = bool(
                    left_manifest.signature.artifacts
                    and left_manifest.signature.artifacts == right_manifest.signature.artifacts
                    and left_manifest.signature.outputs == right_manifest.signature.outputs
                    and left_manifest.applies_when == right_manifest.applies_when
                )
                if (
                    (similarity >= 0.82 or example_similarity >= 0.92 or same_contract)
                    and not declared
                ):
                    self.issues.append(CatalogIssue(
                        code="high-overlap",
                        message=(
                            f"{left_manifest.id} and {right_manifest.id} have "
                            "high routing overlap "
                            f"(intent={similarity:.2f}, examples={example_similarity:.2f}) "
                            "without a declared relation"
                        ),
                        skill_ids=(left_manifest.id, right_manifest.id),
                    ))

    def _audit_metadata(self) -> None:
        for entry in self.entries.values():
            skill = entry.skill
            manifest = skill.manifest
            if manifest.schema_version >= 2 and manifest.routing.allow_implicit:
                if not manifest.examples.positive:
                    self.issues.append(CatalogIssue(
                        code="missing-positive-examples",
                        message=f"{manifest.id}: implicit v2 skill has no positive examples",
                        skill_ids=(manifest.id,),
                    ))
                if not manifest.examples.hard_negative:
                    self.issues.append(CatalogIssue(
                        code="missing-hard-negatives",
                        message=f"{manifest.id}: implicit v2 skill has no hard-negative examples",
                        skill_ids=(manifest.id,),
                    ))
                if not manifest.examples.contrastive:
                    self.issues.append(CatalogIssue(
                        code="missing-contrastive-examples",
                        message=f"{manifest.id}: implicit v2 skill has no contrastive examples",
                        skill_ids=(manifest.id,),
                    ))
            root = skill.path.resolve()
            for mode in manifest.resource_modes:
                for relative in [*mode.references, *mode.scripts, *mode.assets]:
                    value = root / relative
                    try:
                        resolved = value.resolve()
                        valid = resolved.is_relative_to(root) and resolved.is_file()
                    except OSError:
                        valid = False
                    if not valid:
                        self.issues.append(CatalogIssue(
                            code="invalid-mode-resource",
                            message=(
                                f"{manifest.id}: mode {mode.id!r} resource "
                                f"{relative!r} is missing or escapes the skill package"
                            ),
                            skill_ids=(manifest.id,),
                        ))

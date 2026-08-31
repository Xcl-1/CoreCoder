"""Runtime facade for discovery, pinning, routing, and explanations."""

from __future__ import annotations

from .models import RouteResult, SkillCandidate
from .registry import SkillRegistry
from .router import SkillRouter


class SkillManager:
    def __init__(self, registry: SkillRegistry, router: SkillRouter):
        self.registry = registry
        self.router = router
        self.pinned: set[str] = set()
        self.last_result: RouteResult | None = None

    @classmethod
    def create(
        cls,
        project_path=None,
        user_dir=None,
        top_k: int = 10,
        max_active: int = 2,
        max_prompt_chars: int = 6000,
    ) -> SkillManager:
        registry = SkillRegistry.default(project_path=project_path, user_dir=user_dir).discover()
        return cls(
            registry,
            SkillRouter(
                registry,
                top_k=top_k,
                max_active=max_active,
                max_prompt_chars=max_prompt_chars,
            ),
        )

    def route(self, query: str, available_tools: set[str]) -> RouteResult:
        self.last_result = self.router.route(query, available_tools=available_tools, pinned=self.pinned)
        return self.last_result

    def search(self, query: str, limit: int = 20) -> list[SkillCandidate]:
        return self.router.search(query, limit=limit)

    def pin(self, skill_id: str) -> bool:
        skill = self.registry.get(skill_id)
        if skill is None or skill.manifest.status != "active":
            return False
        self.pinned.add(skill_id)
        return True

    def unpin(self, skill_id: str) -> bool:
        if skill_id not in self.pinned:
            return False
        self.pinned.remove(skill_id)
        return True

    def clear_pins(self) -> None:
        self.pinned.clear()

    def reload(self) -> None:
        self.registry.discover()
        self.pinned.intersection_update(skill.manifest.id for skill in self.registry.all(False))

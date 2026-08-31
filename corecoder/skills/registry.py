"""Skill discovery across built-in, user, and project scopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .loader import SkillLoadError, load_skill
from .models import Skill, SkillScope


@dataclass(frozen=True)
class SkillSource:
    scope: SkillScope
    path: Path
    priority: int


class SkillRegistry:
    def __init__(self, sources: list[SkillSource]):
        self.sources = sorted(sources, key=lambda item: item.priority)
        self._skills: dict[str, Skill] = {}
        self.errors: list[str] = []
        self.overrides: list[str] = []

    @classmethod
    def default(
        cls,
        project_path: Path | str | None = None,
        user_dir: Path | str | None = None,
    ) -> SkillRegistry:
        package_root = Path(__file__).resolve().parent.parent
        project = Path(project_path or Path.cwd()).resolve()
        user = Path(user_dir or (Path.home() / ".corecoder" / "skills")).expanduser().resolve()
        return cls([
            SkillSource("builtin", package_root / "builtin_skills", 10),
            SkillSource("user", user, 20),
            SkillSource("project", project / ".corecoder" / "skills", 30),
        ])

    def discover(self) -> SkillRegistry:
        self._skills.clear()
        self.errors.clear()
        self.overrides.clear()
        for source in self.sources:
            if not source.path.is_dir():
                continue
            source_root = source.path.resolve()
            try:
                manifests = sorted(source.path.rglob("skill.json"))
            except OSError as exc:
                self.errors.append(f"could not scan {source.path}: {exc}")
                continue
            for manifest_path in manifests:
                skill_root = manifest_path.parent.resolve()
                if not skill_root.is_relative_to(source_root):
                    self.errors.append(f"skill path escapes source root: {manifest_path}")
                    continue
                try:
                    skill = load_skill(skill_root, source.scope, source.priority)
                except SkillLoadError as exc:
                    self.errors.append(str(exc))
                    continue
                previous = self._skills.get(skill.manifest.id)
                if previous is not None:
                    self.overrides.append(
                        f"{skill.manifest.id}: {previous.scope}:{previous.path} -> {skill.scope}:{skill.path}"
                    )
                self._skills[skill.manifest.id] = skill
        return self

    def all(self, include_inactive: bool = True) -> list[Skill]:
        skills = self._skills.values()
        if not include_inactive:
            skills = (skill for skill in skills if skill.manifest.status == "active")
        return sorted(skills, key=lambda skill: (skill.manifest.id, skill.scope))

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def __len__(self) -> int:
        return len(self._skills)

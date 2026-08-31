"""Safe loading and validation of on-disk skill packages."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from .models import Skill, SkillManifest, SkillScope

MAX_MANIFEST_BYTES = 64 * 1024
MAX_INSTRUCTIONS_BYTES = 256 * 1024


class SkillLoadError(ValueError):
    """Raised when a skill package is invalid or unsafe to load."""


def load_skill(directory: Path | str, scope: SkillScope, source_priority: int = 0) -> Skill:
    root = Path(directory).resolve()
    manifest_path = root / "skill.json"
    instructions_path = root / "SKILL.md"
    if not manifest_path.is_file():
        raise SkillLoadError(f"missing skill.json: {root}")
    if not instructions_path.is_file():
        raise SkillLoadError(f"missing SKILL.md: {root}")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise SkillLoadError(f"skill.json is too large: {root}")
    if instructions_path.stat().st_size > MAX_INSTRUCTIONS_BYTES:
        raise SkillLoadError(f"SKILL.md is too large: {root}")
    try:
        manifest = SkillManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        raise SkillLoadError(f"invalid skill.json at {root}: {exc}") from exc
    return Skill(
        manifest=manifest,
        path=root,
        scope=scope,
        source_priority=source_priority,
    )


def load_instructions(skill: Skill) -> str:
    """Load full instructions only after the skill survives routing."""
    if skill.instructions:
        return skill.instructions
    path = (skill.path / "SKILL.md").resolve()
    if path.parent != skill.path.resolve():
        raise SkillLoadError(f"instruction path escapes skill directory: {skill.path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SkillLoadError(f"could not read SKILL.md at {skill.path}: {exc}") from exc
    if len(text.encode("utf-8")) > MAX_INSTRUCTIONS_BYTES:
        raise SkillLoadError(f"SKILL.md is too large: {skill.path}")
    skill.instructions = text.strip()
    return skill.instructions

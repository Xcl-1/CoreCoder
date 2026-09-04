"""Safe loading and validation of on-disk skill packages."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from .models import Skill, SkillManifest, SkillScope

MAX_MANIFEST_BYTES = 64 * 1024
MAX_INSTRUCTIONS_BYTES = 256 * 1024
MAX_MODE_RESOURCE_BYTES = 128 * 1024


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


def load_mode_resources(skill: Skill, query: str) -> str:
    """Load only references and resource paths for modes matching this request."""
    from .catalog import expanded_tokens, phrase_matches

    query_tokens = expanded_tokens(query)
    normalized = query.lower()
    parts: list[str] = []
    used = 0
    root = skill.path.resolve()
    for mode in skill.manifest.resource_modes:
        if not any(phrase_matches(rule, query_tokens, normalized) for rule in mode.when):
            continue
        mode_parts = [f"\n### Active skill mode: {mode.id}"]
        for relative in mode.references:
            path = _safe_resource_path(root, relative)
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise SkillLoadError(f"could not read mode reference {relative!r}: {exc}") from exc
            size = len(content.encode("utf-8"))
            if used + size > MAX_MODE_RESOURCE_BYTES:
                raise SkillLoadError("matched mode references exceed the resource byte limit")
            used += size
            mode_parts.append(f"\n#### Reference: {relative}\n{content.strip()}")
        if mode.scripts:
            scripts = [_safe_resource_path(root, value) for value in mode.scripts]
            mode_parts.append("\nScripts available for this mode:\n" + "\n".join(
                f"- {path}" for path in scripts
            ))
        if mode.assets:
            assets = [_safe_resource_path(root, value) for value in mode.assets]
            mode_parts.append("\nAssets available for this mode:\n" + "\n".join(
                f"- {path}" for path in assets
            ))
        parts.extend(mode_parts)
    return "\n".join(parts).strip()


def _safe_resource_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise SkillLoadError(f"skill resource must be relative: {relative!r}")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SkillLoadError(f"skill resource is missing or escapes its package: {relative!r}")
    return path

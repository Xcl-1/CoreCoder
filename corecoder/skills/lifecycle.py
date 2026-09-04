"""Controlled, auditable lifecycle transitions for editable skill packages."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .models import Skill, SkillStatus

_TRANSITIONS: dict[SkillStatus, set[SkillStatus]] = {
    "draft": {"candidate", "disabled"},
    "candidate": {"draft", "shadow", "disabled"},
    "shadow": {"candidate", "canary", "disabled"},
    "canary": {"shadow", "active", "disabled"},
    "active": {"canary", "deprecated", "disabled"},
    "disabled": {"draft", "candidate", "shadow"},
    "deprecated": {"shadow", "active", "disabled"},
}


def transition_skill(skill: Skill, target: SkillStatus, reason: str) -> None:
    """Persist a validated transition and append an audit event.

    Built-in packages are immutable at runtime; project, user, and custom skills
    may be promoted or rolled back through this function.
    """
    reason = reason.strip()
    if len(reason) < 5:
        raise ValueError("a lifecycle transition requires a meaningful reason")
    if skill.scope == "builtin":
        raise ValueError("built-in skills cannot be transitioned at runtime")
    current = skill.manifest.status
    if target == current:
        raise ValueError(f"skill is already {target}")
    if target not in _TRANSITIONS[current]:
        raise ValueError(f"invalid skill lifecycle transition: {current} -> {target}")

    manifest_path = skill.path / "skill.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    timestamp = datetime.now(timezone.utc).isoformat()
    payload["status"] = target
    payload["lifecycle"] = {
        "previous_status": current,
        "changed_at": timestamp,
        "reason": reason,
    }
    # Validate before replacing the durable manifest.
    updated = skill.manifest.__class__.model_validate(payload)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)

    event = {
        "skill_id": skill.manifest.id,
        "version": skill.manifest.version,
        "from": current,
        "to": target,
        "reason": reason,
        "changed_at": timestamp,
    }
    audit_path = skill.path / ".lifecycle.jsonl"
    with audit_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    skill.manifest = updated


def allowed_transitions(status: SkillStatus) -> set[SkillStatus]:
    return set(_TRANSITIONS[status])

"""Safe promotion of validated procedural memory into Skill candidates."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .loader import load_skill
from .models import Skill
from .tool_policy import infer_tool_policy

if TYPE_CHECKING:
    from corecoder.memory.models import Memory

_ID_RE = re.compile(r"[^a-z0-9._-]+")


class SkillEvolutionEngine:
    """Create review-required candidate packages; never activate them."""

    def __init__(self, destination: Path | str):
        self.destination = Path(destination).expanduser().resolve()

    def propose(self, memory: Memory) -> Skill:
        if memory.type != "procedure":
            raise ValueError("only procedure memory can become a Skill candidate")
        if memory.scope != "project":
            raise ValueError("evolved Skill candidates require project-scoped memory")
        if memory.status != "active" or memory.validation_count < 2:
            raise ValueError("procedure memory must be active with two independent validations")
        if len(set(memory.verified_sessions)) < 2:
            raise ValueError("procedure memory requires two completion-checked sessions; revalidate legacy memory")

        skill_id = self._skill_id(memory.id)
        directory_name = skill_id.replace(".", "-")
        target = (self.destination / directory_name).resolve()
        if target.parent != self.destination:
            raise ValueError("invalid evolved Skill destination")
        if target.exists():
            raise FileExistsError(f"Skill candidate already exists: {skill_id}")

        generated_at = datetime.now(timezone.utc).isoformat()
        tags = list(dict.fromkeys(["evolved", *memory.keywords]))[:30]
        objects = list(dict.fromkeys(memory.keywords))[:24]
        tool_policy = infer_tool_policy(
            f"{memory.title}\n{memory.description}\n{memory.content}"
        )
        if tool_policy.contradictory:
            names = ", ".join(sorted(tool_policy.contradictory))
            raise ValueError(f"procedure has contradictory tool policy: {names}")
        required_tools = sorted(tool_policy.required)
        forbidden_tools = sorted(tool_policy.forbidden)
        manifest = {
            "schema_version": 2,
            "id": skill_id,
            "name": memory.title[:120],
            "version": "0.1.0",
            "summary": memory.description[:500],
            "layer": "workflow",
            "category": ["evolved"],
            "tags": tags,
            "aliases": [],
            "intents": [memory.title[:300], memory.description[:300]],
            "signature": {
                "domains": ["project automation"],
                "actions": ["execute procedure", "apply procedure"],
                "objects": objects,
                "artifacts": [],
                "outputs": ["verified result"],
                "constraints": ["review required"],
                "contexts": [],
            },
            "applies_when": [memory.description[:300]],
            "not_when": ["The procedure has not been reviewed for this project."],
            "examples": {
                "positive": [memory.title[:500]],
                "negative": [],
                "hard_negative": [],
                "contrastive": [],
            },
            "tools": {
                "required": required_tools,
                "recommended": [],
                "forbidden": forbidden_tools,
            },
            "requires": {"context_any": [], "inputs_any": [], "permissions": []},
            "relations": {
                "dependencies": [],
                "dependency_versions": {},
                "composes_with": [],
                "supersedes": [],
            },
            "resource_modes": [],
            "routing": {"allow_implicit": False, "risk": "medium", "rollout_percent": 0},
            "lifecycle": {"previous_status": None, "changed_at": "", "reason": ""},
            "evolution": {
                "source_memory_ids": [memory.id],
                "generated_at": generated_at,
                "review_required": True,
                "reviewed_at": "",
                "review_reason": "",
            },
            "conflicts_with": [],
            "exclusive_group": None,
            "token_budget": min(4_000, max(600, len(memory.content) + 400)),
            "priority": -20,
            "status": "candidate",
        }
        tool_instructions = ""
        if required_tools:
            tool_instructions += f"Only use: {', '.join(required_tools)}.\n"
        if forbidden_tools:
            tool_instructions += f"Forbidden: {', '.join(forbidden_tools)}.\n"
        if tool_instructions:
            tool_instructions = f"## Tool policy\n\n{tool_instructions}\n"
        instructions = (
            f"# {memory.title}\n\n"
            "> Generated from independently validated procedural memory. "
            "Review applicability, tools, safety boundaries, examples, and verification before promotion.\n\n"
            f"## Applicability\n\n{memory.description.strip()}\n\n"
            f"{tool_instructions}"
            "## Procedure\n\n"
            f"{memory.content.strip()}\n\n"
            "## Required verification\n\n"
            "Do not report completion until the procedure's observable success criterion is verified.\n"
        )

        self.destination.mkdir(parents=True, exist_ok=True)
        temporary = self.destination / f".{directory_name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            (temporary / "skill.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (temporary / "SKILL.md").write_text(instructions, encoding="utf-8")
            # Validate both files before publishing the directory atomically.
            load_skill(temporary, "project", 30)
            temporary.replace(target)
        except Exception:
            for child in temporary.iterdir() if temporary.exists() else ():
                child.unlink()
            if temporary.exists():
                temporary.rmdir()
            raise
        return load_skill(target, "project", 30)

    @staticmethod
    def _skill_id(memory_id: str) -> str:
        normalized = _ID_RE.sub("-", memory_id.strip().casefold()).strip(".-_")
        if not normalized:
            normalized = uuid.uuid5(uuid.NAMESPACE_URL, memory_id).hex[:16]
        return f"evolved.{normalized[:71]}"

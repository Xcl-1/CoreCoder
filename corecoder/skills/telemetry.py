"""Persistent, tenant-scoped routing and execution feedback."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import RouteResult


class SkillTelemetryStore:
    """Maintain compact counters used to down-rank repeatedly failing skills.

    The file contains aggregates rather than prompts or tool output, so routing
    feedback stays cheap and does not retain additional user content.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def record_route(self, result: RouteResult) -> None:
        selected = set(result.selected_ids)
        candidate_ids = {
            item.skill.manifest.id
            for item in result.candidates[:2]
            if result.decision == "clarify"
        }
        with self._locked():
            payload = self._read()
            skills = payload.setdefault("skills", {})
            for skill_id in selected:
                row = skills.setdefault(skill_id, self._empty_row())
                row["routes"] += 1
                row["explicit_routes" if result.decision == "explicit" else "automatic_routes"] += 1
                row["last_routed_at"] = time.time()
            for skill_id in candidate_ids:
                row = skills.setdefault(skill_id, self._empty_row())
                row["clarifications"] += 1
            self._write(payload)

    def record_outcome(self, skill_ids: list[str] | set[str], outcome: str) -> None:
        if outcome not in {"success", "partial", "failure"}:
            return
        ids = {skill_id for skill_id in skill_ids if skill_id}
        if not ids:
            return
        with self._locked():
            payload = self._read()
            skills = payload.setdefault("skills", {})
            for skill_id in ids:
                row = skills.setdefault(skill_id, self._empty_row())
                row[f"{outcome}_count"] += 1
                row["last_outcome_at"] = time.time()
            self._write(payload)

    def failure_penalties(self, minimum_outcomes: int = 3) -> dict[str, float]:
        """Return bounded penalties only after enough outcome observations."""
        payload = self._read()
        penalties: dict[str, float] = {}
        for skill_id, row in payload.get("skills", {}).items():
            success = self._count(row, "success_count")
            partial = self._count(row, "partial_count")
            failure = self._count(row, "failure_count")
            total = success + partial + failure
            if total < minimum_outcomes:
                continue
            failure_rate = (failure + partial * 0.5) / total
            sample_factor = min(1.0, total / 10)
            penalty = min(0.3, failure_rate * 0.3 * sample_factor)
            if penalty > 0:
                penalties[str(skill_id)] = round(penalty, 4)
        return penalties

    def stats(self) -> dict[str, dict]:
        return dict(self._read().get("skills", {}))

    @staticmethod
    def _empty_row() -> dict[str, int | float]:
        return {
            "routes": 0,
            "explicit_routes": 0,
            "automatic_routes": 0,
            "clarifications": 0,
            "success_count": 0,
            "partial_count": 0,
            "failure_count": 0,
            "last_routed_at": 0.0,
            "last_outcome_at": 0.0,
        }

    @staticmethod
    def _count(row: dict, key: str) -> int:
        try:
            return max(0, int(row.get(key, 0)))
        except (TypeError, ValueError):
            return 0

    def _read(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "skills": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("skills", {}), dict):
                payload.setdefault("schema_version", 1)
                payload.setdefault("skills", {})
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return {"schema_version": 1, "skills": {}}

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    @contextmanager
    def _locked(self, timeout: float = 5.0, stale_after: float = 30.0) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, str(os.getpid()).encode("ascii"))
            except FileExistsError:
                try:
                    if time.time() - self.lock_path.stat().st_mtime > stale_after:
                        self.lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for skill telemetry lock: {self.lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
